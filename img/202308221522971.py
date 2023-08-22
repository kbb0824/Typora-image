# encoding=utf8
import json
import os
import time

import graphlearn as gl
import tensorflow as tf
from prada_interface.base_algorithm import BaseAlgorithm

from feature_encoder import WideNDeepEncoder
from gl_client import G2GraphLearnClient


def unsupervised_softmax_cross_entropy_loss(src_emb,
                                            pos_emb,
                                            neg_emb,
                                            temperature=1.0):
    """Sigmoid cross entropy loss for unsuperviesd model.
    Args:
      src_emb: tensor with shape [batch_size, dim]
      pos_emb: tensor with shape [batch_size, dim]
      neg_emb: tensor with shape [batch_size * neg_num, dim]
    Returns:
      loss
    """
    pos_sim = tf.reduce_sum(tf.multiply(src_emb, pos_emb), axis=-1, keep_dims=True)
    neg_sim = tf.matmul(src_emb, tf.transpose(neg_emb))
    pos_label = tf.ones_like(pos_sim)
    neg_label = tf.zeros_like(neg_sim)

    logit = tf.nn.softmax(tf.concat([pos_sim, neg_sim], axis=-1) / temperature)
    label = tf.concat([pos_label, neg_label], axis=-1)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit))
    return loss


class BaseTask(BaseAlgorithm):

    def dnn(self, inp, dnn_units, name="DNN", reuse=False):
        output = inp
        for i, dnn_unit in enumerate(dnn_units):
            output = tf.layers.dense(inp,
                                     dnn_unit,
                                     getattr(tf.nn, self.act),
                                     name="{}_{}".format(name, i),
                                     reuse=reuse)
            output = tf.layers.dropout(output, training=self.is_training, rate=self.in_drop_rate)
        return output

    def init(self, context):
        self.context = context
        self.logger = self.context.get_logger()
        self.logger.info("graph learn server init...")
        self.config = self.context.get_config()
        self.job_conf = self.config.get_all_job_config()
        self.model_conf = self.job_conf["user_param"]
        self.g2_gl_client = G2GraphLearnClient(context, self.config, self.model_conf)
        if not self.g2_gl_client.init_graph_learn_server():
            self.logger.error("Init graph learn server failed")
            return False
        self.logger.info("Run graph learn init success")
        self.graph = self.g2_gl_client.get_graph()
        self.batch_size = self.model_conf["batch_size"]
        self.neg_num = self.model_conf["neg_num"]
        self.user_feature_desc = self.model_conf["user_feature_desc"]
        self.neighs_num = self.model_conf["neighs_num"]
        self.gcn_dims = self.model_conf["gcn_dims"]
        self.agg_type = self.model_conf["agg_type"]
        self.act = self.model_conf["act"]
        self.in_drop_rate = self.model_conf["in_drop_rate"]
        self.temperature = self.model_conf["temperature"]
        self.cluster_spec, self.role_name, self.task_index = self.get_cluster_desc(context)

    def get_cluster_desc(self, context):
        role_name = os.environ['role_name']
        grpcClusterSpec = json.loads(os.environ['cluster_spec'])
        cluster_spec = {"ps": grpcClusterSpec["ps"]}
        cluster_spec["worker"] = {}
        for k, v in grpcClusterSpec["worker"].iteritems():
            cluster_spec["worker"][int(k)] = v
        task_index = context.get_task_id()
        return cluster_spec, role_name, task_index

    def _encoders(self):
        self.in_drop = tf.placeholder(tf.float32, shape=None, name='dropout_ph')
        self.is_training = tf.placeholder(tf.bool, shape=None, name='is_training')
        ps = self.cluster_spec["ps"]
        if len(ps) == 0:
            ps_hosts = None
        else:
            ps_hosts = ",".join(ps)
        feature_encoder = WideNDeepEncoder(
            self.user_feature_desc,
            total_feature_num=len(self.user_feature_desc),
            output_dim=0,
            need_dense=False,
            ps_hosts=ps_hosts,
            multivalent_cols_num=len(self.user_feature_desc)
        )
        feature_encoders = []
        for i in range(len(self.neighs_num) + 1):
            feature_encoders.append(feature_encoder)
        conv_layers = []
        for i in range(len(self.gcn_dims) - 1):
            ipt_dim = self.gcn_dims[i]
            opt_dim = self.gcn_dims[i + 1]
            conv_layers.append(
                gl.layers.GraphSageConv(i,
                                        ipt_dim,
                                        opt_dim,
                                        self.agg_type,
                                        act=getattr(tf.nn, self.act),
                                        name='conv')
            )
        encoder = gl.encoders.EgoGraphEncoder(feature_encoders,
                                              conv_layers,
                                              self.neighs_num,
                                              dropout=self.in_drop)
        return {
            gl.EgoKeys.POS_SRC: encoder,
            gl.EgoKeys.POS_DST: encoder,
            gl.EgoKeys.NEG_DST: encoder
        }

    def _feature_specs(self):
        feat_spec = gl.FeatureSpec(0,
                                   0,
                                   multivalent_attrs_num=len(self.user_feature_desc))
        hops_spec = []
        for i in range(len(self.neighs_num)):
            hops_spec.append(gl.HopSpec(feat_spec))
        ego_spec = gl.EgoSpec(feat_spec, hops_spec=hops_spec)
        return {
            gl.EgoKeys.POS_SRC: ego_spec,
            gl.EgoKeys.POS_DST: ego_spec,
            gl.EgoKeys.NEG_DST: ego_spec,
        }

    def gen_features(self, batch_data):
        return None, None, None

    def _receptive_fn(self, edge_type, prefix, v, strategy):
        alias_list = ["{}_v{}".format(prefix, i + 1) for i in range(len(self.neighs_num))]
        for nbr_count, alias in zip(self.neighs_num, alias_list):
            v = v.outV(edge_type).sample(1).by(strategy). \
                inV(edge_type).sample(nbr_count).by(strategy).alias(alias)
        return alias_list

    def _sample(self,
                edge_type,
                anchor_key=gl.EgoKeys.POS_SRC,
                pos_key=gl.EgoKeys.POS_DST,
                neg_key=gl.EgoKeys.NEG_DST,
                prefix="train"):
        source = self.graph.E(edge_type).shuffle(traverse=True).batch(self.batch_size).alias("{}_seed".format(prefix))
        # 获取连接的account_v
        # user->account
        user_vertex_name, account_vertex_name, neg_user_vertex_name, pos_user_vertex_name = \
            "{}_user_vertex".format(prefix), "{}_account_vertex".format(prefix), \
            "{}_neg_user_vertex".format(prefix), "{}_pos_user_vertex".format(prefix)
        user_vertex = source.outV().alias(user_vertex_name)
        account_vertex = source.inV().alias(account_vertex_name)
        # 通过account采集正样本
        neg_user_vertex = account_vertex.inNeg(edge_type).sample(self.neg_num) \
            .by("random").alias(neg_user_vertex_name)
        pos_user_vertex = account_vertex.inV(edge_type).sample(1).by("edge_weight").alias(pos_user_vertex_name)
        anchor_recept_names = self._receptive_fn(edge_type, user_vertex_name, user_vertex, "edge_weight")
        pos_recept_names = self._receptive_fn(edge_type, pos_user_vertex_name, pos_user_vertex, "edge_weight")
        neg_recept_names = self._receptive_fn(edge_type, neg_user_vertex_name, neg_user_vertex, "edge_weight")
        return source, {
            anchor_key: gl.Ego(user_vertex_name, anchor_recept_names),
            pos_key: gl.Ego(pos_user_vertex_name, pos_recept_names),
            neg_key: gl.Ego(neg_user_vertex_name, neg_recept_names)
        }

    def build_graph(self, context, features, feature_columns, labels):
        self.feature_specs = self._feature_specs()
        self.encoders = self._encoders()
        gl.set_eager_mode(False)
        query, egos = self._sample("interaction")
        dnn_units = self.model_conf["dnn_units"]
        ego_flow = gl.GSLEgoFlow(query, egos, self.feature_specs)
        self.iterator = ego_flow.iterator
        anchor_emb = self.encoders[gl.EgoKeys.POS_SRC].encode(ego_flow.get_tensor(gl.EgoKeys.POS_SRC))
        anchor_emb = self.dnn(anchor_emb, dnn_units, reuse=False)
        pos_dst_emb = self.encoders[gl.EgoKeys.POS_DST].encode(ego_flow.get_tensor(gl.EgoKeys.POS_DST))
        pos_dst_emb = self.dnn(pos_dst_emb, dnn_units, reuse=True)
        neg_dst_emb = self.encoders[gl.EgoKeys.NEG_DST].encode(ego_flow.get_tensor(gl.EgoKeys.NEG_DST))
        neg_dst_emb = self.dnn(neg_dst_emb, dnn_units, reuse=True)
        self.loss = unsupervised_softmax_cross_entropy_loss(anchor_emb,
                                                            pos_dst_emb,
                                                            neg_dst_emb,
                                                            temperature=self.temperature)
        self.global_step = tf.train.get_or_create_global_step()
        self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=5e-4)
        self.train_op = self._optimizer.minimize(self.loss, global_step=self.global_step)

    def feed_training_args(self):
        return {
            self.in_drop: self.in_drop_rate,
            self.is_training: True
        }

    def feed_evaluation_args(self):
        return {
            self.in_drop: 0.0,
            self.is_training: False
        }

    def run_train(self, context, mon_sess, task_id, thread_id):
        local_step = 0
        epoch = 0
        total_epoch = 3
        t = time.time()
        last_global_step = 0
        feed_train_args = self.feed_training_args()
        while not mon_sess.should_stop():
            mon_sess.run(self.iterator.initializer)
            try:
                _, global_step, loss = mon_sess.run([
                    self.train_op,
                    self.global_step,
                    self.loss
                ], feed_dict=feed_train_args)
                local_step += 1
                if local_step % 10 == 0:
                    print "GlobalStep {}, Epoch {}, Iter {}, GlobalStep/sec {:.2f} Loss {:.5f}".format(
                        global_step, epoch, local_step, (global_step - last_global_step) * 1.0 / (time.time() - t), loss
                    )
                    t = time.time()
                    last_global_step = global_step
            except tf.errors.OutOfRangeError:
                print "Out Of RangeError"
                epoch += 1
                mon_sess.run(self.iterator.initializer)
                if epoch >= total_epoch:
                    break
                else:
                    continue
        print "done"
