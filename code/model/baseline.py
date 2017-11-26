from __future__ import division
from __future__ import absolute_import
import tensorflow as tf


class baseline(object):
    def get_baseline_value(self):
        pass
    def update(self, target):
        pass


class ReactiveBaseline(baseline):
    def __init__(self, l):
        self.l = l
        self.b = tf.Variable( 0.0, trainable=False)
    def get_baseline_value(self):
        return self.b
    def update(self, target):
        self.b = tf.add((1-self.l)*self.b, self.l*target)



# class MLPbaseline(baseline):
#     def __init__(self, H):
#
#         def baseline_network(self, target_rewards, num_units=100, mlp_net=False):
#
#             """
#             network to predict the baseline reward. It takes in the feature representation
#             of the current state and regresses over the reward (target).
#             :param target_rewards: [B, T]
#             :param mlp_net: Predict using a single layer MLP if true else a linear regressor
#             :return:
#             """
#             with tf.variable_scope("baseline_net"):
#                 if mlp_net:
#
#                     layer_inp = self.baseline_inputs_tensor
#                     layer_inp = dense('h1', layer_inp.get_shape()[-1], num_units, layer_inp, tf.nn.relu)
#                     # layer_inp = dense('h2', layer_inp.get_shape()[-1], num_units, layer_inp, tf.nn.relu)
#                     # layer_inp = dense('h3', layer_inp.get_shape()[-1], num_units, layer_inp, tf.nn.relu)
#                     baseline_predictions = dense('val', layer_inp.get_shape()[-1], 1, layer_inp, None)
#                 else:
#
#                     # self.base_input = tf.placeholder(tf.float32, shape=[None, 4 * self.embedding_size], name='inputs')
#                     base_w = tf.get_variable('baseline_w', shape=[4 * self.embedding_size, 1],
#                                              dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(),
#                                              regularizer=tf.contrib.layers.l2_regularizer(0.01))
#                     base_b = tf.get_variable('baseline_b', shape=[1],
#                                              dtype=tf.float32, initializer=tf.zeros_initializer(),
#                                              regularizer=tf.contrib.layers.l2_regularizer(0.01))
#                     baseline_predictions = tf.matmul(self.baseline_inputs_tensor,
#                                                      base_w) + base_b  # self.baseline_inputs: [B*T,4D], baseline_predictions: [B*T]
#                 # self.target = tf.placeholder(tf.float32, name='targets')
#                 # baseline_loss = 2*(tf.nn.l2_loss(target_rewards - baseline_predictions,
#                 #                                name='l2_loss'))  # scalar
#                 baseline_loss = tf.reduce_mean((tf.reshape(target_rewards, [-1]) - baseline_predictions) ** 2)  # scalar
#
#             return baseline_predictions, baseline_loss
#
#     def dense(name, input_dim, output_dim, input, activation, std=1e-2):
#         with tf.variable_scope(name):
#             w = tf.get_variable('W', shape=[input_dim, output_dim],
#                                 initializer=tf.truncated_normal_initializer(stddev=std))
#             b = tf.get_variable('b', shape=[output_dim], initializer=tf.zeros_initializer())
#             out = tf.matmul(input, w) + b
#             if activation is not None:
#                 out = activation(out)
#         return out