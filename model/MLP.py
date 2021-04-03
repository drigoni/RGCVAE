import numpy as np
import tensorflow as tf


class MLP(object):
    def __init__(self, in_size, out_size, hid_sizes, dropout_keep_prob, activation_function=tf.nn.relu, name=None,
                 init_function=None, bias=True):
        self.in_size = in_size
        self.out_size = out_size
        self.hid_sizes = hid_sizes
        self.dropout_keep_prob = dropout_keep_prob
        self.activation_function = activation_function
        self.name = name
        self.bias = bias
        if init_function is not None:
            self.init_function = init_function
        else:
            self.init_function = self.init_weights
        if self.name is not None:
            with tf.name_scope(self.name):
                self.params = self.make_network_params()
        else:
            self.params = self.make_network_params()

    def make_network_params(self):
        dims = [self.in_size] + self.hid_sizes + [self.out_size]
        weight_sizes = list(zip(dims[:-1], dims[1:]))
        weights = [tf.Variable(self.init_function(s), name='MLP_W_layer%i' % i)
                   for (i, s) in enumerate(weight_sizes)]
        biases = [tf.Variable(np.zeros(s[-1]).astype(np.float32), name='MLP_b_layer%i' % i)
                  for (i, s) in enumerate(weight_sizes)]

        network_params = {
            "weights": weights,
            "biases": biases,
        }

        return network_params

    def init_weights(self, shape):
        return np.sqrt(6.0 / (shape[-2] + shape[-1])) * (2 * np.random.rand(*shape).astype(np.float32) - 1)

    def cal_l2_loss(self):
        current_sum = 0
        for W, b in zip(self.params["weights"], self.params["biases"]):
            current_sum += tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
        return current_sum

    def print(self, operation):
        for W, b in zip(self.params["weights"], self.params["biases"]):
            operation = tf.Print(operation, [tf.shape(W),
                                             tf.reduce_any(tf.is_nan(W)),
                                             tf.reduce_any(tf.is_inf(W)),
                                             tf.reduce_mean(W),
                                             tf.reduce_max(W),
                                             tf.reduce_min(W)], message=W.name,
                                 summarize=100000)
            operation = tf.Print(operation, [tf.shape(b),
                                             tf.reduce_any(tf.is_nan(b)),
                                             tf.reduce_any(tf.is_inf(b)),
                                             tf.reduce_mean(b),
                                             tf.reduce_max(b),
                                             tf.reduce_min(b)], message=b.name,
                                 summarize=100000)
        return operation

    def __call__(self, inputs, is_training=False):
        acts = inputs
        for W, b in zip(self.params["weights"], self.params["biases"]):
            hid = tf.matmul(acts, tf.nn.dropout(W, self.dropout_keep_prob))
            if self.bias:
                hid = hid + b
            acts = self.activation_function(hid)
        last_hidden = hid
        return last_hidden


class MLP_norm(object):
    def __init__(self, in_size, out_size, hid_sizes, dropout_keep_prob, activation_function=tf.nn.relu, name=None,
                 init_function=None, bias=True):
        self.in_size = in_size
        self.out_size = out_size
        self.hid_sizes = hid_sizes
        self.dropout_keep_prob = dropout_keep_prob
        self.activation_function = activation_function
        self.name = name
        self.bias = bias
        if init_function is not None:
            self.init_function = init_function
        else:
            self.init_function = self.init_weights
        if self.name is not None:
            with tf.name_scope(self.name):
                self.params = self.make_network_params()
        else:
            self.params = self.make_network_params()

    def make_network_params(self):
        dims = [self.in_size] + self.hid_sizes + [self.out_size]
        weight_sizes = list(zip(dims[:-1], dims[1:]))
        weights = [tf.Variable(self.init_function(s), name='MLP_W_layer%i' % i)
                   for (i, s) in enumerate(weight_sizes)]
        biases = [tf.Variable(np.zeros(s[-1]).astype(np.float32), name='MLP_b_layer%i' % i)
                  for (i, s) in enumerate(weight_sizes)]

        network_params = {
            "weights": weights,
            "biases": biases,
        }

        return network_params

    def init_weights(self, shape):
        return np.sqrt(6.0 / (shape[-2] + shape[-1])) * (2 * np.random.rand(*shape).astype(np.float32) - 1)

    def print(self, operation):
        for W, b in zip(self.params["weights"], self.params["biases"]):
            operation = tf.Print(operation, [tf.shape(W),
                                             tf.reduce_any(tf.is_nan(W)),
                                             tf.reduce_any(tf.is_inf(W)),
                                             tf.reduce_mean(W),
                                             tf.reduce_max(W),
                                             tf.reduce_min(W)], message=W.name,
                                 summarize=100000)
            operation = tf.Print(operation, [tf.shape(b),
                                             tf.reduce_any(tf.is_nan(b)),
                                             tf.reduce_any(tf.is_inf(b)),
                                             tf.reduce_mean(b),
                                             tf.reduce_max(b),
                                             tf.reduce_min(b)], message=b.name,
                                 summarize=100000)
        return operation

    def __call__(self, inputs, is_training=False):
        acts = inputs
        for W, b in zip(self.params["weights"], self.params["biases"]):
            hid = tf.matmul(acts, tf.nn.dropout(W, self.dropout_keep_prob))
            if self.bias:
                hid = hid + b
            hid = tf.layers.batch_normalization(hid, training=is_training)
            acts = self.activation_function(hid)
        last_hidden = hid
        return last_hidden
