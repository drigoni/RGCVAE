#!/usr/bin/env/python
"""
Usage:
    RGCVAE.py [options]

Options:
    -h --help                   Show this screen
    --dataset NAME              Dataset name: ZINC or QM9
    --config-file FILE          Hyperparameter configuration file path (in JSON format)
    --config CONFIG             Hyperparameter configuration dictionary (in JSON format)
    --data_dir NAME             Data dir name
    --restore FILE              File to restore weights from.
    --restore_n NAME            Epoch to restore
    --freeze-graph-model        Freeze weights of graph model components
    --restrict_data NAME        [0,1] Load only a subset of the entire dataset
"""

import copy
import sys
import traceback

import pandas as pd
from docopt import docopt
from rdkit.Chem import QED

from model.GGNN_core import *
from model.MLP import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 0, 1, 2, 3

"""
Comments provide the expected tensor shapes where helpful.

Key to symbols in comments:
---------------------------
[...]:  a tensor
; ; :   a list
b:      batch size
e:      number of edege types (3)
es:     maximum number of BFS transitions in this batch
v:      number of vertices per graph in this batch
h:      GNN hidden size
"""


class MolGVAE(ChemModel):
    def __init__(self, args):
        super().__init__(args)

    @classmethod
    def default_params(cls):
        params = dict(super().default_params())
        params.update({
            # general configuration
            'random_seed': 0,  # fixed for reproducibility
            'suffix': None,
            'log_dir': './results',
            'train_file': 'data/molecules_train_%s.json' % dataset,
            'valid_file': 'data/molecules_valid_%s.json' % dataset,
            # 'train_file': 'data/molecules_small_dataset.json',
            # 'valid_file': 'data/molecules_test_%s.json' % dataset,
            'test_file': 'data/molecules_test_%s.json' % dataset,
            "use_gpu": True,
            "tensorboard": 3,  # frequency if we use tensorboard else None
            "use_rec_multi_threads": True,

            # general procedure configuration
            'task_ids': [0],  # id of property prediction
            "fix_molecule_validation": True,
            'generation': 0,  # 0 = do training, 1 = do only gen, 2 = do only rec
            'number_of_generation': 20000,
            'reconstruction_en': 20,  # number of encoding in reconstruction
            'reconstruction_dn': 1,  # number of decoding in reconstruction
            'batch_size': utils.dataset_info(dataset)['batch_size'],
            'num_epochs': utils.dataset_info(dataset)['n_epochs'],
            'hidden_size_encoder': 70,  # encoder hidden size dimension
            'latent_space_size': 70,  # latent space size
            'prior_learning_rate': 0.05,  # gradient ascent optimization
            'hist_local_search': False,
            'use_edge_bias': True,  # whether use edge bias in gnn
            'optimization_step': 0,
            "use_argmax_nodes": False,  # use random sampling or argmax during node sampling
            "use_argmax_bonds": False,  # use random sampling or argmax during bonds generations
            'use_mask': True,  # true to use node mask
            "path_random_order": False,  # False: canonical order, True: random order
            'tie_fwd_bkwd': True,
            "compensate_num": 1,  # how many atoms to be added during generation
            "gen_hist_sampling": False,  # sampling new hist during generation task.

            # loss params
            "kl_trade_off_lambda": 0.05 if "qm9" in dataset else 0.01,
            "kl_trade_off_lambda_factor": 0,
            "qed_trade_off_lambda": 10,
            'learning_rate': 0.001,
            'task_sample_ratios': {},  # weights for properties
            'clamp_gradient_norm': 1.0,
            # dropout
            'graph_state_dropout_keep_prob': 1,
            'out_layer_dropout_keep_prob': 1.0,
            'edge_weight_dropout_keep_prob': 1,

            # GCNN
            'num_timesteps': 5,  # gnn propagation step
            'use_graph': False,  # use gnn
            'use_gin': True,  # use gin as gnn
            'gin_epsilon': 0,  # gin epsilon
            'residual_connection_on': True,  # whether residual connection is on
            'residual_connections': {
                2: [0],
                4: [0, 2],
                6: [0, 2, 4],
                8: [0, 2, 4, 6],
                10: [0, 2, 4, 6, 8],
                12: [0, 2, 4, 6, 8, 10],
                14: [0, 2, 4, 6, 8, 10, 12],
                16: [0, 2, 4, 6, 8, 10, 12, 14],
                18: [0, 2, 4, 6, 8, 10, 12, 14, 16],
                20: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
            },
        })

        return params

    def prepare_specific_graph_model(self) -> None:
        # params
        h_dim_en = self.params['hidden_size_encoder']
        ls_dim = self.params['latent_space_size']
        h_dim_de = ls_dim + 50
        expanded_h_dim = h_dim_de + h_dim_en + 1  # 1 for focus bit
        hist_dim = self.histograms['hist_dim']

        self.placeholders['smiles'] = tf.placeholder_with_default("No smiles as default", [], name='is_training')
        self.placeholders['n_epoch'] = tf.placeholder_with_default(0, [], name='n_epoch')
        self.placeholders['is_training'] = tf.placeholder_with_default(False, [], name='is_training')
        self.placeholders['target_values'] = tf.placeholder(tf.float32, [len(self.params['task_ids']), None],
                                                            name='target_values')
        self.placeholders['target_mask'] = tf.placeholder(tf.float32, [len(self.params['task_ids']), None],
                                                          name='target_mask')
        self.placeholders['num_graphs'] = tf.placeholder(tf.int32, [], name='num_graphs')
        self.placeholders['out_layer_dropout_keep_prob'] = tf.placeholder(tf.float32, [],
                                                                          name='out_layer_dropout_keep_prob')
        self.placeholders['graph_state_keep_prob'] = tf.placeholder(tf.float32, None, name='graph_state_keep_prob')
        self.placeholders['edge_weight_dropout_keep_prob'] = tf.placeholder(tf.float32, None,
                                                                            name='edge_weight_dropout_keep_prob')
        # mask out invalid node
        self.placeholders['node_mask'] = tf.placeholder(tf.float32, [None, None], name='node_mask')  # [b v]
        self.placeholders['num_vertices'] = tf.placeholder(tf.int32, (), name="num_vertices")
        # adj for encoder
        self.placeholders['adjacency_matrix'] = tf.placeholder(tf.float32, [None, self.num_edge_types, None, None],
                                                               name="adjacency_matrix")  # [b, e, v, v]
        # labels for node symbol prediction
        self.placeholders['node_symbols'] = tf.placeholder(tf.float32, [None, None, self.params[
            'num_symbols']])  # [b, v, edge_type]
        # z_prior sampled from standard normal distribution
        self.placeholders['z_prior'] = tf.placeholder(tf.float32, [None, None, ls_dim],
                                                      name='z_prior')  # the prior of z sampled from normal distribution

        # histograms for the first part of the decoder
        self.placeholders['histograms'] = tf.placeholder(tf.int32, (None, hist_dim), name="histograms")
        self.placeholders['n_histograms'] = tf.placeholder(tf.int32, (None), name="n_histograms")
        self.placeholders['hist'] = tf.placeholder(tf.int32, (None, hist_dim), name="hist")
        self.placeholders['incr_hist'] = tf.placeholder(tf.float32, (None, None, hist_dim), name="incr_hist")
        self.placeholders['incr_diff_hist'] = tf.placeholder(tf.float32, (None, None, hist_dim), name="incr_diff_hist")
        self.placeholders['incr_node_mask'] = tf.placeholder(tf.float32, (None, None, self.params['num_symbols']),
                                                             name="incr_node_mask")

        # weights for encoder and decoder GNN.
        with tf.name_scope('graph_convolution_vars'):
            if self.params['use_graph']:
                if self.params["residual_connection_on"]:
                    # weights for encoder and decoder GNN. Different weights for each iteration
                    for scope in ['_encoder', '_decoder']:
                        if scope == '_encoder':
                            new_h_dim = h_dim_en
                        else:
                            new_h_dim = expanded_h_dim
                            # For each GNN iteration
                        for iter_idx in range(self.params['num_timesteps']):
                            with tf.variable_scope("gru_scope" + scope + str(iter_idx), reuse=False):
                                self.weights['edge_weights' + scope + str(iter_idx)] = tf.Variable(
                                    utils.glorot_init([self.num_edge_types, new_h_dim, new_h_dim]))
                                if self.params['use_edge_bias']:
                                    self.weights['edge_biases' + scope + str(iter_idx)] = tf.Variable(
                                        np.zeros([self.num_edge_types, 1, new_h_dim]).astype(np.float32))

                                cell = tf.contrib.rnn.GRUCell(new_h_dim)
                                cell = tf.nn.rnn_cell.DropoutWrapper(cell, state_keep_prob=self.placeholders[
                                    'graph_state_keep_prob'])
                                self.weights['node_gru' + scope + str(iter_idx)] = cell
                else:
                    for scope in ['_encoder', '_decoder']:
                        if scope == '_encoder':
                            new_h_dim = h_dim_en
                        else:
                            new_h_dim = expanded_h_dim
                        self.weights['edge_weights' + scope] = tf.Variable(
                            utils.glorot_init([self.num_edge_types, new_h_dim, new_h_dim]))
                        if self.params['use_edge_bias']:
                            self.weights['edge_biases' + scope] = tf.Variable(
                                np.zeros([self.num_edge_types, 1, new_h_dim]).astype(np.float32))
                        with tf.variable_scope("gru_scope" + scope):
                            cell = tf.contrib.rnn.GRUCell(new_h_dim)
                            cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                                 state_keep_prob=self.placeholders[
                                                                     'graph_state_keep_prob'])
                            self.weights['node_gru' + scope] = cell
            # GIN VERSION
            elif self.params['use_gin']:
                self.weights['gin_epsilon'] = tf.constant(self.params['gin_epsilon'], tf.float32)
                for scope in ['_encoder']:
                    if scope == '_encoder':
                        new_h_dim = h_dim_en
                    else:
                        new_h_dim = expanded_h_dim
                        # For each GNN iteration
                    for iter_idx in range(self.params['num_timesteps']):
                        with tf.variable_scope("gin_scope" + scope + str(iter_idx), reuse=False):
                            for edge_type in range(self.num_edge_types):
                                self.weights['MLP_edge' + str(edge_type) + scope + str(iter_idx)] = MLP(new_h_dim,
                                                                                                        new_h_dim,
                                                                                                        [],
                                                                                                        self.placeholders[
                                                                                                            'out_layer_dropout_keep_prob'],
                                                                                                        name='MLP_edge',
                                                                                                        activation_function=tf.nn.leaky_relu,
                                                                                                        bias=True)
                            self.weights['MLP' + scope + str(iter_idx)] = MLP_norm(new_h_dim,
                                                                                   new_h_dim,
                                                                                   [new_h_dim],
                                                                                   self.placeholders[
                                                                                       'out_layer_dropout_keep_prob'],
                                                                                   activation_function=tf.nn.leaky_relu)
            # ESP1 - base graph convolution layer
            # elif self.params['use_gin']:
            #    self.weights['gin_epsilon'] = tf.constant(self.params['gin_epsilon'], tf.float32)
            #    for scope in ['_encoder']:
            #        if scope == '_encoder':
            #            new_h_dim = h_dim_en
            #        else:
            #            new_h_dim = expanded_h_dim
            #            # For each GNN iteration
            #        for iter_idx in range(self.params['num_timesteps']):
            #            with tf.variable_scope("gin_scope" + scope + str(iter_idx), reuse=False):
            #                self.weights['MLP' + scope + str(iter_idx)] = MLP(new_h_dim,
            #                                                                  new_h_dim,
            #                                                                  [],
            #                                                                  self.placeholders[
            #                            python -u RGCVAE.py --dataset moses --config '{"generation":0, "log_dir":"./results_esp3", "use_mask":false, "num_timesteps":10 "kl_trade_off_lambda":0.05, "hidden_size_encoder":150, "latent_space_size":150, "batch_size":500, "suffix":"kl0.05d150i12"}' | tee output_esp3.txt                                          'out_layer_dropout_keep_prob'],
            #                                                                  activation_function=tf.nn.leaky_relu)
            # ESP2 - link nicolo'
            # elif self.params['use_gin']:
            #     self.weights['gin_epsilon'] = tf.constant(self.params['gin_epsilon'], tf.float32)
            #     for scope in ['_encoder']:
            #         if scope == '_encoder':
            #             new_h_dim = h_dim_en
            #         else:
            #             new_h_dim = expanded_h_dim
            #             # For each GNN iteration
            #         for iter_idx in range(self.params['num_timesteps']):
            #             with tf.variable_scope("gin_scope" + scope + str(iter_idx), reuse=False):
            #                 for edge_type in range(self.num_edge_types):
            #                     self.weights['MLP_edge' + str(edge_type) + scope + str(iter_idx)] = MLP(new_h_dim,
            #                                                                                             new_h_dim,
            #                                                                                             [],
            #                                                                                             self.placeholders[
            #                                                                                                 'out_layer_dropout_keep_prob'],
            #                                                                                             name='MLP_edge',
            #                                                                                             activation_function=tf.nn.leaky_relu)
            #                 self.weights['MLP' + scope + str(iter_idx)] = MLP(new_h_dim,
            #                                                                   new_h_dim,
            #                                                                   [],
            #                                                                   self.placeholders[
            #                                                                       'out_layer_dropout_keep_prob'],
            #                                                                  activation_function=tf.nn.leaky_relu)

        # GIN VERSION
        with tf.name_scope('distribution_vars'):
            # Weights final part encoder. They map all nodes in one point in the latent space
            input_size_distribution = h_dim_en * (self.params['num_timesteps'] + 1)
            # self.weights['mean_MLP'] = MLP(input_size_distribution, ls_dim,
            self.weights['mean_MLP'] = MLP(h_dim_en, ls_dim,
                                           [],
                                           self.placeholders['out_layer_dropout_keep_prob'],
                                           activation_function=tf.nn.leaky_relu,
                                           name='mean_MLP')
            # self.weights['logvariance_MLP'] = MLP(input_size_distribution, ls_dim,
            self.weights['logvariance_MLP'] = MLP(h_dim_en, ls_dim,
                                                  [],
                                                  self.placeholders['out_layer_dropout_keep_prob'],
                                                  activation_function=tf.nn.leaky_relu,
                                                  name='logvariance_MLP')
        # ESP1 - base graph convolution layer
        # with tf.name_scope('distribution_vars'):
        #     # Weights final part encoder. They map all nodes in one point in the latent space
        #     input_size_distribution = h_dim_en * (self.params['num_timesteps'] + 1)
        #     self.weights['mean_MLP'] = MLP(h_dim_en, ls_dim,
        #                                    # [input_size_distribution, input_size_distribution, ls_dim],
        #                                    [],
        #                                    self.placeholders['out_layer_dropout_keep_prob'],
        #                                    activation_function=tf.nn.leaky_relu,
        #                                    name='mean_MLP')
        #     self.weights['logvariance_MLP'] = MLP(h_dim_en, ls_dim,
        #                                           [],
        #                                           self.placeholders['out_layer_dropout_keep_prob'],
        #                                           activation_function=tf.nn.leaky_relu,
        #                                           name='logvariance_MLP')

        with tf.name_scope('gen_nodes_vars'):
            self.weights['histogram_MLP'] = MLP(ls_dim + 2 * hist_dim, 50,
                                                [],
                                                self.placeholders['out_layer_dropout_keep_prob'],
                                                activation_function=tf.nn.leaky_relu,
                                                name='histogram_MLP')
            self.weights['node_symbol_MLP'] = MLP(h_dim_de, self.params['num_symbols'],
                                                  [h_dim_de],
                                                  self.placeholders['out_layer_dropout_keep_prob'],
                                                  activation_function=tf.nn.leaky_relu,
                                                  name='node_symbol_MLP')

        with tf.name_scope('gen_edges_vars'):
            # gen edges
            # input_size_edge = 4 * (h_dim_en + h_dim_de)  # with sum and product as context
            input_size_edge = 3 * (h_dim_en + h_dim_de)  # remove product as context
            self.weights['edge_gen_MLP'] = MLP(input_size_edge, 1,
                                               [input_size_edge, h_dim_de],
                                               self.placeholders['out_layer_dropout_keep_prob'],
                                               activation_function=tf.nn.relu,
                                               name='edge_gen_MLP')
            self.weights['edge_type_gen_MLP'] = MLP(input_size_edge, self.num_edge_types,
                                                    [input_size_edge, h_dim_de],
                                                    self.placeholders['out_layer_dropout_keep_prob'],
                                                    activation_function=tf.nn.relu,
                                                    name='edge_type_gen_MLP')

        with tf.name_scope('qed_vars'):
            # weights for linear projection on qed prediction input
            input_size_qed = h_dim_en + h_dim_de
            self.weights['qed_weights'] = tf.Variable(utils.glorot_init([input_size_qed, input_size_qed]),
                                                      name='qed_weights')
            self.weights['qed_biases'] = tf.Variable(np.zeros([1, input_size_qed]).astype(np.float32),
                                                     name='qed_biases')
            self.weights['plogP_weights'] = tf.Variable(utils.glorot_init([input_size_qed, input_size_qed]),
                                                        name='plogP_weights')
            self.weights['plogP_biases'] = tf.Variable(np.zeros([1, input_size_qed]).astype(np.float32),
                                                       name='plogP_biases')

        # use node embeddings
        self.weights["node_embedding"] = tf.Variable(utils.glorot_init([self.params["num_symbols"], h_dim_en]),
                                                     name='node_embedding')

        # graph state mask
        self.ops['graph_state_mask'] = tf.expand_dims(self.placeholders['node_mask'], 2)

    # transform one hot vector to dense embedding vectors
    def get_node_embedding_state(self, one_hot_state):
        node_nums = tf.argmax(one_hot_state, axis=2)
        return tf.nn.embedding_lookup(self.weights["node_embedding"], node_nums) * self.ops['graph_state_mask']

    def compute_final_node_representations_with_residual(self, h, adj, scope_name):  # scope_name: _encoder or _decoder
        # h: initial representation, adj: adjacency matrix, different GNN parameters for encoder and decoder
        v = self.placeholders['num_vertices']
        # _decoder uses a larger latent space because concat of symbol and latent representation
        if scope_name == "_decoder":
            h_dim = self.params['hidden_size_encoder'] + self.params['hidden_size_decoder'] + 1
        else:
            h_dim = self.params['hidden_size_encoder']
        h = tf.reshape(h, [-1, h_dim])  # [b*v, h]
        # record all hidden states at each iteration
        all_hidden_states = [h]
        for iter_idx in range(self.params['num_timesteps']):
            with tf.variable_scope("gru_scope" + scope_name + str(iter_idx), reuse=None) as g_scope:
                for edge_type in range(self.num_edge_types):
                    # the message passed from this vertice to other vertices
                    m = tf.matmul(h, self.weights['edge_weights' + scope_name + str(iter_idx)][edge_type])  # [b*v, h]
                    if self.params['use_edge_bias']:
                        m += self.weights['edge_biases' + scope_name + str(iter_idx)][edge_type]  # [b, v, h]
                    m = tf.reshape(m, [-1, v, h_dim])  # [b, v, h]
                    # collect the messages from other vertices to each vertice
                    if edge_type == 0:
                        acts = tf.matmul(adj[edge_type], m)
                    else:
                        acts += tf.matmul(adj[edge_type], m)
                # all messages collected for each node
                acts = tf.reshape(acts, [-1, h_dim])  # [b*v, h]
                # add residual connection here
                layer_residual_connections = self.params['residual_connections'].get(iter_idx)
                if layer_residual_connections is None:
                    layer_residual_states = []
                else:
                    layer_residual_states = [all_hidden_states[residual_layer_idx]
                                             for residual_layer_idx in layer_residual_connections]
                # concat current hidden states with residual states
                acts = tf.concat([acts] + layer_residual_states, axis=1)  # [b, (1+num residual connection)* h]

                # feed msg inputs and hidden states to GRU
                h = self.weights['node_gru' + scope_name + str(iter_idx)](acts, h)[1]  # [b*v, h]
                # record the new hidden states
                all_hidden_states.append(h)
        last_h = tf.reshape(all_hidden_states[-1], [-1, v, h_dim])
        return last_h

    def compute_final_node_representations_without_residual(self, h, adj, edge_weights, edge_biases, node_gru,
                                                            gru_scope_name):
        # h: initial representation, adj: adjacency matrix, different GNN parameters for encoder and decoder
        v = self.placeholders['num_vertices']
        if gru_scope_name == "gru_scope_decoder":
            h_dim = self.params['hidden_size_encoder'] + self.params['hidden_size_decoder']
        else:
            h_dim = self.params['hidden_size_encoder']
        h = tf.reshape(h, [-1, h_dim])

        with tf.variable_scope(gru_scope_name) as scope:
            for i in range(self.params['num_timesteps']):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                for edge_type in range(self.num_edge_types):
                    m = tf.matmul(h, tf.nn.dropout(edge_weights[edge_type],
                                                   keep_prob=self.placeholders[
                                                       'edge_weight_dropout_keep_prob']))  # [b*v, h]
                    if self.params['use_edge_bias']:
                        m += edge_biases[edge_type]  # [b, v, h]
                    m = tf.reshape(m, [-1, v, h_dim])  # [b, v, h]
                    if edge_type == 0:
                        acts = tf.matmul(adj[edge_type], m)  # adj[edge_type]->[b,v,v]   m->[b,v,h]  res->b[v,h]
                    else:
                        acts += tf.matmul(adj[edge_type], m)
                acts = tf.reshape(acts, [-1, h_dim])  # [b*v, h]
                h = node_gru(acts, h)[1]  # [b*v, h]
            last_h = tf.reshape(h, [-1, v, h_dim])
        return last_h

    # ESP 1 - base graph convolution
    def compute_final_node_with_GIN(self, h, adj, scope_name):  # scope_name: _encoder or _decoder
        # h: initial representation, adj: adjacency matrix, different GNN parameters for encoder and decoder
        v = self.placeholders['num_vertices']
        # _decoder uses a larger latent space because concat of symbol and latent representation
        h_dim = self.params['hidden_size_encoder']
        weigths_concat = h
        for iter_idx in range(self.params['num_timesteps']):
            with tf.variable_scope("gin_scope" + scope_name + str(iter_idx), reuse=None) as g_scope:
                simple_adj = tf.reduce_sum(adj, axis=0)  # [b, v, v]
                acts = tf.matmul(simple_adj, h, name="gin_matmul")  # [b, v, h]
                neig_sum = tf.reduce_sum(simple_adj, axis=-1, keepdims=True)
                neig_check = tf.where(neig_sum > 0,
                                      neig_sum,
                                      tf.ones_like(neig_sum))
                acts = acts / neig_check  # broadcasting
                # all messages collected for each node
                input = h + acts
                input = tf.reshape(input, [-1, h_dim])  # [b*v, h]
                h = tf.nn.leaky_relu(
                    self.weights['MLP' + scope_name + str(iter_idx)](input, self.placeholders['is_training']))
                # tensorboard
                tf.summary.histogram("gin_scope" + scope_name + str(iter_idx) + "_node_state", h)
                h = tf.reshape(h, [-1, v, h_dim])
                weigths_concat = tf.concat([weigths_concat, h], axis=-1)
        last_h = h
        # tensorboard
        tf.summary.histogram("last_weigths_concat", weigths_concat)
        last_h = last_h * self.ops['graph_state_mask']
        average_pooling = tf.reduce_mean(last_h, axis=1, keepdims=False)
        return last_h, weigths_concat, average_pooling

    def compute_mean_and_logvariance(self):
        v = self.placeholders['num_vertices']
        h_dim = self.params['hidden_size_encoder']
        ls_dim = self.params['latent_space_size']
        # Concatenation
        # reshped_last_h = tf.reshape(self.ops['final_node_representations'][1], [-1, h_dim * (self.params['num_timesteps'] + 1)])
        # Just last rep.
        reshped_last_h = tf.reshape(self.ops['final_node_representations'][0], [-1, h_dim])
        mean = self.weights['mean_MLP'](reshped_last_h, self.placeholders['is_training'])
        logvariance = tf.minimum(self.weights['logvariance_MLP'](reshped_last_h, self.placeholders['is_training']), 2.5)

        # mask everything
        mean = tf.reshape(mean, [-1, v, ls_dim]) * self.ops['graph_state_mask']
        logvariance = tf.reshape(logvariance, [-1, v, ls_dim]) * self.ops['graph_state_mask']
        mean = tf.reshape(mean, [-1, ls_dim])
        logvariance = tf.reshape(logvariance, [-1, ls_dim])

        self.ops['mean'] = mean
        self.ops['logvariance'] = logvariance

        tf.summary.histogram("mean", self.ops['mean'])
        tf.summary.histogram("logvariance", self.ops['logvariance'])

    def sample_with_mean_and_logvariance(self):
        v = self.placeholders['num_vertices']
        ls_dim = self.params['latent_space_size']
        # Sample from normal distribution
        z_prior = tf.reshape(self.placeholders['z_prior'], [-1, ls_dim])
        # Train: sample from u, Sigma. Generation: sample from 0,1
        if self.params['generation'] in [0, 2, 4]:  # Training, reconstruction and validation
            z_sampled = tf.add(self.ops['mean'], tf.multiply(tf.sqrt(tf.exp(self.ops['logvariance'])), z_prior))
        else:
            z_sampled = z_prior
        # filter
        z_sampled = tf.reshape(z_sampled, [-1, v, ls_dim]) * self.ops['graph_state_mask']

        self.ops['z_sampled'] = z_sampled
        tf.summary.histogram("z_sampled", self.ops['z_sampled'])

    """
    Construct the nodes representations
    """

    def construct_nodes(self):
        h_dim_en = self.params['hidden_size_encoder']
        ls_dim = self.params['latent_space_size']
        h_dim_de = ls_dim + 50  # check tf.tensorboard value
        batch_size = tf.shape(self.ops['z_sampled'])[0]
        v = self.placeholders['num_vertices']

        if self.params['generation'] in [0, 4]:  # Training and test
            initial_nodes_decoder, node_symbol_prob, sampled_atoms = self.train_procedure()
        elif self.params['generation'] in [1, 2, 3]:  # Reconstruction and Generation
            initial_nodes_decoder, node_symbol_prob, sampled_atoms = self.gen_rec_procedure()
        # batch normalization
        initial_nodes_decoder_masked = initial_nodes_decoder * self.ops['graph_state_mask']
        initial_nodes_decoder_reshaped = tf.reshape(initial_nodes_decoder_masked, [-1, h_dim_de])
        initial_nodes_decoder_bn = tf.layers.batch_normalization(initial_nodes_decoder_reshaped,
                                                                 training=self.placeholders['is_training'])
        initial_nodes_decoder_final = tf.reshape(initial_nodes_decoder_bn, [batch_size, v, h_dim_de])
        self.ops['initial_nodes_decoder'] = initial_nodes_decoder_final * self.ops['graph_state_mask']

        # uncomment only if we want to remove the bn
        # self.ops['initial_nodes_decoder'] = initial_nodes_decoder * self.ops['graph_state_mask']
        self.ops['node_symbol_prob'] = node_symbol_prob * self.ops['graph_state_mask']
        self.ops['sampled_atoms'] = sampled_atoms * tf.cast(self.placeholders['node_mask'], tf.int32)

        # calc one hot repr. for type of atom
        self.ops['latent_node_symbols'] = tf.one_hot(self.ops['sampled_atoms'],
                                                     self.params['num_symbols'],
                                                     name='latent_node_symbols') * self.ops['graph_state_mask']

        tf.summary.histogram("hist_embedding", self.ops['initial_nodes_decoder'][:, :, -50:])
        tf.summary.histogram("initial_nodes_decoder", self.ops['initial_nodes_decoder'])
        tf.summary.histogram("sampled_atoms", self.ops['sampled_atoms'])

    def train_procedure(self):
        latent_space_dim = self.params['latent_space_size']
        h_dim_de = latent_space_dim + 50
        hist_dim = self.histograms['hist_dim']
        num_symbols = self.params['num_symbols']
        batch_size = tf.shape(self.ops['z_sampled'])[0]
        v = self.placeholders['num_vertices']  # bucket size dimension, not all time the real one.

        # calc emb hist
        input_z_hist = tf.concat(
            [self.ops['z_sampled'], self.placeholders['incr_hist'], self.placeholders['incr_diff_hist']], -1)
        z_input = tf.reshape(input_z_hist, [-1, latent_space_dim + 2 * hist_dim])
        hist_emb = tf.nn.tanh(self.weights['histogram_MLP'](z_input, self.placeholders['is_training']))

        float_z_sampled = tf.reshape(self.ops['z_sampled'], [-1, latent_space_dim])
        conc_z_hist = tf.concat([float_z_sampled, hist_emb], -1)
        initial_nodes_decoder = tf.reshape(conc_z_hist, [batch_size, v, h_dim_de])

        # calc prob with or without mask
        atom_logits = self.weights['node_symbol_MLP'](conc_z_hist, self.placeholders['is_training'])
        if self.params['use_mask']:
            flat_incr_node_mask = tf.reshape(self.placeholders['incr_node_mask'], [-1, num_symbols])
            atom_logits = tf.where(tf.reduce_sum(flat_incr_node_mask, 1) > 0,
                                   atom_logits + (flat_incr_node_mask * utils.LARGE_NUMBER - utils.LARGE_NUMBER),
                                   atom_logits)
        atom_prob = tf.nn.softmax(atom_logits)
        node_symbol_prob = tf.reshape(atom_prob, [batch_size, v, num_symbols])

        sampled_atoms = self.placeholders['node_symbols']
        sampled_atoms_casted = tf.argmax(sampled_atoms, axis=-1, output_type=tf.int32)
        return initial_nodes_decoder, node_symbol_prob, sampled_atoms_casted

    def gen_rec_procedure(self):
        latent_space_dim = self.params['latent_space_size']
        h_dim_de = latent_space_dim + 50
        num_symbols = self.params['num_symbols']
        batch_size = tf.shape(self.ops['z_sampled'])[0]

        # save the new atom [b, v, h]
        atoms = tf.TensorArray(dtype=tf.int32, size=batch_size, element_shape=[None, 1])
        init_atoms = tf.TensorArray(dtype=tf.float32, size=batch_size, element_shape=[None, h_dim_de])
        fx_prob = tf.TensorArray(dtype=tf.float32, size=batch_size, element_shape=[None, num_symbols])

        # iteration on all the molecules in the batch size
        idx, atoms, init_atoms, fx_prob = tf.while_loop(
            lambda idx, atoms, init_atoms, fx_prob: tf.less(idx, batch_size),
            # numbers of example sampled
            self.for_each_molecula,
            (tf.constant(0), atoms, init_atoms, fx_prob),
            parallel_iterations=self.params['batch_size']
        )

        init_h_states = init_atoms.stack()
        nodes_type_probs = fx_prob.stack()
        atoms = atoms.stack()

        return init_h_states, nodes_type_probs, tf.squeeze(atoms, -1)

    def for_each_molecula(self, idx_sample, atoms, init_vertices_all, fx_prob_all):
        latent_space_dim = self.params['latent_space_size']
        h_dim_de = latent_space_dim + 50
        num_symbols = self.params['num_symbols']
        v = self.placeholders['num_vertices']  # bucket size dimension, not all time the real one.
        current_hist = self.placeholders['hist'][idx_sample]
        hist_dim = self.histograms['hist_dim']
        zero_hist = tf.zeros([hist_dim], tf.int32)

        # select the rigth function according to reconstruction or generation
        if self.params['generation'] == 1:
            funct_to_call = self.generate_nodes
        else:
            funct_to_call = self.reconstruct_nodes

        sampled_atoms = tf.TensorArray(dtype=tf.int32, size=v, element_shape=[1])
        vertices = tf.TensorArray(dtype=tf.float32, size=v, element_shape=[h_dim_de])
        fx_prob = tf.TensorArray(dtype=tf.float32, size=v, element_shape=[num_symbols])

        # iteration on all the atoms in a molecule
        idx_atoms, a, v, fx, _, _, _ = tf.while_loop(
            lambda idx_atoms, s_atoms, vertices, fx_prob, s_idx, zero_hist, current_hist: tf.less(idx_atoms, v),
            funct_to_call,
            (tf.constant(0), sampled_atoms, vertices, fx_prob, idx_sample, zero_hist, current_hist)
        )

        atoms = atoms.write(idx_sample, a.stack())
        init_vertices_all = init_vertices_all.write(idx_sample, v.stack())
        fx_prob_all = fx_prob_all.write(idx_sample, fx.stack())

        return idx_sample + 1, atoms, init_vertices_all, fx_prob_all

    def reconstruct_nodes(self, idx_atom, atoms, init_vertices, fx_prob, idx_sample, updated_hist, sampled_hist):
        current_sample_z = self.ops['z_sampled'][idx_sample][idx_atom]
        current_mask_value = self.placeholders['node_mask'][idx_sample][idx_atom]
        current_sample_hist_casted = tf.cast(sampled_hist, dtype=tf.float32)

        # concatenation with the histogram embedding
        current_hist_casted = tf.cast(updated_hist, dtype=tf.float32)
        hist_diff = tf.subtract(current_sample_hist_casted, current_hist_casted)
        hist_diff_pos = tf.where(hist_diff > 0, hist_diff, tf.zeros_like(hist_diff))

        conc = tf.concat([current_sample_z, current_hist_casted, hist_diff_pos], axis=0)
        exp = tf.expand_dims(conc, 0)  # [1, z + Hdiff + Hcurrent]
        # build a node with NN (K)
        hist_emb = tf.nn.tanh(self.weights['histogram_MLP'](exp, self.placeholders['is_training']))
        new_z_concat = tf.concat([tf.expand_dims(current_sample_z, 0), hist_emb], -1)

        atom_logits = self.weights['node_symbol_MLP'](new_z_concat, self.placeholders['is_training'])
        atom_logits = tf.squeeze(atom_logits)
        if self.params['use_mask']:
            atom_logits, mask = self.mask_mols(atom_logits, hist_diff_pos)
        node_probs = tf.nn.softmax(atom_logits)

        probs_value = node_probs
        s_atom = self.sample_atom(probs_value, True)
        new_updated_hist = self.update_hist(updated_hist, s_atom)
        # if the node should be masked, we do not update the current_histogram. In this way, it is always the last one
        # which represent the molecule.
        # new_updated_hist = tf.cond(tf.equal(current_mask_value, 0),
        #                            lambda: updated_hist,
        #                            lambda: self.update_hist(updated_hist, s_atom))

        s_atom, init_v, fx = tf.expand_dims(s_atom, 0), tf.squeeze(new_z_concat), probs_value

        atoms = atoms.write(idx_atom, s_atom)
        init_vertices = init_vertices.write(idx_atom, init_v)
        fx_prob = fx_prob.write(idx_atom, fx)
        return idx_atom + 1, atoms, init_vertices, fx_prob, idx_sample, new_updated_hist, sampled_hist

    """
    For each node (atoms) calculates histograms and new hidden representations
    """

    def generate_nodes(self, idx_atom, atoms, init_vertices, fx_prob, idx_sample, updated_hist, sampled_hist):
        hist_dim = self.histograms['hist_dim']
        current_sample_z = self.ops['z_sampled'][idx_sample][idx_atom]
        current_mask_value = self.placeholders['node_mask'][idx_sample][idx_atom]

        current_sample_hist_casted = tf.cast(sampled_hist, dtype=tf.float32)
        current_hist_casted = tf.cast(updated_hist, dtype=tf.float32)
        hist_diff = tf.subtract(current_sample_hist_casted, current_hist_casted)
        hist_diff_pos = tf.where(hist_diff > 0, hist_diff, tf.zeros_like(hist_diff))

        conc = tf.concat([current_sample_z, current_hist_casted, hist_diff_pos], axis=0)
        exp = tf.expand_dims(conc, 0)
        # build a node with NN (K)
        hist_emb = tf.nn.tanh(self.weights['histogram_MLP'](exp, self.placeholders['is_training']))
        new_z_concat = tf.concat([tf.expand_dims(current_sample_z, 0), hist_emb], -1)

        atom_logits = self.weights['node_symbol_MLP'](new_z_concat, self.placeholders['is_training'])
        atom_logits = tf.squeeze(atom_logits)
        if self.params['use_mask']:
            atom_logits, mask = self.mask_mols(atom_logits, hist_diff_pos)
        node_probs = tf.nn.softmax(atom_logits)
        s_atom = self.sample_atom(node_probs, False)
        new_updated_hist = self.update_hist(updated_hist, s_atom)
        # if the node should be masked, we do not update the current_histogram. In this way, it is always the last one
        # which represent the molecule.
        # new_updated_hist = tf.cond(tf.equal(current_mask_value, 0),
        #                            lambda: updated_hist,
        #                            lambda: self.update_hist(updated_hist, s_atom))

        if self.params['gen_hist_sampling']:
            # sampling one compatible histogram with the current new histogram
            reshape = tf.reshape(new_updated_hist,
                                 (-1, hist_dim))  # reshape the dimension from [n_valences] to [1, n_valences]
            m1 = self.placeholders['histograms'] >= reshape  # vector of 0 and 1

            m2 = tf.reduce_sum(tf.cast(m1, dtype=tf.int32), axis=1)  # [b]
            m3 = tf.equal(m2, tf.constant(hist_dim))  # [b]
            m4 = tf.cast(m3, dtype=tf.int32)  # [b]
            m5 = tf.multiply(self.placeholders['n_histograms'], m4)  # [b]
            mSomma = tf.reduce_sum(m5)
            new_sampled_hist = tf.cond(tf.equal(tf.constant(0), mSomma),
                                       lambda: self.case_random_sampling(),
                                       lambda: self.case_sampling(m5, mSomma))
        else:
            new_sampled_hist = sampled_hist

        s_atom, init_v, fx = tf.expand_dims(s_atom, 0), tf.squeeze(new_z_concat), node_probs

        atoms = atoms.write(idx_atom, s_atom)
        init_vertices = init_vertices.write(idx_atom, init_v)
        fx_prob = fx_prob.write(idx_atom, fx)
        return idx_atom + 1, atoms, init_vertices, fx_prob, idx_sample, new_updated_hist, new_sampled_hist

    def mask_mols(self, logits, hist):
        mol_valence_list = []
        for key in dataset_info(self.params['dataset'])['maximum_valence'].keys():
            mol_valence_list.append(dataset_info(self.params['dataset'])['maximum_valence'][key])
        mol_valence = tf.constant(mol_valence_list)
        H_b = tf.cast(hist > 0, tf.int32)  # obtaining a vector of only 1 and 0
        idx = tf.where(tf.less_equal(H_b, 0))  # obtaining al the indexes with 0-values in the hist
        valences = tf.cast(idx + 1, tf.int32)  # valences to avoid
        equals = tf.not_equal(mol_valence, valences)  # broadcasting.
        mask_bool = tf.reduce_all(equals, 0)
        mask = tf.cast(mask_bool, tf.float32)
        logits_masked = tf.cond(tf.reduce_any(mask_bool),
                                lambda: logits + (mask * utils.LARGE_NUMBER - utils.LARGE_NUMBER),
                                lambda: logits)
        return logits_masked, mask_bool

    """
    Histograms sampling with probs
    """

    def case_sampling(self, m5, mSomma):
        prob = m5 / mSomma
        m8 = tf.distributions.Categorical(probs=prob).sample()
        m9 = self.placeholders['histograms'][m8]
        return m9

    """
    Histograms uniform sampling
    """

    def case_random_sampling(self):
        max_n = tf.shape(self.placeholders['histograms'])[0]
        idx = tf.random_uniform([], maxval=max_n, dtype=tf.int32)
        return self.placeholders['histograms'][idx]

    """
    Sample the id of the atom for a value fo probabilities. 
    In training always apply argmax, while in generation or optimization it is possible to choose among distribution or argmax
    """

    def sample_atom(self, fx_prob, training):
        if training or self.params['generation'] == 3:
            idx = tf.argmax(fx_prob, output_type=tf.int32)
        else:
            if self.params['use_argmax_nodes']:
                idx = tf.argmax(fx_prob, output_type=tf.int32)
            else:
                idx = tf.distributions.Categorical(probs=fx_prob).sample()
        return idx

    """
    Update of the histogram according to the new atom.
    """

    def update_hist(self, old_hist, id_atom):
        hist_dim = self.histograms['hist_dim']
        mol_valence_list = []
        # this is ok even if the dictionary is not ordered
        for key in dataset_info(self.params['dataset'])['maximum_valence'].keys():
            mol_valence_list.append(dataset_info(self.params['dataset'])['maximum_valence'][key])
        # make the mol-valence array
        mol_valence = tf.constant(mol_valence_list)
        # take the atom valence
        atmo_val = mol_valence[id_atom]
        # build an array to be used in add operation
        array = tf.one_hot(atmo_val - 1, hist_dim, dtype=tf.int32)  # remember that valence start from 1
        # summing the two array
        new_hist = tf.add(old_hist, array)
        return new_hist

    def construct_logit_matrices(self):
        v = self.placeholders['num_vertices']
        batch_size = tf.shape(self.ops['latent_node_symbols'])[0]
        # prep valences
        mol_valence_list = []
        for key in dataset_info(self.params['dataset'])['maximum_valence'].keys():
            mol_valence_list.append(dataset_info(self.params['dataset'])['maximum_valence'][key])
        mol_valence = tf.constant(mol_valence_list)
        indexes = tf.argmax(self.ops['latent_node_symbols'], axis=-1)  # [b, v]
        valences = tf.nn.embedding_lookup(mol_valence, indexes)

        # representation in input to edge decoder
        latent_node_state = self.get_node_embedding_state(self.ops['latent_node_symbols'])
        filtered_z_sampled = tf.concat([self.ops['initial_nodes_decoder'], latent_node_state], axis=2)
        self.ops["initial_repre_for_decoder"] = filtered_z_sampled  # [b, v, 2h]

        #    The tensor array used to collect the cross entropy losses at each step
        edges_pred = tf.TensorArray(dtype=tf.float32, size=v)
        edges_type_pred = tf.TensorArray(dtype=tf.float32, size=v)
        idx_final, edges_pred, edges_type_pred, _ = \
            tf.while_loop(
                lambda idx, edges_pred, edges_type_pred, valences: idx < v, self.generate_edges,
                (tf.constant(0), edges_pred, edges_type_pred, valences)
            )
        self.ops['edges_pred'] = tf.transpose(edges_pred.stack(), [1, 0, 2]) * self.ops['graph_state_mask']
        self.ops['edges_type_pred'] = tf.transpose(edges_type_pred.stack(), [1, 3, 0, 2]) * \
                                      tf.expand_dims(self.ops['graph_state_mask'], 1)

        # mask diagonal in order to put all probabilities to 1 in the non existence of the edge
        diag_0 = tf.one_hot(tf.range(v), depth=v, on_value=0.0, off_value=1.0, dtype=tf.float32)
        self.ops['edges_pred'] = self.ops['edges_pred'] * tf.expand_dims(diag_0, 0)
        self.ops['edges_type_pred'] = self.ops['edges_type_pred'] * tf.expand_dims(tf.expand_dims(diag_0, 0), 0)

        tf.summary.histogram("edges_pred", self.ops['edges_pred'])
        tf.summary.histogram("edges_type_pred", self.ops['edges_type_pred'])

        # calc the GT to use
        gt_edges_pred = tf.reduce_sum(self.placeholders['adjacency_matrix'], axis=1)
        gt_edges_type_pred = self.placeholders['adjacency_matrix']

        # binary cross-entropy balanced
        n_edges = dataset_info(self.params['dataset'])['n_edges']
        n_yes_edges = dataset_info(self.params['dataset'])['loss_yes_edge']
        n_no_edges = dataset_info(self.params['dataset'])['loss_no_edge']
        edge_loss = - tf.reduce_sum(
            (tf.log(self.ops['edges_pred'] + utils.SMALL_NUMBER) * gt_edges_pred) * n_yes_edges +
            tf.log((1 - self.ops['edges_pred']) + utils.SMALL_NUMBER) * (1 - gt_edges_pred) * n_no_edges,
            axis=[1, 2])

        # edge type cross entropy balanced
        n_edges = [dataset_info(self.params['dataset'])['loss_edge_weights']]  # added a dimension
        loss_batchEdge = tf.reduce_sum(tf.log(self.ops['edges_type_pred'] + utils.SMALL_NUMBER) * gt_edges_type_pred,
                                       axis=[2, 3])
        edge_type_loss = -tf.reduce_sum(loss_batchEdge * n_edges, axis=-1)

        # sum losses
        self.ops['cross_entropy_losses'] = edge_loss + edge_type_loss
        # self.ops['cross_entropy_losses'] = 2*(2*edge_loss + edge_type_loss)

        corr_edge = tf.cast(self.ops['edges_pred'] >= 0.5, tf.float32)
        corr_edge = tf.cast(tf.not_equal(corr_edge, gt_edges_pred),
                            tf.float32)
        edges_type_pred_masked = self.ops['edges_type_pred'] * tf.expand_dims(gt_edges_pred, axis=1)
        corr_type_edge = tf.cast(tf.not_equal(tf.argmax(edges_type_pred_masked, axis=1),
                                              tf.argmax(gt_edges_type_pred, axis=1)),
                                 tf.float32)
        self.ops['edge_pred_error'] = tf.reduce_sum(corr_edge, axis=[1, 2])
        self.ops['edge_type_pred_error'] = tf.reduce_sum(corr_type_edge, axis=[1, 2])

    def generate_edges(self, idx, edges_pred, edges_type_pred, valences):
        v = self.placeholders['num_vertices']
        h_dim_en = self.params['hidden_size_encoder']
        latent_space_dim = self.params['latent_space_size']
        h_dim_de = latent_space_dim + 50
        batch_size = tf.shape(self.ops['latent_node_symbols'])[0]

        edges_val_req = [i + 1 for i in range(0, self.num_edge_types)]
        edges_val_req = tf.expand_dims(edges_val_req, 0)
        edges_val_req = tf.expand_dims(edges_val_req, 0)
        edges_val_req = tf.tile(edges_val_req, [batch_size, v, 1])

        filtered_z_sampled = self.ops["initial_repre_for_decoder"]

        # graph features
        graph_sum = tf.reduce_mean(filtered_z_sampled, axis=1, keep_dims=True)  # [b, 1, 2h]
        graph_sum = tf.tile(graph_sum, [1, v, 1])
        # graph_prod = tf.reduce_prod(filtered_z_sampled, axis=1, keep_dims=True)  # [b, 1, 2h]
        # graph_prod = tf.tile(graph_prod, [1, v, 1])

        # node in focus feature
        node_focus = filtered_z_sampled[:, idx, :]
        node_focus = tf.tile(tf.expand_dims(node_focus, axis=1), [1, v, 1])
        node_focus_sum = node_focus + filtered_z_sampled
        node_focus_prod = node_focus * filtered_z_sampled  # [b, v, 2h]
        # input_features = tf.concat([node_focus_sum, node_focus_prod, graph_sum, graph_prod], axis=-1)
        # dim_input_network = 4 * (h_dim_en + h_dim_de)
        input_features = tf.concat([node_focus_sum, node_focus_prod, graph_sum], axis=-1)
        dim_input_network = 3 * (h_dim_en + h_dim_de)
        # input_features = tf.concat([node_focus, filtered_z_sampled, graph_sum], axis=-1)
        # dim_input_network = 3 * (h_dim_en + h_dim_de)

        # node in focus valences
        node_focus_valences = valences[:, idx]
        node_focus_valences = tf.expand_dims(node_focus_valences, axis=1)
        node_focus_feature_valences = tf.tile(node_focus_valences, [1, v])

        # generate mask
        mask_min = tf.stack([node_focus_feature_valences, valences], axis=-1)
        mask_min = tf.reduce_min(mask_min, -1)
        mask_min = tf.tile(tf.expand_dims(mask_min, 2), [1, 1, self.num_edge_types])
        mask = tf.cast(edges_val_req <= mask_min, tf.float32)
        mask = tf.reshape(mask, [-1, self.num_edge_types])

        # edge prediction
        edge_rep = tf.reshape(input_features, [-1, dim_input_network])
        edge_pred_tmp = self.weights['edge_gen_MLP'](edge_rep, self.placeholders['is_training'])
        edge_pred_tmp = tf.nn.sigmoid(edge_pred_tmp)
        edge_pred_tmp = tf.reshape(edge_pred_tmp, [batch_size, v, 1]) * self.ops['graph_state_mask']
        edge_pred_tmp = tf.squeeze(edge_pred_tmp, axis=-1)

        # edge type prediction
        edge_type_pred_tmp = self.weights['edge_type_gen_MLP'](edge_rep, self.placeholders['is_training'])
        edge_type_pred_tmp = tf.nn.softmax(edge_type_pred_tmp + (mask * utils.LARGE_NUMBER - utils.LARGE_NUMBER))
        edge_type_pred_tmp = tf.reshape(edge_type_pred_tmp, [batch_size, v, self.num_edge_types]) * self.ops[
            'graph_state_mask']

        edges_pred = edges_pred.write(idx, edge_pred_tmp)
        edges_type_pred = edges_type_pred.write(idx, edge_type_pred_tmp)
        return idx + 1, edges_pred, edges_type_pred, valences

    def fully_connected(self, input, hidden_weight, hidden_bias, output_weight):
        output = tf.nn.relu(tf.matmul(input, hidden_weight) + hidden_bias)
        output = tf.matmul(output, output_weight)
        return output

    def construct_loss(self):
        v = self.placeholders['num_vertices']
        ls_dim = self.params['latent_space_size']
        h_dim_de = ls_dim + 50
        h_dim_en = self.params['hidden_size_encoder']
        kl_trade_off_lambda_start = self.params['kl_trade_off_lambda']
        kl_trade_off_lambda_factor = self.params['kl_trade_off_lambda_factor']
        n_epoch = self.placeholders['n_epoch']
        kl_trade_off_lambda = kl_trade_off_lambda_start + kl_trade_off_lambda_factor * tf.cast(n_epoch, tf.float32)

        # Edge loss
        self.ops["edge_loss"] = self.ops['cross_entropy_losses']

        # KL loss
        kl_loss = 1 + self.ops['logvariance'] - tf.square(self.ops['mean']) - tf.exp(self.ops['logvariance'])
        kl_loss = tf.reshape(kl_loss, [-1, v, ls_dim]) * self.ops['graph_state_mask']
        self.ops['kl_loss'] = -0.5 * tf.reduce_sum(kl_loss, [1, 2])

        # Node symbol loss
        loss_node_weights = [
            [dataset_info(self.params['dataset'])['loss_node_weights']]]  # added two dimension [1, 1, t_nodes]
        node_symbol_loss = tf.log(self.ops['node_symbol_prob'] + utils.SMALL_NUMBER) * self.placeholders['node_symbols']
        self.ops['node_symbol_loss'] = -tf.reduce_sum(node_symbol_loss * loss_node_weights, axis=[1, 2])

        # Other statistics
        latent_node_symbol = tf.cast(tf.not_equal(tf.argmax(self.ops['node_symbol_prob'], axis=-1),
                                                  tf.argmax(self.placeholders['node_symbols'], axis=-1)),
                                     tf.float32)
        mols_errors = self.ops['edge_pred_error'] + self.ops['edge_type_pred_error'] + tf.reduce_sum(latent_node_symbol,
                                                                                                     axis=-1)
        self.ops['reconstruction'] = tf.reduce_sum(tf.cast(tf.equal(mols_errors, 0), tf.float32))
        # after because it rewrite the operations
        self.ops['node_pred_error'] = tf.reduce_mean(latent_node_symbol)
        self.ops['edge_pred_error'] = tf.reduce_mean(self.ops['edge_pred_error'])
        self.ops['edge_type_pred_error'] = tf.reduce_mean(self.ops['edge_type_pred_error'])

        # Add in the loss for calculating QED
        for (internal_id, task_id) in enumerate(self.params['task_ids']):
            with tf.variable_scope("out_layer_task%i" % task_id):
                input_size_qed = h_dim_de + h_dim_en
                with tf.variable_scope("regression_gate"):
                    self.weights['regression_gate_task%i' % task_id] = MLP(input_size_qed, 1, [],
                                                                           self.placeholders[
                                                                               'out_layer_dropout_keep_prob'],
                                                                           activation_function=tf.nn.leaky_relu)
                with tf.variable_scope("regression"):
                    self.weights['regression_transform_task%i' % task_id] = MLP(input_size_qed, 1, [],
                                                                                self.placeholders[
                                                                                    'out_layer_dropout_keep_prob'],
                                                                                activation_function=tf.nn.leaky_relu)
                histograms_input = tf.cast(self.placeholders['hist'], tf.float32)
                v = self.placeholders['num_vertices']  # bucket size dimension, not all time the real one.
                # histograms = tf.tile(tf.expand_dims(histograms_input, 1), [1, v, 1]) * self.ops['graph_state_mask']
                initial_nodes_decoder = self.ops['initial_repre_for_decoder']
                if task_id == 0:
                    computed_values = self.gated_regression_QED(initial_nodes_decoder,
                                                                self.weights['regression_gate_task%i' % task_id],
                                                                self.weights['regression_transform_task%i' % task_id],
                                                                input_size_qed,
                                                                self.weights['qed_weights'],
                                                                self.weights['qed_biases'],
                                                                self.placeholders['num_vertices'],
                                                                self.placeholders['node_mask'])
                    self.ops['qed_computed_values'] = computed_values
                else:
                    computed_values = self.gated_regression_plogP(initial_nodes_decoder,
                                                                  self.weights['regression_gate_task%i' % task_id],
                                                                  self.weights['regression_transform_task%i' % task_id],
                                                                  input_size_qed,
                                                                  self.weights['plogP_weights'],
                                                                  self.weights['plogP_biases'],
                                                                  self.placeholders['num_vertices'],
                                                                  self.placeholders['node_mask'])
                    self.ops['plogp_computed_values'] = computed_values
                diff = computed_values - self.placeholders['target_values'][internal_id, :]  # [b]
                task_target_mask = self.placeholders['target_mask'][internal_id, :]
                task_target_num = tf.reduce_sum(task_target_mask) + utils.SMALL_NUMBER
                diff = diff * task_target_mask  # Mask out unused values [b]
                self.ops['accuracy_task%i' % task_id] = tf.reduce_sum(tf.abs(diff)) / task_target_num
                task_loss = tf.reduce_sum(0.5 * tf.square(diff)) / task_target_num  # number
                # Normalise loss to account for fewer task-specific examples in batch:
                task_loss = task_loss * (1.0 / (self.params['task_sample_ratios'].get(task_id) or 1.0))
                self.ops['qed_loss'].append(task_loss)  # note that is the mean, not only the sum of all the batch
                initial_nodes_decoder_shape = tf.shape(self.ops['initial_repre_for_decoder'])
                if task_id == 0:  # Assume it is the QED score
                    flattened_z = tf.reshape(initial_nodes_decoder, [initial_nodes_decoder_shape[0], -1])
                    self.ops['l2_loss'] = 0.01 * tf.reduce_sum(flattened_z * flattened_z, axis=1) / 2
                    # Calculate the derivative with respect to QED + l2 loss
                    self.ops['derivative_z_sampled'] = tf.gradients(
                        self.ops['qed_computed_values'] - self.ops['l2_loss'],
                        self.placeholders['z_prior'])
                    self.ops['derivative_hist'] = tf.gradients(self.ops['qed_computed_values'] - self.ops['l2_loss'],
                                                               histograms_input)
                elif task_id == 1:  # Assume it is the plogP score
                    flattened_z = tf.reshape(initial_nodes_decoder, [initial_nodes_decoder_shape[0], -1])
                    self.ops['l2_loss_plop'] = 0.01 * tf.reduce_sum(flattened_z * flattened_z, axis=1) / 2
                    # Calculate the derivative with respect to QED + l2 loss
                    self.ops['derivative_z_sampled_plop'] = tf.gradients(
                        self.ops['plogp_computed_values'] - self.ops['l2_loss'],
                        self.placeholders['z_prior'])
                    self.ops['derivative_hist_plogp'] = tf.gradients(
                        self.ops['plogp_computed_values'] - self.ops['l2_loss'],
                        histograms_input)
        self.ops['total_qed_loss'] = tf.reduce_sum(
            self.ops['qed_loss'])  # number representing the sum of the mean of the loss for each property
        self.ops['mean_edge_loss'] = tf.reduce_mean(self.ops["edge_loss"])  # record the mean edge loss
        self.ops['mean_node_symbol_loss'] = tf.reduce_mean(self.ops["node_symbol_loss"])
        self.ops['mean_kl_loss'] = tf.reduce_mean(self.ops['kl_loss'])
        self.ops['mean_total_qed_loss'] = 1000 * self.ops['total_qed_loss']

        # 0.01 - keras default scalar
        # l2_loss = 0
        # for iter_idx in range(self.params['num_timesteps']):
        #     for edge_type in range(self.num_edge_types):
        #          l2_loss += self.weights['MLP_edge' + str(edge_type) + '_encoder' + str(iter_idx)].cal_l2_loss()

        loss = tf.reduce_mean(self.ops["edge_loss"] + self.ops['node_symbol_loss']
                              + kl_trade_off_lambda * self.ops['kl_loss']) \
               + self.params["qed_trade_off_lambda"] * self.ops['total_qed_loss']
        # tf summary
        tf.summary.scalar('total_qed_loss', self.ops['total_qed_loss'])
        tf.summary.scalar('mean_edge_loss', self.ops['mean_edge_loss'])
        tf.summary.scalar('mean_node_symbol_loss', self.ops['mean_node_symbol_loss'])
        tf.summary.scalar('mean_kl_loss', self.ops['mean_kl_loss'])
        tf.summary.scalar('mean_total_qed_loss', self.ops['mean_total_qed_loss'])
        tf.summary.scalar('loss', self.ops['total_qed_loss'])
        tf.summary.scalar('reconstruction', tf.reduce_mean(tf.cast(tf.equal(mols_errors, 0), tf.float32)))
        tf.summary.scalar('node_pred_error', self.ops['node_pred_error'])
        tf.summary.scalar('edge_pred_error', self.ops['edge_pred_error'])
        tf.summary.scalar('edge_type_pred_error', self.ops['edge_type_pred_error'])

        return loss

    def gated_regression_QED(self, last_h, regression_gate, regression_transform, hidden_size, projection_weight,
                             projection_bias, v, mask):
        # last_h: [b x v x h]
        last_h = tf.reshape(last_h, [-1, hidden_size])  # [b*v, h]
        # linear projection on last_h
        last_h = tf.nn.leaky_relu(tf.matmul(last_h, projection_weight) + projection_bias)  # [b*v, h]
        # same as last_h
        gate_input = last_h
        # linear projection and combine                                       
        gated_outputs = tf.nn.sigmoid(regression_gate(gate_input)) * tf.nn.tanh(
            regression_transform(last_h))  # [b*v, 1]
        gated_outputs = tf.reshape(gated_outputs, [-1, v])  # [b, v]
        masked_gated_outputs = gated_outputs * mask  # [b x v]
        output = tf.reduce_sum(masked_gated_outputs, axis=1)  # [b]
        output = tf.sigmoid(output)
        return output

    def gated_regression_plogP(self, last_h, regression_gate, regression_transform, hidden_size, projection_weight,
                               projection_bias, v, mask):
        # last_h: [b x v x h]
        last_h = tf.reshape(last_h, [-1, hidden_size])  # [b*v, h]
        # linear projection on last_h
        last_h = tf.nn.leaky_relu(tf.matmul(last_h, projection_weight) + projection_bias)  # [b*v, h]
        # same as last_h
        gate_input = last_h
        # linear projection and combine
        gated_outputs = tf.nn.sigmoid(regression_gate(gate_input)) * regression_transform(last_h)  # [b*v, 1]
        gated_outputs = tf.reshape(gated_outputs, [-1, v])  # [b, v]
        masked_gated_outputs = gated_outputs * mask  # [b x v]
        output = tf.reduce_sum(masked_gated_outputs, axis=1)  # [b]
        output = output
        return output

    # ----- Data preprocessing and chunking into minibatches:
    def process_raw_graphs(self, raw_data, is_training_data, file_name, bucket_sizes=None):
        if bucket_sizes is None:
            bucket_sizes = dataset_info(self.params["dataset"])["bucket_sizes"]
        bucketed = defaultdict(list)
        x_dim = len(raw_data[0]["node_features"][0])
        hist_dim = dataset_info(dataset)['hist_dim']

        for idx, d in enumerate(raw_data):
            # choose a bucket
            chosen_bucket_idx = np.argmax(bucket_sizes > max([v for e in d['graph']
                                                              for v in [e[0], e[2]]]))
            chosen_bucket_size = bucket_sizes[chosen_bucket_idx]

            # calc incremental hist for node
            # incr_hist, incr_diff_hist, incr_node_mask = incr_node(d, self.params['dataset'])
            incr_hist, incr_diff_hist, incr_node_mask = d['incr_node']

            # total number of nodes in this data point
            n_active_nodes = len(d["node_features"])
            bucketed[chosen_bucket_idx].append({
                'smiles': d['smiles'],
                'adj_mat': utils.graph_to_adj_mat(d['graph'], chosen_bucket_size, self.num_edge_types,
                                                  self.params['tie_fwd_bkwd']),
                'init': d["node_features"] + [[0 for _ in range(x_dim)] for __ in
                                              range(chosen_bucket_size - n_active_nodes)],
                'labels': [d["targets"][task_id][0] for task_id in self.params['task_ids']],
                'mask': [1. for _ in range(n_active_nodes)] + [0. for _ in range(chosen_bucket_size - n_active_nodes)],
                'hist': d['hist'],
                'incr_hist': incr_hist + [[0 for _ in range(hist_dim)] for __ in
                                          range(chosen_bucket_size - n_active_nodes)],
                'incr_diff_hist': incr_diff_hist + [[0 for _ in range(hist_dim)] for __ in
                                                    range(chosen_bucket_size - n_active_nodes)],
                'incr_node_mask': incr_node_mask + [[0 for _ in range(x_dim)] for __ in
                                                    range(chosen_bucket_size - n_active_nodes)],
            })
            print('Finish preprocessing %d graph' % idx, end="\r")

        if is_training_data:
            for (bucket_idx, bucket) in bucketed.items():
                np.random.shuffle(bucket)
                for task_id in self.params['task_ids']:
                    task_sample_ratio = self.params['task_sample_ratios'].get(str(task_id))
                    if task_sample_ratio is not None:
                        ex_to_sample = int(len(bucket) * task_sample_ratio)
                        for ex_id in range(ex_to_sample, len(bucket)):
                            bucket[ex_id]['labels'][task_id] = None

        bucket_at_step = [[bucket_idx for _ in range(len(bucket_data) // self.params['batch_size'])]
                          for bucket_idx, bucket_data in bucketed.items()]

        # every position indicates the bucket size
        bucket_at_step = [x for y in bucket_at_step for x in y]

        return (bucketed, bucket_sizes, bucket_at_step)

    def make_batch(self, elements, maximum_vertice_num):
        batch_data = {'smiles': [], 'adj_mat': [], 'init': [], 'labels': [], 'node_mask': [], 'task_masks': [],
                      'hist': [], 'incr_hist': [], 'incr_diff_hist': [], 'incr_node_mask': []}
        for d in elements:
            batch_data['adj_mat'].append(d['adj_mat'])
            batch_data['init'].append(d['init'])
            batch_data['node_mask'].append(d['mask'])
            batch_data['hist'].append(d['hist'])
            batch_data['incr_hist'].append(d['incr_hist'])
            batch_data['incr_diff_hist'].append(d['incr_diff_hist'])
            batch_data['incr_node_mask'].append(d['incr_node_mask'])
            batch_data['smiles'].append(d['smiles'])

            target_task_values = []
            target_task_mask = []
            for target_val in d['labels']:
                if target_val is None:  # This is one of the examples we didn't sample...
                    target_task_values.append(0.)
                    target_task_mask.append(0.)
                else:
                    target_task_values.append(target_val)
                    target_task_mask.append(1.)
            batch_data['labels'].append(target_task_values)
            batch_data['task_masks'].append(target_task_mask)
        return batch_data

    """
    Prepare the feed dict for obtaining the nodes (atoms) during generation
    """

    def get_dynamic_feed_dict(self, elements, latent_node_symbol, incre_adj_mat, num_vertices,
                              distance_to_others, overlapped_edge_dense, node_sequence, edge_type_masks, edge_masks,
                              random_normal_states, current_hist, values):
        if incre_adj_mat is None:
            latent_node_symbol = np.zeros((1, 1, self.params["num_symbols"]))
        return {
            self.placeholders['z_prior']: random_normal_states,  # [1, v, h]
            self.placeholders['num_vertices']: num_vertices,  # v
            # self.placeholders['node_symbols']: [elements['init']],
            self.ops['latent_node_symbols']: latent_node_symbol,
            self.placeholders['adjacency_matrix']: [elements['adj_mat']],
            self.placeholders['node_mask']: [elements['mask']],
            # self.placeholders['graph_state_keep_prob']: 1,
            # self.placeholders['edge_weight_dropout_keep_prob']: 1,
            self.placeholders['out_layer_dropout_keep_prob']: 1.0,
            self.placeholders['histograms']: self.histograms['train'][0],
            self.placeholders['n_histograms']: values,  # frequencies for sampling
            self.placeholders['hist']: [current_hist],
        }

    """
    Prepare the feed dict for accessing the nodes (atoms)
    """

    def get_dynamic_nodes_feed_dict(self, elements, num_vertices, z_sampled):
        return {
            self.ops['z_sampled']: z_sampled,  # [hl]
            self.placeholders['num_vertices']: num_vertices,  # v
            self.placeholders['node_symbols']: [elements['init']],
            self.placeholders['node_mask']: [elements['mask']],
            # self.placeholders['graph_state_keep_prob']: 1,
            # self.placeholders['edge_weight_dropout_keep_prob']: 1,
            self.placeholders['out_layer_dropout_keep_prob']: 1.0,
            self.placeholders['histograms']: self.histograms['train'][0],
            self.placeholders['n_histograms']: self.histograms['train'][1],
            self.placeholders['hist']: [elements['hist']],
        }

    """
    Prepare the feed dict for searching the edges amongs atoms
    """

    def get_dynamic_edge_feed_dict(self, elements, latent_nodes, latent_node_symbol, num_vertices):
        return {
            self.placeholders['num_vertices']: num_vertices,  # v
            self.ops['initial_nodes_decoder']: latent_nodes,
            self.ops['latent_node_symbols']: latent_node_symbol,
            # self.placeholders['adjacency_matrix']: [elements['adj_mat']],
            self.placeholders['node_mask']: [elements['mask']],
            # self.placeholders['graph_state_keep_prob']: 1,
            # self.placeholders['edge_weight_dropout_keep_prob']: 1,
            self.placeholders['out_layer_dropout_keep_prob']: 1.0,
        }

    """
    Prepare the feed dict for accessing the sampling point in the latent space
    """

    def get_dynamic_mean_feed_dict(self, elements, num_vertices, latent_points):
        return {
            self.placeholders['z_prior']: latent_points,  # [hl]
            self.placeholders['num_vertices']: num_vertices,  # v
            self.placeholders['node_mask']: [elements['mask']],
            self.placeholders['node_symbols']: [elements['init']],
            self.placeholders['adjacency_matrix']: [elements['adj_mat']],
            self.placeholders['graph_state_keep_prob']: 1,
            self.placeholders['edge_weight_dropout_keep_prob']: 1,
            self.placeholders['out_layer_dropout_keep_prob']: 1.0,
        }

    def get_node_symbol(self, batch_feed_dict):
        fetch_list = [self.ops['initial_nodes_decoder'], self.ops['node_symbol_prob'], self.ops['sampled_atoms']]
        result = self.sess.run(fetch_list, feed_dict=batch_feed_dict)
        return result

    def node_symbol_one_hot(self, sampled_node_symbol, real_n_vertices, max_n_vertices):
        one_hot_representations = []
        for idx in range(max_n_vertices):
            representation = [0] * self.params["num_symbols"]
            if idx < real_n_vertices:
                atom_type = sampled_node_symbol[idx]
                representation[atom_type] = 1
            one_hot_representations.append(representation)
        return one_hot_representations

    def search_and_generate_molecule(self, valences,
                                     sampled_node_symbol, real_n_vertices,
                                     elements, max_n_vertices, latent_nodes):
        # New molecule
        new_mol = Chem.MolFromSmiles('')
        new_mol = Chem.rdchem.RWMol(new_mol)
        # Add atoms
        add_atoms(new_mol, sampled_node_symbol, self.params["dataset"])
        # Add edges
        sampled_node_symbol_one_hot = self.node_symbol_one_hot(sampled_node_symbol, real_n_vertices, max_n_vertices)

        # get feed_dict
        feed_dict = self.get_dynamic_edge_feed_dict(elements, latent_nodes, [sampled_node_symbol_one_hot],
                                                    max_n_vertices)
        # fetch nn predictions
        fetch_list = [self.ops['edges_pred'], self.ops['edges_type_pred']]
        edge_probs, edge_type_probs = self.sess.run(fetch_list, feed_dict=feed_dict)
        edge_probs = edge_probs[0]
        edge_type_probs = edge_type_probs[0]
        edge_probs_bin = edge_probs > 0.5

        list_probs = []
        for row in range(len(edge_probs[0])):
            for col in range(row + 1, len(edge_probs[0])):  # only half matrix
                list_probs.append((edge_probs[row, col], row, col))

        list_probs.sort(key=lambda tup: tup[0], reverse=True)
        for trip in range(len(list_probs)):
            prob, row, col = list_probs[trip]
            if prob > 0.5:  # remember edges that are masked.
                continue_arch = True
                while continue_arch:
                    # choose an edge type
                    if not self.params["use_argmax_bonds"]:
                        bond = np.random.choice(np.arange(self.num_edge_types), p=edge_type_probs[:, row, col])
                    else:
                        bond = np.argmax(edge_type_probs[:, row, col])
                    # add the bond if valence is ok else take the second best and so on.
                    if self.params['fix_molecule_validation']:
                        if min(valences[row], valences[col]) >= bond + 1:
                            new_mol.AddBond(int(row), int(col), number_to_bond[bond])
                            valences[row] -= bond + 1
                            valences[col] -= bond + 1
                            continue_arch = False
                        else:
                            # remove this bond type
                            edge_type_probs[bond, row, col] = 0
                            sum_prob = sum(edge_type_probs[:, row, col])
                            if sum_prob > 0:
                                edge_type_probs[:, row, col] = edge_type_probs[:, row, col] / sum_prob
                            else:
                                continue_arch = False
                            # continue_arch = False

                    else:
                        new_mol.AddBond(int(row), int(col), number_to_bond[bond])
                        continue_arch = False

        # Remove unconnected node
        utils.remove_extra_nodes(new_mol)
        # new_mol.UpdatePropertyCache(strict=False)  # needed for the following command
        # Chem.AssignStereochemistry(new_mol, force=True, cleanIt=False)  # fix properties
        smiles = Chem.MolToSmiles(new_mol)
        new_mol = Chem.MolFromSmiles(smiles)
        # if new_mol is None:
        #     pass
        return new_mol

    def gradient_ascent(self, random_normal_states, derivative_z_sampled):
        return random_normal_states + self.params['prior_learning_rate'] * derivative_z_sampled

    """
    Optimization in latent space. Generate one molecule for each optimization step.
    """

    def optimization_over_prior(self, random_normal_states, num_vertices, generated_all_similes, generated_all_QED,
                                elements, count):
        # record how many optimization steps are taken
        current_smiles = []
        step = 0
        SMILES = []
        QED_values = []
        # fix the choice of the first histogram)
        hist_prob_per_num_atoms = self.histograms['filter'][1][int(num_vertices)]
        hist_freq_per_num_atoms = self.histograms['filter'][0][int(num_vertices)]
        prob_sum = np.sum(hist_prob_per_num_atoms)
        # if there are no histograms with the same number (or higher) of atoms, we use all the histograms
        #   otherwise we give only the histograms with at least the same number fo atoms
        if prob_sum == 0:
            sampled_idx_hist = np.random.choice(len(self.histograms['train'][0]))
            hist_freq_per_num_atoms = self.histograms['train'][1]
        else:
            sampled_idx_hist = np.random.choice(len(self.histograms['train'][0]), p=hist_prob_per_num_atoms)
        # generate a new molecule
        current_hist = self.histograms['train'][0][sampled_idx_hist]
        temp = self.generate_graph_with_state(random_normal_states, num_vertices, generated_all_similes, elements, step,
                                              count, current_hist, hist_freq_per_num_atoms)
        SMILES.append(temp)
        # fetch_list = [self.ops['derivative_z_sampled'], self.ops['qed_computed_values'], self.ops['derivative_hist']]
        if self.params['task_ids'][0] == 0:
            fetch_list = [self.ops['derivative_z_sampled'], self.ops['qed_computed_values']]
            current_function = QED.qed
        else:
            fetch_list = [self.ops['derivative_z_sampled_plogP'], self.ops['plogP_computed_values']]
            current_function = utils.penalized_logP
        for _ in range(self.params['optimization_step']):
            # get current qed and derivative
            batch_feed_dict = self.get_dynamic_feed_dict(elements, None, None,
                                                         num_vertices, None, None, None, None, None,
                                                         random_normal_states, current_hist,
                                                         hist_freq_per_num_atoms)
            derivative_z_sampled, qed_computed_values = self.sess.run(fetch_list, feed_dict=batch_feed_dict)
            # print("grad_hist: ", grad_hist)
            tmp_mol = Chem.MolFromSmiles(SMILES[step])
            if tmp_mol is not None:
                QED_values.append([current_function(tmp_mol), qed_computed_values])
            else:
                QED_values.append([0, 0])
                # print("Optimization: ", step, " - ", "None molecule warning", end="\n")
            # update the states
            random_normal_states = self.gradient_ascent(random_normal_states, derivative_z_sampled[0])
            # do local search on histograms
            # if self.params['hist_local_search']:
            #     grad_hist_clean = grad_hist[0].tolist()[0]
            #     abs_hist_grads = [abs(i) for i in grad_hist_clean]
            #     max_hist_val = max(abs_hist_grads)
            #     max_hist_idx = abs_hist_grads.index(max_hist_val)
            #     if abs_hist_grads[max_hist_idx] != 0:
            #         if grad_hist_clean[max_hist_idx] > 0:
            #             current_hist[max_hist_idx] += 1
            #             print("Hist update:: ", current_hist)
            #         elif grad_hist_clean[max_hist_idx] < 0 and current_hist[max_hist_idx] > 0:
            #             current_hist[max_hist_idx] -= 1
            #             print("Hist update:: ", current_hist)
            # generate a new molecule
            step += 1
            temp_opt = self.generate_graph_with_state(random_normal_states, num_vertices, generated_all_similes,
                                                      elements, step, count, current_hist, hist_freq_per_num_atoms)
            SMILES.append(temp_opt)

        generated_all_similes.append(SMILES)
        generated_all_QED.append(QED_values)
        return random_normal_states

    def optimization(self, data):
        # bucketed: data organized by bucket
        (bucketed, bucket_sizes, bucket_at_step) = data
        # all generated similes
        generated_all_similes = []
        generated_all_QED = []
        # counter
        count = 0
        # shuffle the lengths
        np.random.shuffle(bucket_at_step)
        for step in range(len(bucket_at_step)):
            bucket = bucket_at_step[step]  # bucket number
            # data index
            idx = random.randint(0, len(bucketed[bucket]) - 1)
            elements_original = bucketed[bucket][idx]
            elements = copy.deepcopy(elements_original)
            maximum_length = bucket_sizes[bucket]
            if self.params['compensate_num'] > 0:
                maximum_length = self.compensate_node_length(elements, bucket_sizes[bucket])
            # initial state
            random_normal_states = utils.generate_std_normal(1, maximum_length,
                                                             self.params['latent_space_size'])  # [1, h]
            random_global_normal_states = np.random.normal(0, 1, [1, self.params['latent_space_size']])
            random_normal_states = self.optimization_over_prior(random_normal_states, random_global_normal_states,
                                                                maximum_length,
                                                                generated_all_similes, generated_all_QED,
                                                                elements, count)
            n_gen_max = self.params['number_of_generation']
            n_gen_cur = len(generated_all_similes)
            print("Molecules optimized: ", n_gen_cur, end='\r')
            if n_gen_cur >= n_gen_max:
                # analysis
                generated_smiles = np.array(generated_all_similes)
                generated_QED = np.array(generated_all_QED)
                generated_QED_delta = np.nanmax(generated_QED[:, :, 0], axis=1) - generated_QED[:, 0, 0]
                generated_QED_max = np.nanmax(generated_QED[:, :, 0], axis=1)
                generated_QED_argmax = np.nanargmax(generated_QED[:, :, 0], axis=1)

                best_smiles = generated_smiles[np.arange(0, generated_QED.shape[0]), generated_QED_argmax]
                # print(np.nanmax(generated_QED[:, :, 0], axis=1))

                # save
                suff = "_" + self.params['suffix'] if self.params['suffix'] is not None else ""
                mask = "_masked" if self.params['use_mask'] else "_noMask"
                log_dir = self.params['log_dir']
                priors_file = log_dir + "/" + str(dataset) + "_optimization_" \
                              + str(self.params['number_of_generation']) + "_" \
                              + str(self.params['optimization_step']) \
                              + mask + suff + ".txt"
                f = open(priors_file, "w")
                val_triple = zip(generated_QED[:, 0, 0], generated_QED_max, generated_QED_delta, best_smiles.flatten())
                # val_triple_sorted = val_triple.sort(key=lambda x: x[1] if x[2] > 0 else 0, reverse=True)
                val_triple_sorted = sorted(val_triple,
                                           key=lambda x: x[1] if x[1] is not None else 0,
                                           reverse=True)
                for qed_init, qed_max, qed_delta, best_smiles in val_triple_sorted[:5]:
                    # for val in line:
                    #     f.write(str(val))
                    #     f.write(", ")
                    f.write(str(qed_init))
                    f.write(", ")
                    f.write(str(qed_max))
                    f.write(", ")
                    f.write(str(qed_delta))
                    f.write(", ")
                    f.write(str(best_smiles))
                    f.write("\n")
                f.close()
                print("Optimization done")
                exit(0)
            count += 1

    def generation_new_graphs(self, random_normal_states, random_global_normal_states, num_vertices,
                              generated_all_similes, elements, count):
        # record how many optimization steps are taken
        step = 0
        # fix the choice of the first histogram)
        hist_prob_per_num_atoms = self.histograms['filter'][1][int(num_vertices)]
        hist_freq_per_num_atoms = self.histograms['filter'][0][int(num_vertices)]
        prob_sum = np.sum(hist_prob_per_num_atoms)
        # if there are no histograms with the same number (or higher) of atoms, we use all the histograms
        #   otherwise we give only the histograms with at least the same number fo atoms
        if prob_sum == 0:
            sampled_idx_hist = np.random.choice(len(self.histograms['train'][0]))
            hist_freq_per_num_atoms = self.histograms['train'][1]
        else:
            sampled_idx_hist = np.random.choice(len(self.histograms['train'][0]), p=hist_prob_per_num_atoms)
        # generate a new molecule
        current_hist = self.histograms['train'][0][sampled_idx_hist]
        temp = self.generate_graph_with_state(random_normal_states, random_global_normal_states, num_vertices,
                                              generated_all_similes, elements, step,
                                              count, current_hist, hist_freq_per_num_atoms)
        generated_all_similes.append(temp)
        return random_normal_states

    def generation(self, data):
        # bucketed: data organized by bucket
        (bucketed, bucket_sizes, bucket_at_step) = data
        # all generated similes
        generated_all_similes = []
        # counter
        count = 0
        # shuffle the lengths
        np.random.shuffle(bucket_at_step)
        for step in range(len(bucket_at_step)):
            bucket = bucket_at_step[step]  # bucket number
            # data index
            idx = random.randint(0, len(bucketed[bucket]) - 1)
            elements_original = bucketed[bucket][idx]
            elements = copy.deepcopy(elements_original)
            maximum_length = bucket_sizes[bucket]
            # if idx == 11867 or elements['smiles'] == 'CC1COC1CC#N':  # 'CC1COC1CC#N'  # 'CC1[NH+]2CC(O)C12CO'
            #     print('IDX: ', idx)
            # compensate for the length during generation
            # (this is a result that BFS may not make use of all candidate nodes during generation)
            if self.params['compensate_num'] > 0:
                maximum_length = self.compensate_node_length(elements, bucket_sizes[bucket])
            # initial state
            random_normal_states = utils.generate_std_normal(1, maximum_length,
                                                             self.params['latent_space_size'])  # [1, h]
            random_global_normal_states = np.random.normal(0, 1, [1, self.params['latent_space_size']])
            random_normal_states = self.generation_new_graphs(random_normal_states, random_global_normal_states,
                                                              maximum_length,
                                                              generated_all_similes,
                                                              elements, count)
            # SAVING
            n_gen_max = self.params['number_of_generation']
            n_gen_cur = len(generated_all_similes)
            print("Molecules generated: ", n_gen_cur, end='\r')
            if n_gen_cur >= n_gen_max:
                suff = "_" + self.params['suffix'] if self.params['suffix'] is not None else ""
                mask = "_masked" if self.params['use_mask'] else "_noMask"
                log_dir = self.params['log_dir']
                priors_file = log_dir + "/" + str(dataset) + "_decoded_generation_" + str(
                    self.params["kl_trade_off_lambda"]) \
                              + mask + suff + ".txt"

                if self.params['dataset'] == 'moses':
                    df = pd.DataFrame(columns=['SMILES'])
                    df['SMILES'] = np.array(generated_all_similes).flatten().tolist()
                    df.to_csv(priors_file, index=None)
                else:
                    generated = np.reshape(np.array(generated_all_similes).flatten().tolist(), (1000, -1))
                    f = open(priors_file, "w")
                    for line in generated:
                        for res in line:
                            f.write(res)
                            f.write(";,;")
                        f.write("\n")
                    f.close()
                print("Generation done")
                exit(0)
            count += 1

    def generate_graph_with_state(self, random_normal_states, random_global_normal_states, num_vertices,
                                  generated_all_similes, elements, step, count, current_hist, values_hist):
        # Get back node symbol predictions
        # Prepare dict
        node_symbol_batch_feed_dict = self.get_dynamic_feed_dict(elements, None, None,
                                                                 num_vertices, None, None, None, None, None,
                                                                 random_normal_states,
                                                                 current_hist, values_hist)
        # Get predicted node probabilities
        [latent_nodes, predicted_node_symbol_prob, real_values] = self.get_node_symbol(node_symbol_batch_feed_dict)
        # Node numbers for each graph
        real_length = utils.get_graph_length([elements['mask']])[0]  # [valid_node_number]
        sampled_node_symbol = np.squeeze(real_values)[:real_length]
        # Maximum valences for each node
        valences = utils.get_initial_valence(sampled_node_symbol, self.params["dataset"])  # [v]

        # generate a new molecule
        new_mol = self.search_and_generate_molecule(np.copy(valences), sampled_node_symbol, real_length,
                                                    elements, num_vertices, latent_nodes)
        if new_mol is None:
            return "None"
        else:
            # return Chem.MolToSmiles(utils.convert_radical_electrons_to_hydrogens(new_mol))
            return Chem.MolToSmiles(new_mol)

    def compensate_node_length(self, elements, bucket_size):
        maximum_length = bucket_size + self.params["compensate_num"]
        real_length = utils.get_graph_length([elements['mask']])[0] + self.params["compensate_num"]
        elements['mask'] = [1] * real_length + [0] * (maximum_length - real_length)
        elements['init'] = np.zeros((maximum_length, self.params["num_symbols"]))
        elements['adj_mat'] = np.zeros((self.num_edge_types, maximum_length, maximum_length))
        return maximum_length

    def generate_new_graphs(self, data):
        # bucketed: data organized by bucket
        (bucketed, bucket_sizes, bucket_at_step) = data
        # all generated similes
        generated_all_similes = []
        # counter
        count = 0
        # shuffle the lengths
        np.random.shuffle(bucket_at_step)
        for step in range(len(bucket_at_step)):
            bucket = bucket_at_step[step]  # bucket number
            # data index
            idx = random.randint(0, len(bucketed[bucket]) - 1)
            elements_original = bucketed[bucket][idx]
            elements = copy.deepcopy(elements_original)
            maximum_length = bucket_sizes[bucket]
            # if idx == 11867 or elements['smiles'] == 'CC1COC1CC#N':  # 'CC1COC1CC#N'  # 'CC1[NH+]2CC(O)C12CO'
            #     print('IDX: ', idx)
            # compensate for the length during generation
            # (this is a result that BFS may not make use of all candidate nodes during generation)
            if self.params['compensate_num'] > 0:
                maximum_length = self.compensate_node_length(elements, bucket_sizes[bucket])
            # initial state
            random_normal_states = utils.generate_std_normal(1, maximum_length,
                                                             self.params['latent_space_size'])  # [1, h]
            random_normal_states = self.optimization_over_prior(random_normal_states, maximum_length,
                                                                generated_all_similes,
                                                                elements, count)
            count += 1

    def reconstruction_new_graphs(self, num_vertices, generated_all_similes, elements):
        all_decoded = []
        # add the original molecule as first one in the list
        all_decoded.append(elements['smiles'])
        for n_en in range(self.params['reconstruction_en']):
            # take latent from the input encoding or from prior
            random_normal_states = utils.generate_std_normal(1, num_vertices,
                                                             self.params['latent_space_size'])  # [1, h]
            # is generative is always false here due to the sampling in the latent space
            feed_dict = self.get_dynamic_mean_feed_dict(elements, num_vertices, random_normal_states)
            # get the latent point according to the encoder distribution
            fetch_list = [self.ops['z_sampled']]
            [latent_point] = self.sess.run(fetch_list, feed_dict=feed_dict)
            for n_dn in range(self.params['reconstruction_dn']):
                # Get back node symbol predictions
                # Prepare dict
                node_symbol_batch_feed_dict = self.get_dynamic_nodes_feed_dict(elements, num_vertices, latent_point)
                # Get predicted node probabilities
                [latent_nodes, predicted_node_symbol_prob, real_values] = self.get_node_symbol(
                    node_symbol_batch_feed_dict)
                # Node numbers for each graph
                real_length = utils.get_graph_length([elements['mask']])[0]
                sampled_node_symbol = np.squeeze(real_values)[:real_length]
                # Maximum valences for each node
                valences = utils.get_initial_valence(sampled_node_symbol, self.params["dataset"])  # [v]
                # randomly pick the starting point or use zero
                new_mol = self.search_and_generate_molecule(np.copy(valences),
                                                            sampled_node_symbol, real_length,
                                                            elements, num_vertices,
                                                            latent_nodes)
                if new_mol is None:
                    all_decoded.append('None')
                else:
                    all_decoded.append(Chem.MolToSmiles(new_mol))

        generated_all_similes.append(all_decoded)

    def reconstruction(self, data):
        (bucketed, bucket_sizes, bucket_at_step) = data
        bucket_counters = defaultdict(int)
        # all generated similes
        generated_all_similes = []
        for step in range(len(bucket_at_step)):
            bucket = bucket_at_step[step]  # bucket number
            # data index
            start_idx = bucket_counters[bucket] * self.params['batch_size']
            end_idx = (bucket_counters[bucket] + 1) * self.params['batch_size']
            # batch data
            elements_batch = bucketed[bucket][start_idx:end_idx]

            if self.params["use_rec_multi_threads"]:
                thr = []
                for elements in elements_batch:
                    maximum_length = bucket_sizes[bucket]
                    # initial state
                    thr.append(utils.ThreadWithReturnValue(target=self.reconstruction_new_graphs,
                                                           args=(maximum_length, generated_all_similes, elements)))
                [t.join() for t in thr]
            else:
                for elements in elements_batch:
                    maximum_length = bucket_sizes[bucket]
                    # initial state
                    self.reconstruction_new_graphs(maximum_length, generated_all_similes, elements)

            print("Molecules reconstructed: ", len(generated_all_similes), end='\r')
            # exit(0)
            bucket_counters[bucket] += 1

        suff = "_" + self.params['suffix'] if self.params['suffix'] is not None else ""
        mask = "_masked" if self.params['use_mask'] else "_noMask"
        parent = "(" + str(self.params["reconstruction_en"]) + ":" + str(self.params["reconstruction_dn"]) + ")"
        log_dir = self.params['log_dir']
        recon_file = log_dir + "/" + str(dataset) + "_decoded_reconstruction_" + parent + "_" + str(
            self.params["kl_trade_off_lambda"]) + mask + suff + ".txt"
        f = open(recon_file, "w")
        for line in generated_all_similes:
            for res in line:
                f.write(res)
                f.write(";,;")
            f.write("\n")
        f.close()
        print('Reconstruction done')
        exit(0)

    def make_minibatch_iterator(self, data, is_training: bool):
        (bucketed, bucket_sizes, bucket_at_step) = data
        if is_training:
            np.random.shuffle(bucket_at_step)
            for _, bucketed_data in bucketed.items():
                np.random.shuffle(bucketed_data)
        bucket_counters = defaultdict(int)
        dropout_keep_prob = self.params['graph_state_dropout_keep_prob'] if is_training else 1.
        edge_dropout_keep_prob = self.params['edge_weight_dropout_keep_prob'] if is_training else 1.
        n_batches = len(bucket_at_step)
        for step in range(n_batches):
            bucket = bucket_at_step[step]
            start_idx = bucket_counters[bucket] * self.params['batch_size']
            end_idx = (bucket_counters[bucket] + 1) * self.params['batch_size']
            elements = bucketed[bucket][start_idx:end_idx]
            batch_data = self.make_batch(elements, bucket_sizes[bucket])
            num_graphs = len(batch_data['init'])

            batch_feed_dict = {
                self.placeholders['node_symbols']: batch_data['init'],
                self.placeholders['target_values']: np.transpose(batch_data['labels'], axes=[1, 0]),
                self.placeholders['target_mask']: np.transpose(batch_data['task_masks'], axes=[1, 0]),
                self.placeholders['num_graphs']: num_graphs,
                self.placeholders['num_vertices']: bucket_sizes[bucket],
                self.placeholders['adjacency_matrix']: batch_data['adj_mat'],
                self.placeholders['node_mask']: batch_data['node_mask'],
                self.placeholders['graph_state_keep_prob']: dropout_keep_prob,
                self.placeholders['edge_weight_dropout_keep_prob']: edge_dropout_keep_prob,
                self.placeholders['histograms']: self.histograms['train'][0],
                self.placeholders['n_histograms']: self.histograms['train'][1],
                self.placeholders['hist']: batch_data['hist'],
                self.placeholders['incr_hist']: batch_data['incr_hist'],
                self.placeholders['incr_diff_hist']: batch_data['incr_diff_hist'],
                self.placeholders['incr_node_mask']: batch_data['incr_node_mask'],
                self.placeholders['smiles']: '-'.join(batch_data['smiles']),
            }
            bucket_counters[bucket] += 1
            yield batch_feed_dict


if __name__ == "__main__":
    args = docopt(__doc__)
    start = time.time()
    dataset = args.get('--dataset')
    model = MolGVAE(args)
    try:
        model.train()
    except:
        typ, value, tb = sys.exc_info()
        traceback.print_exc()
    finally:
        end = time.time()
        print("Time for the overall execution: " + model.get_time_diff(end, start))
