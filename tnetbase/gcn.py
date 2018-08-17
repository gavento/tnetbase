#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import networkx as nx
import random

from . import basenet, math


class GCNet(basenet.BaseNet):

    def edge_gadget(self, g_layer, e_layer, v_in, v_out, dim=None):
        cn = tf.concat([
            tf.tile(g_layer, [self.g_m, 1]),
            v_in,
            v_out,
        ], axis=1)
        return tf.layers.dense(cn, dim, activation=tf.nn.relu)

    def vertex_gadget(self, g_layer, v_layer, v_sum, v_mean, v_max, dim=None):
        cn = tf.concat([
            tf.tile(g_layer, [self.g_n, 1]),
            v_layer,
            v_sum,
            v_mean,
            v_max,
        ], axis=1)
        return tf.layers.dense(cn, dim, activation=tf.nn.relu)

    def graph_gadget(self, g_layer, v_layer, dim=None):
        cn = tf.concat([
            g_layer,
            tf.reduce_sum(v_layer, axis=0, keepdims=True),
            tf.reduce_max(v_layer, axis=0, keepdims=True),
            tf.reduce_mean(v_layer, axis=0, keepdims=True),
        ], axis=1)
        return tf.layers.dense(cn, dim, activation=tf.nn.relu)

    def construct(self, v_dims=[2, 4, 8], e_dims=[2, 4, 8], g_dims=[4, 8, 16]):
        with self.session.graph.as_default():
            assert(len(v_dims) == len(e_dims))
            assert(len(v_dims) == len(g_dims))

            # Input graph order and size
            self.g_n = tf.placeholder(tf.int32, [], name="g_n")
            self.g_n_f = tf.cast(self.g_n, tf.float32)
            self.g_m = tf.placeholder(tf.int32, [], name="g_m")
            self.g_m_f = tf.cast(self.g_m, tf.float32)

            # Input graph edge sources and destinations
            self.edge_srcs = tf.placeholder(tf.int32, [None], name="edge_srcs")
            self.edge_dsts = tf.placeholder(tf.int32, [None], name="edge_dsts")

            # Initial values
            self.v_layer = tf.ones([self.g_n, 1])
            self.g_layer = tf.ones([1, 0])
            self.e_layer = tf.ones([self.g_m, 0])

            for i in range(len(v_dims)):

                # For every edge, the input and output vertex
                e_in_layer = tf.gather(self.v_layer, self.edge_srcs, axis=0)
                e_out_layer = tf.gather(self.v_layer, self.edge_dsts, axis=0)
                e_layer_new = self.edge_gadget(self.g_layer, self.e_layer, e_in_layer, e_out_layer, dim=e_dims[i])

                v_sum = tf.unsorted_segment_sum(e_layer_new, self.edge_dsts, self.g_n)
                v_max = tf.maximum(tf.unsorted_segment_max(e_layer_new, self.edge_dsts, self.g_n), 0.0)
                v_mean = tf.unsorted_segment_mean(e_layer_new, self.edge_dsts, self.g_n)
                v_layer_new = self.vertex_gadget(self.g_layer, self.v_layer, v_sum, v_mean, v_max, dim=v_dims[i])

                g_layer_new = self.graph_gadget(self.g_layer, self.v_layer, dim=g_dims[i])

                self.g_layer = g_layer_new
                self.v_layer = v_layer_new
                self.e_layer = e_layer_new

    def make_feed(self, G):
        """
        Return a basic feed for the network from the given graph.
        """
        if isinstance(G, nx.Graph):
            edges = list(G.edges())
            edges = edges + [(v, u) for (u, v) in edges]
        elif isinstance(G, nx.DiGraph):
            edges = list(G.edges())
        else:
            raise TypeError("Graph or Digraph expected, got {}".format(type(G)))
        idx = math.index(G.nodes())
        edges_ord = [(idx[u], idx[v]) for (u, v) in edges]
        return {
            self.g_n: G.order(),
            self.g_m: len(edges),
            self.edge_srcs: [u for (u, v) in edges_ord],
            self.edge_dsts: [v for (u, v) in edges_ord],
        }
