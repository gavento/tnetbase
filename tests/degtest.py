#!/usr/bin/env python3

import tensorflow as tf
import networkx as nx
import random

from tnetbase import GCNet


class GCDegNet(GCNet):

    def construct(self, **kwargs):
        super().construct(**kwargs)
        with self.session.graph.as_default():
            self.degs = tf.placeholder(tf.float32, [None])
            self.max_deg = tf.placeholder(tf.float32, [])
            self.min_deg = tf.placeholder(tf.float32, [])

            v_deg = tf.squeeze(tf.layers.dense(self.v_layer, 1), axis=1)
            v_loss = tf.losses.mean_squared_error(self.degs, v_deg)
            min_deg = tf.squeeze(tf.layers.dense(self.g_layer, 1), axis=1)
            min_loss = tf.losses.mean_squared_error([self.min_deg], min_deg)
            max_deg = tf.squeeze(tf.layers.dense(self.g_layer, 1), axis=1)
            max_loss = tf.losses.mean_squared_error([self.max_deg], max_deg)
            self.loss = v_loss + min_loss + max_loss

            with self.summary_context():
                self.summaries.extend([
                    tf.contrib.summary.scalar("train/loss_degs", v_loss),
                    tf.contrib.summary.scalar("train/loss_min_deg", min_loss),
                    tf.contrib.summary.scalar("train/loss_max_deg", max_loss),
                ])


if __name__ == "__main__":

    parser = GCDegNet.parser()
    parser.add_argument("--hidden_layer", default=4, type=int, help="Size of hidden layer.")
    args = parser.parse_args()

    # Construct the network
    net = GCDegNet(args)
    net.construct_pre()
    net.construct(v_dims=[2, 4, 4, 4], e_dims=[2, 4, 4, 4], g_dims=[2, 4, 4, 4])
    net.construct_post()

    for ep in range(args.epochs):
        G = nx.generators.random_graphs.gnp_random_graph(random.randint(5, 30), 0.2)
        f = net.make_feed(G)
        f[net.degs] = [G.degree()[i] for i in G.nodes()]
        f[net.max_deg] = max(G.degree().values())
        f[net.min_deg] = min(G.degree().values())
        rate = args.learning_rate
        if ep >= args.epochs / 3:
            rate *= 0.1
        if ep >= args.epochs * 2 / 3:
            rate *= 0.1
        loss = net.train(f, ret=[net.loss], rate=rate)
        print(ep, loss)
