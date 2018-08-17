#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import networkx as nx
import os
import argparse
import contextlib
import datetime
import sys


class BaseNet:

    @classmethod
    def parser(_cls):
        "Create an ArgumentParser with common parameters."
        # Parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--learning_rate", default=0.001, type=float, help="Base learning rate.")
        parser.add_argument("--threads", default=2, type=int, help="Maximum number of threads to use.")
        parser.add_argument("--clip_gradient", default=10.0, type=float, help="Gradient clipping.")
        parser.add_argument("--name", default="", type=str, help="Any name comment.")
        parser.add_argument("--beta_2", default=0.99, type=float, help="Adam beta_2 parameter.")
        parser.add_argument("--logdir", default="logs", type=str, help="Logging subdirectory (default: 'logs/').")
        parser.add_argument("--summary_every", default=32, type=int, help="Summary record period.")
        parser.add_argument("--keep_saves", default=2, type=int, help="Net saves to keep.")
        parser.add_argument("--epochs", default=20, type=int, help="Training epochs.")
        parser.add_argument("--seed", default=None, type=int, help="Random seed.")
        return parser

    def __init__(self, args, basename=None):
        if basename is None:
            basename = self.__class__.__name__
        self.bname = "{}-{}".format(
            basename, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        if args.name:
            self.bname += "-" + args.name
        self.logdir = os.path.join(args.logdir, self.bname)
        self.graph = tf.Graph()
        self.graph.seed = args.seed
        config = tf.ConfigProto(
            inter_op_parallelism_threads=args.threads,                                                                    intra_op_parallelism_threads=args.threads)
        self.session = tf.Session(graph=self.graph, config=config)
        self.args = args

    def construct_pre(self):
        """
        Creates directory and opens the logfile.
        """
        os.makedirs(self.logdir, exist_ok=True)
        self.log = open(os.path.join(self.logdir, "log.txt"), "wt")
        hello = "Running {} in {}\nCmdline: {}\nArgs: {}\n".format(
            self.bname, self.logdir, sys.argv, self.args)
        self.log.write(hello + "\n")
        print(hello)

        with self.session.graph.as_default():
            # Training variables and global step
            self.t_rate = tf.placeholder_with_default(self.args.learning_rate, [], name="t_rate")
            self.global_step = tf.train.create_global_step()
            # Summaries
            self.summary_writer = tf.contrib.summary.create_file_writer(self.logdir, flush_millis=1000)
            self.summaries = []

    def construct_post(self):
        """
        Creates optimizer (Adam with grad clipping) and `self.training`, basic summaries and saver.
        Initializes all the variables and summaries. Needs `self.loss` to be set.
        """
        with self.session.graph.as_default():

            # Optimizer and gradient
            self.optimizer = tf.train.AdamOptimizer(self.t_rate, beta2=self.args.beta_2)
            gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
            gradient_norm = tf.global_norm(gradients)
            if self.args.clip_gradient:
                gradients, _ = tf.clip_by_global_norm(gradients, self.args.clip_gradient, gradient_norm)
            self.training = self.optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

            # Summaries
            variables_l2 = tf.add_n([tf.nn.l2_loss(variable) for variable in tf.trainable_variables()])
            with self.summary_context():
                self.summaries.extend([
                    tf.contrib.summary.scalar("train/loss", self.loss),
                    tf.contrib.summary.scalar("train/vars_l2_norm", variables_l2),
                    tf.contrib.summary.scalar("train/grad_l2_norm", gradient_norm),
                ])

            self.saver = tf.train.Saver(max_to_keep=self.args.keep_saves)

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with self.summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def compute(self, feed, ret=[]):
        "Simple computation of `ret` from `feed` - to be overriden."
        return self.session.run(ret, feed)

    def train(self, feed, ret=[], rate=None):
        "Simple training on `feed`, returning values of `ret` - to be overriden."
        if rate is not None:
            feed[self.t_rate] = rate
        return self.session.run([self.training, self.summaries] + ret, feed)[2:]

    @contextlib.contextmanager
    def summary_context(self):
        """
        Context for adding summaries to self.summaries.
        """
        with self.summary_writer.as_default():
            with tf.contrib.summary.record_summaries_every_n_global_steps(self.args.summary_every, self.global_step):
                yield
