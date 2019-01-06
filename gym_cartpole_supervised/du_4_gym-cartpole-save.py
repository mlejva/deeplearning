from __future__ import division
from __future__ import print_function

import subprocess
import datetime
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import tensorflow.contrib.losses as tf_losses
import tensorflow.contrib.metrics as tf_metrics

class Network:
    OBSERVATIONS = 4

    def __init__(self, threads=1, logdir=None, expname=None, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

        if logdir:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            self.summary_writer = tf.train.SummaryWriter(("{}/{}-{}" if expname else "{}/{}").format(logdir, timestamp, expname), flush_secs=10)
        else:
            self.summary_writer = None

    def construct(self):
        with self.session.graph.as_default():
            with tf.name_scope("inputs"):
                self.observations = tf.placeholder(tf.float32, [None, self.OBSERVATIONS])
                self.labels = tf.placeholder(tf.int64, [None, 1])

            input_layer = tf_layers.fully_connected(self.observations, num_outputs=4, activation_fn=tf.nn.tanh)
            output_layer = tf_layers.fully_connected(input_layer, num_outputs=2, activation_fn=tf.nn.tanh)

            # Global step
            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")

            loss = tf_losses.sparse_softmax_cross_entropy(output_layer, self.labels)
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=self.global_step)


            self.action = tf.argmax(output_layer, 1)
            self.accuracy = tf_metrics.accuracy(self.action, self.labels)

            # Construct the saver
            tf.add_to_collection("end_points/observations", self.observations)
            tf.add_to_collection("end_points/action", self.action)
            self.saver = tf.train.Saver(max_to_keep=None)

            # Initialize the variables
            self.session.run(tf.initialize_all_variables())

        # Finalize graph and log it if requested
        self.session.graph.finalize()
        if self.summary_writer:
            self.summary_writer.add_graph(self.session.graph)

    # Save the graph
    def save(self, path):
        self.saver.save(self.session, path)


    def train(self, observations_arg, labels_arg, i):
        args = {'feed_dict': {self.observations: observations_arg,
                              self.labels: labels_arg}}

        result = self.session.run([self.accuracy, self.training, self.action], **args)
        print(result[0])


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="", type=str, help="Logdir name.")
    parser.add_argument("--exp", default="1-gym-save", type=str, help="Experiment name.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Construct the network
    network = Network(threads=args.threads, logdir=args.logdir, expname=args.exp)
    network.construct()

    # Train the network
    training_examples_count = 100
    epoch_num = 2000
    batch_size = 10
    batches_total = int(training_examples_count / batch_size)

    for i in range(1, epoch_num):
        # Shuffle data at the start of each epoch
        data = np.loadtxt('input.txt')
        data = np.random.permutation(data)
        for j in range(1, batches_total):
            batch_start = (j - 1) * 5
            batch_end = batch_start + 5
            data_batch = data[batch_start:batch_end]
            observations = data_batch[0:5,:4]
            labels = data_batch[0:5,4]
            labels = labels.reshape((labels.shape[0], 1))
            print(observations.shape)
            print(labels.shape)
            network.train(observations, labels, i)

    # Save the network
    network.save("1-gym")
