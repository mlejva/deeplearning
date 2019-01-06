from __future__ import division
from __future__ import print_function
from enum import Enum

import datetime
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import tensorflow.contrib.losses as tf_losses
import tensorflow.contrib.metrics as tf_metrics

class Optimizer(Enum):
    SGD = 'SGD'
    SGD_decay = 'SGD_decay'
    SGD_momentum = 'SGD_momentum'
    Adam = 'Adam'

class Network:
    WIDTH = 28
    HEIGHT = 28
    LABELS = 10

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

    def construct(self, hidden_layer_size, optimizer, learning_rate, steps_total, momentum=None):
        with self.session.graph.as_default():
            with tf.name_scope("inputs"):
                self.images = tf.placeholder(tf.float32, [None, self.WIDTH, self.HEIGHT, 1], name="images")
                self.labels = tf.placeholder(tf.int64, [None], name="labels")

            flattened_images = tf_layers.flatten(self.images, scope="preprocessing")
            hidden_layer = tf_layers.fully_connected(flattened_images, num_outputs=hidden_layer_size, activation_fn=tf.nn.relu, scope="hidden_layer")
            output_layer = tf_layers.fully_connected(hidden_layer, num_outputs=self.LABELS, activation_fn=None, scope="output_layer")
            self.predictions = tf.argmax(output_layer, 1)

            loss = tf_losses.sparse_softmax_cross_entropy(output_layer, self.labels, scope="loss")
            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")

            if optimizer is Optimizer.SGD:
                self.training = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=self.global_step)

            elif optimizer is Optimizer.SGD_decay:
                starting_learning_rate = learning_rate[0]
                final_learning_rate = learning_rate[1]

                decay_rate = final_learning_rate / starting_learning_rate
                learning_rate = tf.train.exponential_decay(starting_learning_rate, self.global_step,
                                                           steps_total, final_learning_rate)
                self.training = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=self.global_step)

            elif optimizer is Optimizer.SGD_momentum:
                self.training = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss, global_step=self.global_step)

            elif optimizer is Optimizer.Adam:
                self.training = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=self.global_step)

            self.accuracy = tf_metrics.accuracy(self.predictions, self.labels)

            # Summaries
            self.summaries = {"training": tf.merge_summary([tf.scalar_summary("train/loss", loss),
                                                            tf.scalar_summary("train/accuracy", self.accuracy)])}
            for dataset in ["dev", "test"]:
                self.summaries[dataset] = tf.scalar_summary(dataset+"/accuracy", self.accuracy)

            # Initialize variables
            self.session.run(tf.initialize_all_variables())

        # Finalize graph and log it if requested
        self.session.graph.finalize()
        if self.summary_writer:
            self.summary_writer.add_graph(self.session.graph)

    @property
    def training_step(self):
        return self.session.run(self.global_step)

    def train(self, images, labels, summaries=False, run_metadata=False):
        if (summaries or run_metadata) and not self.summary_writer:
            raise ValueError("Logdir is required for summaries or run_metadata.")

        args = {"feed_dict": {self.images: images, self.labels: labels}}
        targets = [self.training]
        if summaries:
            targets.append(self.summaries["training"])
        if run_metadata:
            args["options"] = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            args["run_metadata"] = tf.RunMetadata()

        results = self.session.run(targets, **args)
        if summaries:
            self.summary_writer.add_summary(results[-1], self.training_step - 1)
        if run_metadata:
            self.summary_writer.add_run_metadata(args["run_metadata"], "step{:05}".format(self.training_step - 1))

    def evaluate(self, dataset, images, labels, summaries=False, report_dev=False):
        if summaries and not self.summary_writer:
            raise ValueError("Logdir is required for summaries.")

        targets = [self.accuracy]
        if summaries:
            targets.append(self.summaries[dataset])

        results = self.session.run(targets, {self.images: images, self.labels: labels})

        if report_dev:
            accuracy_dev = float(results[0])
            print('\nDataset: %s\nAccuracy: %.5f' % (dataset, accuracy_dev))

        if summaries:
            self.summary_writer.add_summary(results[-1], self.training_step)
        return results[0]

def createAndRunExperiment(args, experiment_name, batch_size, optimizer, learning_rate, momentum=None):
    print('Experiment name: %s\n' % experiment_name)

    # Load the data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("mnist_data/", reshape=False)

    # Construct the network
    steps_total = (mnist.train.num_examples / batch_size) * args.epochs

    network = Network(threads=args.threads, logdir=args.logdir, expname=experiment_name)
    network.construct(100, optimizer=optimizer, learning_rate=learning_rate, steps_total=steps_total, momentum=momentum)

    # Train
    for i in range(args.epochs):

        while mnist.train.epochs_completed == i:
            images, labels = mnist.train.next_batch(batch_size)
            network.train(images, labels, network.training_step % 100 == 0, network.training_step == 0)

        if (i == args.epochs - 1):
            network.evaluate("dev", mnist.validation.images, mnist.validation.labels, True, True)
        else:
            network.evaluate("dev", mnist.validation.images, mnist.validation.labels, True)

        network.evaluate("test", mnist.test.images, mnist.test.labels, True)
    print('============================================')

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
    parser.add_argument("--logdir", default="logs", type=str, help="Logdir name.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    batch_sizes = [10, 50]
    optimizers = [Optimizer.SGD, Optimizer.SGD_decay, Optimizer.SGD_momentum, Optimizer.Adam]

    for optimizer in optimizers:
        for batch_size in batch_sizes:

            if optimizer is Optimizer.SGD:
                learning_rates = [0.01, 0.001, 0.0001]
                for learning_rate in learning_rates:
                    exp_name = 'optimizer_%s-batch_%d-learning_rate_%.5f' % (optimizer, batch_size, learning_rate)
                    createAndRunExperiment(args, exp_name, batch_size, optimizer, learning_rate)

            elif optimizer is Optimizer.SGD_decay:
                learning_rate_pairs = [(0.01, 0.001), (0.01, 0.0001), (0.001, 0.0001)]
                for learning_rate_pair in learning_rate_pairs:
                    starting = learning_rate_pair[0]
                    final = learning_rate_pair[1]
                    exp_name = 'optimizer_%s-batch_%d-learning_rate_start_%.5f_final_%.5f' % (optimizer, batch_size, starting, final)
                    createAndRunExperiment(args, exp_name, batch_size, optimizer, learning_rate_pair)

            elif optimizer is Optimizer.SGD_momentum:
                momentum = 0.9
                learning_rates = [0.01, 0.001, 0.0001]
                for learning_rate in learning_rates:
                    exp_name = 'optimizer_%s-batch_%d-learning_rate_%.5f-momentum_%.2f' % (optimizer, batch_size, learning_rate, momentum)
                    createAndRunExperiment(args, exp_name, batch_size, optimizer, learning_rate, momentum)

            elif optimizer is Optimizer.Adam:
                learning_rates = [0.002, 0.001, 0.0005]
                for learning_rate in learning_rates:
                    exp_name = 'optimizer_%s-batch_%d-learning_rate_%.5f' % (optimizer, batch_size, learning_rate)
                    createAndRunExperiment(args, exp_name, batch_size, optimizer, learning_rate)
