from __future__ import division
from __future__ import print_function

import datetime
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import tensorflow.contrib.losses as tf_losses
import tensorflow.contrib.metrics as tf_metrics

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

    def construct(self, hidden_layer_size):
        with self.session.graph.as_default():
            with tf.name_scope("inputs"):
                self.images = tf.placeholder(tf.float32, [None, self.WIDTH, self.HEIGHT, 1], name="images")
                self.labels = tf.placeholder(tf.int64, [None], name="labels")

            with tf.name_scope('params'):
                self.training_mode = tf.placeholder(tf.bool, [], name='training_mode')
                #self.keep_prob = tf.placeholder_with_default(1.0, [], name='keep_prob')

            # Nejlip dopadne 17 epoch, 40, batch norm => 0.99420

            # flattened_images = tf_layers.flatten(self.images, scope="preprocessing")

            # Na zacatku 28x28x1

            # conv 3x3x5 (3x3x20) SAME
            conv_layer1 = tf_layers.convolution2d(self.images, 20, [3, 3], padding='SAME', normalizer_fn=tf_layers.batch_norm)
            # relu (default)
            # batch_norm
            # batch_norm1 = tf_layers.batch_norm(conv_layer1)

            # conv 3x3x5 (3x3x20) SAME
            conv_layer2 = tf_layers.convolution2d(conv_layer1, 20, [3, 3], padding='SAME', normalizer_fn=tf_layers.batch_norm)
            # relu (default)
            # batch_norm
            # batch_norm2 = tf_layers.batch_norm(conv_layer2)

            # max pooling 3x3, stride 2 VALUE
            max_pooled1 = tf_layers.max_pool2d(conv_layer2, 3, 2, 'VALID')

            # Budu mit 13x13x5 (13x13x10)

            # conv 3x3x10 (3x3x40)
            conv_layer3 = tf_layers.convolution2d(max_pooled1, 40, [3, 3], padding='SAME', normalizer_fn=tf_layers.batch_norm)
            # relu
            # batch_norm
            # batch_norm3 = tf_layers.batch_norm(conv_layer3)

            # conv 3x3x10 (3x3x40)
            conv_layer4 = tf_layers.convolution2d(conv_layer3, 40, [3, 3], padding='SAME', normalizer_fn=tf_layers.batch_norm)
            # relu (default)
            # batch_norm
            # batch_norm4 = tf_layers.batch_norm(conv_layer4)

            # max pooling 3x3, stride 2
            max_pooled2 = tf_layers.max_pool2d(conv_layer4, 3, 2, 'VALID')

            # Budu mit 6x6x10 (6x6x40)

            # flatten
            flattened = tf_layers.flatten(max_pooled2)
            # FC (FullyConnected) bez aktivacni fce., output_num = 10 (40)
            fc = tf_layers.fully_connected(flattened, num_outputs=40, activation_fn=None)

            self.predictions = tf.argmax(fc, 1)
            loss = tf_losses.sparse_softmax_cross_entropy(fc, self.labels, scope="loss")

            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=self.global_step)
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

        args = {"feed_dict": {self.images: images, self.labels: labels, self.training_mode: True}}
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

    def evaluate(self, dataset, images, labels, summaries=False, test_final=False):
        if summaries and not self.summary_writer:
            raise ValueError("Logdir is required for summaries.")

        targets = [self.accuracy]
        if summaries:
            targets.append(self.summaries[dataset])

        results = self.session.run(targets, {self.images: images, self.labels: labels, self.training_mode: False})
        if test_final:
            accuracy = float(results[0])
            print('Final accuracy: %.5f' % accuracy)

        if (dataset == 'test'):
            print('Accuracy: %.5f' % float(results[0]))

        if summaries:
            self.summary_writer.add_summary(results[-1], self.training_step)
        return results[0]


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=17, type=int, help="Number of epochs.")
    parser.add_argument("--logdir", default="logs", type=str, help="Logdir name.")
    parser.add_argument("--exp", default="mnist_conv_40_with_batch_17epochs", type=str, help="Experiment name.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Load the data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("mnist_data/", reshape=False)

    # Construct the network
    network = Network(threads=args.threads, logdir=args.logdir, expname=args.exp)
    network.construct(100)

    # Train
    for i in range(args.epochs):
        while mnist.train.epochs_completed == i:
            images, labels = mnist.train.next_batch(args.batch_size)
            network.train(images, labels, network.training_step % 100 == 0, network.training_step == 0)

        print(i)
        network.evaluate("dev", mnist.validation.images, mnist.validation.labels, True)
        network.evaluate("test", mnist.test.images, mnist.test.labels, True)
        '''
        if (i == args.epochs - 1):
            network.evaluate("test", mnist.test.images, mnist.test.labels, True, True)
        else:
            network.evaluate("test", mnist.test.images, mnist.test.labels, True)
        '''
