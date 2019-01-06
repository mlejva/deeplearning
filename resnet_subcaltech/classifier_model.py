from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import tensorflow.contrib.losses as tf_losses
import tensorflow.contrib.metrics as tf_metrics

class Network:
    FEATURES_COUNT = 2048

    def __init__(self, threads=1, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                        intra_op_parallelism_threads=threads))

    def construct(self):
        with self.session.graph.as_default():
            # Construct the model
            self.features = tf.placeholder(tf.float32, [None, self.FEATURES_COUNT])
            self.labels = tf.placeholder(tf.int64, [None])

            output = tf_layers.fully_connected(self.features, num_outputs=50, activation_fn=None)
            self.predictions = tf.argmax(output, 1)

            loss = tf_losses.sparse_softmax_cross_entropy(output, self.labels)

            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=self.global_step)
            self.accuracy = tf_metrics.accuracy(self.predictions, self.labels)

            # Initialize variables
            self.session.run(tf.initialize_all_variables())

    @property
    def training_step(self):
        return self.session.run(self.global_step)

    def train(self, features, labels):
        args = {"feed_dict": {self.features: features, self.labels: labels}}
        targets = [self.training]
        results = self.session.run(targets, **args)

    def evaluate(self, dataset, epoch, features, labels):
        targets = [self.accuracy]
        results = self.session.run(targets, {self.features: features, self.labels: labels})
        print('Accuracy on %s, epoch %d: %.5f' % (dataset, epoch, results[0]))

def load_numpy_array(filename):
    arr = np.load(filename)
    return arr

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Construct the network
    network = Network(threads=args.threads)
    network.construct()

    train_features_path = 'export/train/train_features.npy'
    train_labels_path = 'export/train/train_labels.npy'
    test_features_path = 'export/test/test_features.npy'
    test_labels_path = 'export/test/test_labels.npy'

    train_features = load_numpy_array(train_features_path)
    train_labels = load_numpy_array(train_labels_path)
    test_features = load_numpy_array(test_features_path)
    test_labels = load_numpy_array(test_labels_path)

    # Load train data
    train_data_length = train_features.shape[0]
    batches_total = int(train_data_length / args.batch_size)

    for i in range(args.epochs):
        for j in range(batches_total):
            batch_start = (j - 1) * args.batch_size
            batch_end = batch_start + args.batch_size

            batch_features = train_features[batch_start:batch_end]
            batch_labels = train_labels[batch_start:batch_end]
            # Train the network
            network.train(batch_features, batch_labels)

        # Evaluate network after each epoch
        network.evaluate('test', i + 1, test_features, test_labels)
