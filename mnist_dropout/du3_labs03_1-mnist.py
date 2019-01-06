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
                # Implicitne maji obe dropout nodes hodnotu 1
                self.dropout_keep_prob_input = tf.placeholder_with_default(1.0, [], name='dropout_keep_prob_input')
                self.dropout_keep_prob_hidden = tf.placeholder_with_default(1.0, [], name='dropout_keep_prob_hidden')

            # Dropout musi byt vlastni node v grafu - placeholder, protoze pri eval musi byt probability = 1, ale pri eval je uz graf postaveny
            # placeholder_with_default ma implicitne hodnotu 1
            # Tzn. chci node grafu, ktera bude brat ruzne hodnoty ze vstupu -> placeholder

            flattened_images = tf_layers.flatten(self.images, scope="preprocessing")
            flattened_images = tf.nn.dropout(flattened_images, self.dropout_keep_prob_input)

            hidden_layer = tf_layers.fully_connected(flattened_images, num_outputs=hidden_layer_size, activation_fn=tf.nn.relu, scope="hidden_layer")
            hidden_layer = tf.nn.dropout(hidden_layer, self.dropout_keep_prob_hidden)

            output_layer = tf_layers.fully_connected(hidden_layer, num_outputs=self.LABELS, activation_fn=None, scope="output_layer")
            self.predictions = tf.argmax(output_layer, 1)

            loss = tf_losses.sparse_softmax_cross_entropy(output_layer, self.labels, scope="loss")            

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

    def train(self, images, labels, keep_prob_input, keep_prob_hidden, summaries=False, run_metadata=False):
        if (summaries or run_metadata) and not self.summary_writer:
            raise ValueError("Logdir is required for summaries or run_metadata.")

        args = {"feed_dict": {self.images: images, self.labels: labels, self.dropout_keep_prob_input: keep_prob_input, self.dropout_keep_prob_hidden: keep_prob_hidden}}
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

    def evaluate(self, dataset, images, labels, summaries=False, test_final=False, dev_final=False, keep_prob_input=None, keep_prob_hidden=None, best_probs_dict=None):
        if summaries and not self.summary_writer:
            raise ValueError("Logdir is required for summaries.")

        targets = [self.accuracy]
        if summaries:
            targets.append(self.summaries[dataset])

        results = self.session.run(targets, {self.images: images, self.labels: labels})

        if dev_final:
            accuracy = float(results[0])
            print('Accuracy on the dev dataset: %.5f' % accuracy)
            accuracy_new = float(results[0])
            accuracy_last = best_probs_dict['accuracy_dev']
            if (accuracy_new > accuracy_last):
                best_probs_dict['accuracy_dev'] = accuracy_new
                best_probs_dict['keep_prob_input'] = keep_prob_input
                best_probs_dict['keep_prob_hidden'] = keep_prob_hidden

        if test_final:
            accuracy = float(results[0])
            best_probs_dict['accuracy_test'] = accuracy

        if summaries:
            self.summary_writer.add_summary(results[-1], self.training_step)
        return results[0]

def createAndRunExperiment(args, exp_name, keep_prob_input, keep_prob_hidden, best_probs_dict, only_test=False):
    print('Experiment: %s\n' % exp_name)

    # Fix random seed
    np.random.seed(42)

    # Load the data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("mnist_data/", reshape=False)

    # Construct the network
    network = Network(threads=args.threads, logdir=args.logdir, expname=exp_name)
    network.construct(100)

    # Train
    for i in range(args.epochs):

        while mnist.train.epochs_completed == i:
            images, labels = mnist.train.next_batch(args.batch_size)
            # Dropout chci pouzivat jen pri trainu a pri eval (= inferenci) uz chci probability = 1 (tj. zadny dropout)
            network.train(images, labels, keep_prob_input, keep_prob_hidden, network.training_step % 100 == 0, network.training_step == 0) # Dropout jen zde

        if only_test:
            if (i == args.epochs - 1):
                network.evaluate("test", mnist.test.images, mnist.test.labels, True, test_final=True, best_probs_dict=best_probs_dict)
            else:
                network.evaluate("test", mnist.test.images, mnist.test.labels, True)
        else:
            if (i == args.epochs - 1):
                print('\nkeep_prob_input: %.3f' % keep_prob_input)
                print('keep_prob_hidden: %.3f' % keep_prob_hidden)
                network.evaluate("dev", mnist.validation.images, mnist.validation.labels, True, False, True, keep_prob_input, keep_prob_hidden, best_probs_dict)
                print('-----------------------')
            else:
                network.evaluate("dev", mnist.validation.images, mnist.validation.labels, True)

if __name__ == "__main__":

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
    parser.add_argument("--logdir", default="logs", type=str, help="Logdir name.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    best_probs_dict = {'keep_prob_input': 0.0, 'keep_prob_hidden': 0.0, 'accuracy_dev': 0.0, 'accuracy_test': 0.0}

    keep_probs_input = [0.8, 0.9, 1.0]
    keep_probs_hidden = [0.8, 0.9, 1.0]

    for prob_input in keep_probs_input:
        for prob_hidden in keep_probs_hidden:
            exp_name = '1-mnist-keep_prob_input_%.3f-keep_prob_hidden_%.3f' % (prob_input, prob_hidden)
            createAndRunExperiment(args, exp_name, prob_input, prob_hidden, best_probs_dict)


    # Run on the test dataset with the best hyperparameters
    print('Running network on the test dataset with the best hyperparameters...\n')

    best_prob_input = best_probs_dict['keep_prob_input']
    best_prob_hidden = best_probs_dict['keep_prob_hidden']

    exp_name = '1-mnist-dropout_best_hyperparams'
    createAndRunExperiment(args, exp_name, best_prob_input, best_prob_hidden, best_probs_dict, True)

    print('\n-----Best combination for dropout-----')
    print('Keep probability for input layer: %.3f' % best_prob_input)
    print('Keep probability for hidden layer: %.3f' % best_prob_hidden)
    print('Accuracy on the dev dataset: %.5f' % best_probs_dict['accuracy_dev'])
    print('Accuracy on the test dataset: %.5f' % best_probs_dict['accuracy_test'])
    print('======================================')
