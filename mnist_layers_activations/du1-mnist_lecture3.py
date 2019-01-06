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
        '''
        if logdir:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            self.summary_writer = tf.train.SummaryWriter(("{}/{}-{}" if expname else "{}/{}").format(logdir, timestamp, expname), flush_secs=10)
        else:
            self.summary_writer = None
        '''
        self.summary_writer = None
    def construct(self, hidden_layer_size, activation, layers_count):
        with self.session.graph.as_default():
            with tf.name_scope("inputs"):
                self.images = tf.placeholder(tf.float32, [None, self.WIDTH, self.HEIGHT, 1], name="images")
                print("==========self.images============")
                print(self.images.get_shape())
                print("======================")
                self.labels = tf.placeholder(tf.int64, [None], name="labels")
                print("==========self.labels============")
                print(self.labels.get_shape())
                print("======================")

            flattened_images = tf_layers.flatten(self.images, scope="preprocessing")
            print("==========flattened_images============")
            print(flattened_images.get_shape())
            print("======================")

            activation_fn_name = str(activation).split()[1]


            # Should be possible this way, if not use commented piece of code lower and comment this
            hidden_layer = tf_layers.fully_connected(flattened_images, num_outputs=hidden_layer_size, activation_fn=activation, scope="hidden_layer-1-%d-%s" % (layers_count, activation_fn_name))
            print("==========hidden_layer============")
            print(hidden_layer.get_shape())
            print("======================")
            '''
            for i in range(layers_count):
                 hidden_layer_ = tf_layers.fully_connected(hidden_layer, num_outputs=hidden_layer_size, activation_fn=activation, scope="hidden_layer-%d-%d-%s" % ((i + 1), layers_count, activation_fn_name))
            output_layer = tf_layers.fully_connected(hidden_layer, num_outputs=self.LABELS, activation_fn=None, scope="output-%d-%s" % (layers_count, activation_fn_name))
            '''

            output_layer = None
            if layers_count == 1:
                output_layer = tf_layers.fully_connected(hidden_layer, num_outputs=self.LABELS, activation_fn=None, scope="output-%d-%s" % (layers_count, activation_fn_name))
            elif layers_count == 2:
                hidden_layer_2 = tf_layers.fully_connected(hidden_layer, num_outputs=hidden_layer_size, activation_fn=activation, scope="hidden_layer2-%d-%s" % (layers_count, activation_fn_name))
                output_layer = tf_layers.fully_connected(hidden_layer_2, num_outputs=self.LABELS, activation_fn=None, scope="output2-%d-%s" % (layers_count, activation_fn_name))
            elif layers_count == 3:
                hidden_layer_2 = tf_layers.fully_connected(hidden_layer, num_outputs=hidden_layer_size, activation_fn=activation, scope="hidden_layer2-%d-%s" % (layers_count, activation_fn_name))
                hidden_layer_3 = tf_layers.fully_connected(hidden_layer_2, num_outputs=hidden_layer_size, activation_fn=activation, scope="hidden_layer3-%d-%s" % (layers_count, activation_fn_name))
                output_layer = tf_layers.fully_connected(hidden_layer_3, num_outputs=self.LABELS, activation_fn=None, scope="output3-%d-%s" % (layers_count, activation_fn_name))

            self.predictions = tf.argmax(output_layer, 1)
            print("==========self.predictions============")
            print(self.predictions.get_shape())
            print("======================")

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

    def evaluate(self, dataset, images, labels, summaries=False, test_final=False, dev_final=False, act_fn=None, layers_count=None, best_hyperparam_dict=None):
        if summaries and not self.summary_writer:
            raise ValueError("Logdir is required for summaries.")

        targets = [self.accuracy]
        if summaries:
            targets.append(self.summaries[dataset])

        results = self.session.run(targets, {self.images: images, self.labels: labels})
        if dev_final: # A final step for the development dataset
            accuracy_new = float(results[0])
            accuracy_last = best_hyperparam_dict["accuracy_dev"]
            if (accuracy_new > accuracy_last):
                best_hyperparam_dict["accuracy_dev"] = accuracy_new
                best_hyperparam_dict["activation_fn"] = act_fn
                best_hyperparam_dict["hidden_layers"] = layers_count
            print("Dataset: %s\nAccuracy: %.5f" % (dataset, accuracy_new))
        if test_final: # A final step for the test dataset
            accuracy = float(results[0])
            best_hyperparam_dict["accuracy_test"] = accuracy

        if summaries:
            self.summary_writer.add_summary(results[-1], self.training_step)
        return results[0]


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    best_hyperparam_dict = { "activation_fn": None, "hidden_layers": 0, "accuracy_dev": 0.0, "accuracy_test": 0.0 }
    activations = [tf.tanh, tf.nn.relu]

    import argparse

    for layers_count in range(1, 4):
        for activation in activations:
            activation_fn_name = str(activation).split()[1]
            experiment_name = "mnist-hidden-%d-function-%s" % (layers_count, activation_fn_name)
            print("Experiment: %s\n" % experiment_name)

            # Parse arguments
            parser = argparse.ArgumentParser()
            parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
            parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
            parser.add_argument("--logdir", default="logs", type=str, help="Logdir name.")
            parser.add_argument('--exp', default=experiment_name, type=str, help='Experiment name.')
            parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
            args = parser.parse_args()

            # Construct the network
            network = Network(threads=args.threads, logdir=args.logdir, expname=args.exp)

            # Load the data
            from tensorflow.examples.tutorials.mnist import input_data
            mnist = input_data.read_data_sets("mnist_data/", reshape=False)

            network.construct(100, activation=activation, layers_count=layers_count)

            # Train
            for i in range(args.epochs):
                while mnist.train.epochs_completed == i:
                    images, labels = mnist.train.next_batch(args.batch_size)
                    print(images.shape)
                    print(labels.shape)
                    print(labels)
                    network.train(images, labels, network.training_step % 100 == 0, network.training_step == 0)

                network.evaluate("test", mnist.test.images, mnist.test.labels, True)
                if (i == args.epochs - 1): # Last step
                    print("\nHidden layers: %d" % layers_count)
                    print("Activation function: %s" % activation_fn_name)
                    network.evaluate("dev", mnist.validation.images, mnist.validation.labels, True, False, True, activation, layers_count, best_hyperparam_dict)
                    print("---------------------")
                else:
                    network.evaluate("dev", mnist.validation.images, mnist.validation.labels, True)

    ### Run network with the best combination of hyperparameters ###
    # Parse arguments
    print("Running network on the test dataset with the best combination of hyperparameters...\n")
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
    parser.add_argument("--logdir", default="logs", type=str, help="Logdir name.")
    parser.add_argument('--exp', default="mnist_best_hyperparam", type=str, help='Experiment name.')
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()
    # Construct the network
    network = Network(threads=args.threads, logdir=args.logdir, expname=args.exp)
    # Load the data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("mnist_data/", reshape=False)
    network.construct(100, activation=best_hyperparam_dict["activation_fn"], layers_count=best_hyperparam_dict["hidden_layers"])
    for i in range(args.epochs):
        while mnist.train.epochs_completed == i:
            images, labels = mnist.train.next_batch(args.batch_size)
            network.train(images, labels, network.training_step % 100 == 0, network.training_step == 0)

        network.evaluate("dev", mnist.validation.images, mnist.validation.labels, True)
        if (i == args.epochs - 1): # Last step
            # Save an accuracy on test dataset to the best_hyperparam_dict
            network.evaluate("test", mnist.test.images, mnist.test.labels, True, True, False, best_hyperparam_dict["activation_fn"], best_hyperparam_dict["hidden_layers"], best_hyperparam_dict)
        else:
            network.evaluate("test", mnist.test.images, mnist.test.labels, True)

    print("=============")
    print("---The best combination of hyperparameters based on the development set evaluation---")
    print("Activation function: %s" % str(best_hyperparam_dict["activation_fn"]).split()[1])
    print("Hidden layers: %d" % best_hyperparam_dict["hidden_layers"])
    print("Accuracy on the development dataset: %.5f" % best_hyperparam_dict["accuracy_dev"])
    print("Accuracy on the test dataset: %.5f" % best_hyperparam_dict["accuracy_test"])
    print("-----------------------------------------")
