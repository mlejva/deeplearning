#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import datetime
import numpy as np
import tensorflow as tf
import tensorflow.contrib.losses as tf_losses
import tensorflow.contrib.layers as tf_layers
import tensorflow.contrib.metrics as tf_metrics

class Dataset:
    def __init__(self, filename, alphabet = None):
        # Load the sentences
        sentences = []
        with open(filename, "r") as file:
            for line in file:
                sentences.append(line.rstrip("\r\n"))

        # Compute sentence lengths
        self._sentence_lens = np.zeros([len(sentences)], np.int32)
        for i in range(len(sentences)):
            self._sentence_lens[i] = len(sentences[i])
        max_sentence_len = np.max(self._sentence_lens)

        # Create alphabet_map
        alphabet_map = {'<pad>': 0, '<unk>': 1}
        if alphabet is not None:
            for index, letter in enumerate(alphabet):
                alphabet_map[letter] = index

        # Remap input characters using the alphabet_map
        self._sentences = np.zeros([len(sentences), max_sentence_len], np.int32)
        self._labels = np.zeros([len(sentences), max_sentence_len], np.int32)
        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                char = sentences[i][j].lower()
                if char not in alphabet_map:
                    if alphabet is None:
                        alphabet_map[char] = len(alphabet_map)
                    else:
                        char = '<unk>'
                self._sentences[i, j] = alphabet_map[char]
                self._labels[i, j] = 0 if sentences[i][j].lower() == sentences[i][j] else 1

        # Compute alphabet
        self._alphabet = [""] * len(alphabet_map)
        for key, value in alphabet_map.items():
            self._alphabet[value] = key

        self._permutation = np.random.permutation(len(self._sentences))

    @property
    def alphabet(self):
        return self._alphabet

    @property
    def sentences(self):
        return self._sentences

    @property
    def sentence_lens(self):
        return self._sentence_lens

    @property
    def labels(self):
        return self._labels

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm = self._permutation[:batch_size]
        self._permutation = self._permutation[batch_size:]
        batch_len = np.max(self._sentence_lens[batch_perm])
        return self._sentences[batch_perm, 0:batch_len], self._sentence_lens[batch_perm], self._labels[batch_perm, 0:batch_len]

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._sentences))
            return True
        return False


class Network:
    def __init__(self, alphabet_size, rnn_cell, rnn_cell_dim, logdir, expname, threads=1, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.summary_writer = tf.train.SummaryWriter("{}/{}-{}".format(logdir, timestamp, expname), flush_secs=10)

        # Construct the graph
        with self.session.graph.as_default():
            if rnn_cell == "LSTM":
                rnn_cell = tf.nn.rnn_cell.LSTMCell(rnn_cell_dim)
            elif rnn_cell == "GRU":
                rnn_cell = tf.nn.rnn_cell.GRUCell(rnn_cell_dim)
            else:
                raise ValueError("Unknown rnn_cell {}".format(rnn_cell))

            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
            self.sentences = tf.placeholder(tf.int32, [None, None])
            self.sentence_lens = tf.placeholder(tf.int32, [None])
            self.labels = tf.placeholder(tf.int32, [None, None])

            # encoded_letters are basically whole words as an input but encoded using one-hot vector
            encoded_letters = tf.one_hot(self.sentences, alphabet_size) # every letter will have its own one-hot vector with the length of an alphabet

            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(rnn_cell, rnn_cell, encoded_letters, dtype=tf.float32, sequence_length=self.sentence_lens)

            self.output_total = output_fw + output_bw

            mask2d = tf.sequence_mask(self.sentence_lens) # creates a 2D tensor of [true, true, ..., true, false, false, ..., false] with length of the max length of 'row' of sentence_lens
            # We want to add another axis to mask2d so it can be applied to the cells' output
            mask3d = tf.pack(np.repeat(mask2d, rnn_cell_dim).tolist(), axis=2)

            masked_labels = tf.boolean_mask(self.labels, mask2d)
            masked_output_total = tf.boolean_mask(self.output_total, mask3d)
            # masked_output_total is currently of shape (?,) wee need shape (?, rnn_cell_dim)
            # because tensor for fully_connected layers must have at least rank 2 (this has rank 1) + last dimensions must be known
            masked_output_total = tf.reshape(masked_output_total, [-1, rnn_cell_dim]) # -1 will infer the shape of the first axis based on the rnn_cell_dim

            output_layer = tf_layers.fully_connected(masked_output_total, 2)
            self.predictions = tf.argmax(output_layer, 1)

            loss = tf_losses.sparse_softmax_cross_entropy(output_layer, masked_labels)
            self.training = tf.train.AdamOptimizer().minimize(loss, self.global_step)
            self.accuracy = tf_metrics.accuracy(tf.cast(self.predictions, dtype=tf.int32), masked_labels)

            '''
            Ruzne vety mohou byt jinak dlouhe -> musi se vyresit (= neni pevna delka grafu)
            Dynamicke nodes:
                outputs, state = tf.nn.dynamic_rnn(cell, inputs, sequence_lens=None, initial_state=None, time_major=False)

                tady ale musime pouzit:
                    outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, ...)
                    vraci dva outputy, jeden output forward site, druhy ouput je backward site
                    vysledny output je soucet tech dvou
            '''

            '''
            TODO: Jak udelat embedding?
                tf.nn.embedding_lookup(weights, indices) (indices je seznam indexu, ktere nas zajimaji)
                vraci radky z matice weights podle toho, co jsme zadali do indices

                tf.set_variable('alpha_emb', shape=[alphabet_size, dim_of_embedding])

            ze states se musi vyrezat jen spravny kus, protoze je to zarovnane nulama
            hodi se k tomu: tf.sequence_mask(lengths, maxlen=None, dtype=tf.bool, name=None)
            + tf.cast()
            '''

            self.dataset_name = tf.placeholder(tf.string, [])
            self.summary = tf.scalar_summary(self.dataset_name+"/accuracy", self.accuracy)

            # Initialize variables
            self.session.run(tf.initialize_all_variables())

    @property
    def training_step(self):
        return self.session.run(self.global_step)

    def train(self, sentences, sentence_lens, labels):
        accuracy, _, summary = self.session.run([self.accuracy, self.training, self.summary],
                                      {self.sentences: sentences, self.sentence_lens: sentence_lens,
                                       self.labels: labels, self.dataset_name: "train"})
        self.summary_writer.add_summary(summary, self.training_step)
        return accuracy

    def evaluate(self, sentences, sentence_lens, labels, dataset_name):
        accuracy, summary = self.session.run([self.accuracy, self.summary], {self.sentences: sentences, self.sentence_lens: sentence_lens,
                                                  self.labels: labels, self.dataset_name: dataset_name})
        self.summary_writer.add_summary(summary, self.training_step)
        return accuracy

def create_and_run_experiment(args):
    print('Running new experiment....')

    expname = "uppercase-letters-{}{}-bs{}-epochs{}".format(args.rnn_cell, args.rnn_cell_dim, args.batch_size, args.epochs)
    print('Experiment name: %s\n' % expname)
    print('Cell type: %s' % args.rnn_cell)
    print('Cell dimension: %d' % args.rnn_cell_dim)
    print('Batch size: %d' % args.batch_size)
    print('Epochs: %d\n' % args.epochs)

    # Load the data
    data_train = Dataset(args.data_train)
    data_dev = Dataset(args.data_dev, data_train.alphabet)
    data_test = Dataset(args.data_test, data_train.alphabet)

    # Construct the network
    network = Network(alphabet_size=len(data_train.alphabet), rnn_cell=args.rnn_cell, rnn_cell_dim=args.rnn_cell_dim, logdir=args.logdir, expname=expname, threads=args.threads)

    # Train
    for epoch in range(args.epochs):
        print("Training epoch {}".format(epoch))
        while not data_train.epoch_finished():
            sentences, sentence_lens, labels = data_train.next_batch(args.batch_size)
            accuracy = network.train(sentences, sentence_lens, labels)
            #print('Accuracy on train dataset: %.5f' % accuracy)
        accuracy_dev = network.evaluate(data_dev.sentences, data_dev.sentence_lens, data_dev.labels, "dev")
        print('Accuracy on the development dataset: %.5f' % accuracy_dev)
        accuracy_test = network.evaluate(data_test.sentences, data_test.sentence_lens, data_test.labels, "test")
        print('Accuracy on the test dataset: %.5f' % accuracy_test)
    print('===========================')

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    cell_dimensions = [10, 20, 40, 80]
    cell_types = ['LSTM', 'GRU']
    epochs_num = [10, 20]
    batch_size = [10, 15, 20, 25]

    for cdim in cell_dimensions:
        for ctype in cell_types:
            for epnum in epochs_num:
                for bsize in batch_size:
                    # Parse arguments
                    import argparse
                    parser = argparse.ArgumentParser()
                    parser.add_argument("--batch_size", default=bsize, type=int, help="Batch size.")
                    parser.add_argument("--data_train", default="en-ud-train.txt", type=str, help="Training data file.")
                    parser.add_argument("--data_dev", default="en-ud-dev.txt", type=str, help="Development data file.")
                    parser.add_argument("--data_test", default="en-ud-test.txt", type=str, help="Testing data file.")
                    parser.add_argument("--epochs", default=epnum, type=int, help="Number of epochs.")
                    parser.add_argument("--logdir", default="logs", type=str, help="Logdir name.")
                    parser.add_argument("--rnn_cell", default=ctype, type=str, help="RNN cell type.")
                    parser.add_argument("--rnn_cell_dim", default=cdim, type=int, help="RNN cell dimension.")
                    parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
                    args = parser.parse_args()

                    create_and_run_experiment(args)
