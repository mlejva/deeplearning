# The `resnet_v1_50.ckpt` can be downloaded from http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz

from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import tensorflow.contrib.losses as tf_losses
import tensorflow.contrib.metrics as tf_metrics
import tensorflow.contrib.slim as tf_slim
import tensorflow.contrib.slim.nets

class Network:
    WIDTH = 224
    HEIGHT = 224
    CLASSES = 1000

    def __init__(self, checkpoint, threads):
        # Create the session
        self.session = tf.Session(graph = tf.Graph(), config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                            intra_op_parallelism_threads=threads))

        with self.session.graph.as_default():
            # Construct the model
            self.images = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 3])

            with tf_slim.arg_scope(tf_slim.nets.resnet_v1.resnet_arg_scope(is_training=False)):
                self.resnet, _ = tf_slim.nets.resnet_v1.resnet_v1_50(self.images, num_classes=None)

            # Load the checkpoint
            self.saver = tf.train.Saver()
            self.saver.restore(self.session, checkpoint)

            # JPG loading
            self.jpeg_file = tf.placeholder(tf.string, [])
            self.jpeg_data = tf.image.resize_image_with_crop_or_pad(tf.image.decode_jpeg(tf.read_file(self.jpeg_file), channels=3), self.HEIGHT, self.WIDTH)

    def load_jpeg(self, jpeg_file):
        return self.session.run(self.jpeg_data, {self.jpeg_file: jpeg_file})

    def predict(self, image):
        return self.session.run(self.predictions, {self.images: [image]})[0]

    def eval_resnet(self, image):
      	return self.session.run(self.resnet, {self.images: [image]})[0]


def save_numpy_array(filename, arr):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.save(filename, arr)

def save_resnet_features(path, dataset):
    export_path_train = 'export/train/'
    export_path_test = 'export/test/'

    all_features = []
    all_labels = []
    for (dirpath, _, _) in os.walk(path):
        print(dirpath)
        all_classes = [f for f in os.listdir(path) if not f.startswith('.')] # Remove all the hidden files

        for filename in os.listdir(dirpath):
            if filename.endswith('.jpg'):
                print(dirpath+'/'+filename)
                relative_image_path = dirpath+'/'+filename

                image_data = network.load_jpeg(relative_image_path)
                resnet_features = network.eval_resnet(image_data)
                resnet_features = resnet_features[0,0,:]
                all_features.append(resnet_features)

                image_class = dirpath.split('/')[-1] # Get the class of image from its path
                image_class_index = all_classes.index(image_class)
                all_labels.append(image_class_index)
        print('========')

    export_features = np.array(all_features)
    export_labels = np.array(all_labels)
    print('all_features: %s' % (export_features.shape,))
    print('all_labels: %s' % (export_labels.shape,))

    features_path = ''
    labels_path = ''
    if dataset == 'test':
        features_path = export_path_test+'test_features.npy'
        labels_path = export_path_test+'test_labels.npy'
    else:
        features_path = export_path_train+'train_features.npy'
        labels_path = export_path_train+'train_labels.npy'
    save_numpy_array(features_path, export_features)
    save_numpy_array(labels_path, export_labels)

if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="resnet_v1_50.ckpt", type=str, help="Name of ResNet50 checkpoint.")
    parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Load the network
    network = Network(args.checkpoint, args.threads)

    images_train_path = 'subcaltech-50/train/'
    images_test_path = 'subcaltech-50/test/'

    save_resnet_features(images_train_path, 'train')
    save_resnet_features(images_test_path, 'test')
