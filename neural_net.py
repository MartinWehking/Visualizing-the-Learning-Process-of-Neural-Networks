from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import pandas as pd
import tensorflow as tf
import random as rd
from sklearn.cluster import DBSCAN


class ObservableNet:
    def __init__(self, input_units):
        self.previous_input = self.first_input = tf.layers.Input(shape=[input_units])
        self.agg_gradients = None
        self.gradients = pd.DataFrame(columns=['gradient', 'epoch', 'layer'])
        self.weights = pd.DataFrame(columns=['weight', 'epoch', 'layer'])

    def add_layer(self, units, name, type='dense', activation='relu'):
        if type == 'dense':
            if activation == 'relu':
                self.previous_input = tf.layers.dense(self.previous_input, units, tf.nn.relu, name=name)
            else:
                self.previous_input = tf.layers.dense(self.previous_input, units, name=name)

    def create_net(self, learning_rate):
        y_ = tf.placeholder(tf.int32)
        y = tf.one_hot(y_, depth=10)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.previous_input,
                                                                         labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(loss)
        apply_operation = optimizer.apply_gradients(gradients)

        correct_prediction = tf.equal(tf.argmax(self.previous_input, axis=1), tf.argmax(y, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return gradients, apply_operation, y_, accuracy

    def train(self, epochs=10, learning_rate=0.001):

        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        complete_train_data = mnist.train.images  # Returns np.array
        complete_train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        eval_data = mnist.test.images  # Returns np.array
        eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

        gradients, apply_operation, y_, accuracy = self.create_net(learning_rate)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            indices = [index for index in range(55000)]
            for epoch in range(epochs):
                print('Starting epoch: ' + str(epoch))
                save_grad_weights = []
                rd.shuffle(indices)
                # sess.run([train_image_batch, train_label_batch])
                for i in range(50):
                    train_indices = indices[i * 1000: (i + 1) * 1000]
                    train_data = [complete_train_data[i] for i in train_indices]
                    train_labels = [complete_train_labels[i] for i in train_indices]
                    grad_weights, s = sess.run([gradients, apply_operation], feed_dict={self.first_input: train_data, y_:
                        train_labels})
                    save_grad_weights = [grad_weight for i, grad_weight in enumerate(grad_weights) if
                                         i % 2 == 0]
                    self.add_gradients(save_grad_weights)
                print('testing ' + str(accuracy.eval(feed_dict={self.first_input: eval_data, y_: eval_labels})))
                self.save_weights(save_grad_weights, epoch)
                self.save_gradients(epoch)

    def save_weights(self, grad_weights, epoch):
        weights = [(grad_weight[0], epoch, layer) for layer, grad_weight in enumerate(grad_weights)]
        self.weights = self.weights.append(pd.DataFrame(weights, columns=['weight', 'epoch', 'layer']))

    def add_gradients(self, grad_weights):
        gradients = [grad_weight[1] for grad_weight in grad_weights]
        if self.agg_gradients is None:
            self.agg_gradients = gradients
        else:
            self.agg_gradients = [np.add(self.agg_gradients[i], gradient) for i, gradient in enumerate(gradients)]

    def save_gradients(self, epoch):
        #    ToDo average!
        self.agg_gradients = [(agg_gradient, epoch, layer) for layer, agg_gradient in enumerate(self.agg_gradients)]
        self.gradients = self.gradients.append(pd.DataFrame(self.agg_gradients, columns=['gradient', 'epoch', 'layer']))
        self.agg_gradients = None

    def create_time_vectors(self, prop, layer):
        if prop == 'gradient':
            data = self.gradients
        else:
            data = self.weights

        time_vectors = list()
        data_epochs = data.loc[(data['layer'] == layer)][prop].tolist()
        for row in range(data_epochs[0].shape[0]):
            time_vector_rows = list()
            for col in range(data_epochs[0].shape[1]):
                vector = np.array(tuple([entry[row][col] for entry in data_epochs]))
                time_vector_rows.append(vector.T)
            time_vector_rows = np.array(time_vector_rows)
            time_vectors.append(time_vector_rows)
        return time_vectors


def cluster_time_vectors(self, time_vectors):
    print()

def normalize_time_vectors(time_vectors):
    #ToDo Very Slow. Needs to be changed.
    normalized_time_vectors = list()
    for layer in time_vectors:
        normalized_layer = list()
        for row in layer:
            #normalized_row = list()
            #for time_vector in row:
            #    normalized_row.append(normalize_time_vector(time_vector))
            normalized_layer.append(np.apply_along_axis())
            normalized_layer.append(np.array(normalized_row))
        normalized_time_vectors.append(normalized_layer)
    return normalized_time_vectors


def normalize_time_vector(time_vector):
    minimum = np.min(time_vector)
    maximum = np.max(time_vector)
    normalized_time_vector = list()
    for element in time_vector:
        if element < 0:
            if not minimum == 0:
                normalized_time_vector.append(-element / minimum)
            else:
                normalized_time_vector.append(element)
        else:
            if not maximum == 0:
                normalized_time_vector.append(element / maximum)
            else:
                normalized_time_vector.append(element)
    return np.array(normalized_time_vector)


def cluster_time_vectors(time_vectors):
    DBSCAN()
