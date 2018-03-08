from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import pandas as pd
import tensorflow as tf
import random as rd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler


class ObservableNet:
    def __init__(self, input_units):
        self.previous_input = self.first_input = tf.layers.Input(shape=[input_units])
        self.agg_gradients = None
        self.mini_batches = 1000
        self.y_ = None
        self.accuracy = None
        self.test_data = None
        self.test_labels = None
        self.sess = None
        self.restore_layers = list()

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
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(loss)
        apply_operation = optimizer.apply_gradients(gradients)

        correct_prediction = tf.equal(tf.argmax(self.previous_input, axis=1), tf.argmax(y, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.y_ = y_
        self.accuracy = accuracy
        return gradients, apply_operation, optimizer.variables()

    def train(self, epochs=5, learning_rate=0.05, bad_training=False):

        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        complete_train_data = mnist.train.images  # Returns np.array
        complete_train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        self.test_data = mnist.test.images  # Returns np.array
        self.test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

        gradients, apply_operation, variables = self.create_net(learning_rate)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        indices = [index for index in range(55000)]
        for epoch in range(epochs):
            print('Starting epoch: ' + str(epoch))
            save_grad_weights = []
            rd.shuffle(indices)
            if bad_training:
                random_label = [index for index in range(55000)]
                rd.shuffle(random_label)
            # sess.run([train_image_batch, train_label_batch])
            for i in range(self.mini_batches):
                train_indices = indices[i * 55: (i + 1) * 55]
                train_data = [complete_train_data[i] for i in train_indices]
                if not bad_training:
                    train_labels = [complete_train_labels[i] for i in train_indices]
                else:
                    train_labels = [complete_train_labels[i] for i in random_label[i * 50: (i + 1) * 50]]
                grad_weights, s = self.sess.run([gradients, apply_operation],
                                                feed_dict={self.first_input: train_data, self.y_:
                                                    train_labels})
                save_grad_weights = [grad_weight for i, grad_weight in enumerate(grad_weights) if
                                     i % 2 == 0]
                self.add_gradients(save_grad_weights)
            # x = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[2]
            # l = x[:, 0].assign(tf.zeros(512))
            # x = sess.run(l)
            # print(x)
            print('training ' + str(
                self.accuracy.eval(session=self.sess, feed_dict={self.first_input: complete_train_data,
                                                                 self.y_: complete_train_labels})))
            print('testing ' + str(
                self.accuracy.eval(session=self.sess, feed_dict={self.first_input: self.test_data, self.y_:
                    self.test_labels})))
            self.save_weights(save_grad_weights, epoch)
            self.save_gradients(epoch)
        return self.accuracy.eval(session=self.sess, feed_dict={self.first_input: self.test_data, self.y_:
                    self.test_labels})

    def save_weights(self, grad_weights, epoch):
        weights = [(grad_weight[1], epoch, layer) for layer, grad_weight in enumerate(grad_weights)]
        self.weights = self.weights.append(pd.DataFrame(weights, columns=['weight', 'epoch', 'layer']))

    def add_gradients(self, grad_weights):
        gradients = [grad_weight[0] for grad_weight in grad_weights]
        if self.agg_gradients is None:
            self.agg_gradients = gradients
        else:
            self.agg_gradients = [np.add(self.agg_gradients[i], gradient) for i, gradient in enumerate(gradients)]

    def save_gradients(self, epoch):
        self.agg_gradients = [(agg_gradient / self.mini_batches, epoch, layer)
                              for layer, agg_gradient in enumerate(self.agg_gradients)]
        self.gradients = self.gradients.append(pd.DataFrame(self.agg_gradients, columns=['gradient', 'epoch', 'layer']))
        self.agg_gradients = None

    def remove_neuron(self, layer, neuron):
        l = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        neurons = [la for i, la in enumerate(l) if i % 2 == 0][layer]
        self.sess.run(neurons[neuron, :].assign(tf.zeros(neurons.shape[1])))

    def test(self):
        return self.accuracy.eval(session=self.sess, feed_dict={self.first_input: self.test_data,
                                                                self.y_: self.test_labels})

    def save_status(self):
        layers = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
        layers = [x for i, x in enumerate(layers) if i % 2 == 0]
        for element in layers:
            self.restore_layers.append(element.eval(session=self.sess))

    def reset(self):
        train_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        layers = [la for i, la in enumerate(train_var) if i % 2 == 0]
        for i, layer in enumerate(layers):
            self.sess.run(layers[i].assign(self.restore_layers[i]))


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


def normalize_time_vectors(time_vectors):
    normalized_time_vectors = list()
    for layer in time_vectors:
        normalized_layer = list()
        for row in layer:
            # normalized_row = np.apply_along_axis(lambda x: MinMaxScaler().fit_transform(np.asmatrix(x)), 0, row)
            row = np.asmatrix(row)
            # normalized_row = (row - np.min(row, axis=1)) / (np.max(row, axis=1) - np.min(row, axis=1))
            normalized_row = list()
            for time_vector in row:
                scaler = MinMaxScaler()
                scaler.fit(time_vector)
                normalized_row.append(scaler.transform(time_vector))
            normalized_layer.append(np.array(normalized_row))
        normalized_time_vectors.append(normalized_layer)
    return normalized_time_vectors


def get_all_time_vectors(time_vectors):
    all_time_vectors = list()
    for row in time_vectors:
        for time_vector in row:
            all_time_vectors.append(time_vector)
    return all_time_vectors


def cluster_time_vectors(time_vectors, epsilon):
    clustered_vectors = DBSCAN(eps=epsilon).fit_predict(time_vectors)
    return clustered_vectors


def sum_columns(time_vectors):
    end_vector = np.copy(time_vectors[0])
    for time_vector in time_vectors[1:]:
        end_vector += time_vector
    return end_vector
