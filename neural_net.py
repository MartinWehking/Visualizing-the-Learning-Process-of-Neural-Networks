import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.random import shuffle
from sklearn.cluster import DBSCAN


class ObservableNet:
    def __init__(self, input_units, seed=3125):
        self.seed = seed
        tf.set_random_seed(self.seed)
        self.previous_input = self.first_input = tf.keras.Input(shape=[input_units], dtype=tf.float64)
        self.agg_gradients = None
        self.mini_batches = 100
        self.y_ = self.accuracy = self.val_data = self.val_labels = self.test_data = self.test_labels = self.sess = None
        self.restore_layers = list()

        self.gradients = pd.DataFrame(columns=['gradient', 'epoch', 'layer'])
        self.weights = pd.DataFrame(columns=['weight', 'epoch', 'layer'])

    def add_layer(self, units, name, type='dense', activation='relu', seed=12222):
        if type == 'dense':
            if activation == 'relu':
                self.previous_input = tf.layers.dense(self.previous_input, units, tf.nn.relu, name=name, use_bias=False,
                                                      kernel_initializer=tf.glorot_uniform_initializer(seed=seed,
                                                                                                       dtype=tf.float64))
            elif activation == 'linear':
                self.previous_input = tf.layers.dense(self.previous_input, units, name=name, use_bias=False,
                                                      kernel_initializer=tf.glorot_uniform_initializer(seed=seed,
                                                                                                       dtype=tf.float64))
            elif activation == 'sigmoid':
                self.previous_input = tf.layers.dense(self.previous_input, units, tf.nn.sigmoid,
                                                      name=name, use_bias=False,
                                                      kernel_initializer=tf.glorot_uniform_initializer(seed=seed,
                                                                                                       dtype=tf.float64))
            else:
                raise AttributeError('Activation has to be relu, linear or sigmoid.')

    def create_net(self, learning_rate):
        y = tf.placeholder(tf.int32, shape=[None, 10])

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.previous_input,
                                                                         labels=y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(loss)
        apply_operation = optimizer.apply_gradients(gradients)

        correct_prediction = tf.equal(tf.argmax(self.previous_input, axis=1), tf.argmax(y, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.y_ = y
        self.accuracy = accuracy
        return gradients, apply_operation, optimizer.variables()

    def train(self, epochs, learning_rate=0.1, bad_training=False):
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        self.val_labels = np.asarray(mnist.validation.labels, dtype=np.int32)
        self.val_data = mnist.validation.images
        complete_train_data = mnist.train.images  # Returns np.array
        complete_train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        self.test_data = mnist.test.images  # Returns np.array
        self.test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

        gradients, apply_operation, variables = self.create_net(learning_rate)

        np.random.seed(self.seed)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.val_labels = self.sess.run(tf.one_hot(self.val_labels, depth=10, dtype=tf.int32))
        self.test_labels = self.sess.run(tf.one_hot(self.test_labels, depth=10, dtype=tf.int32))
        complete_train_labels = self.sess.run(tf.one_hot(complete_train_labels, depth=10, dtype=tf.int32))
        indices = [index for index in range(55000)]
        for epoch in range(epochs):
            print('Starting epoch: ' + str(epoch))
            shuffle(indices)
            if bad_training:
                random_label = [index for index in range(55000)]
                shuffle(random_label)
            for i in range(self.mini_batches):
                train_indices = indices[i * 550: (i + 1) * 550]
                train_data = [complete_train_data[i] for i in train_indices]
                if not bad_training:
                    train_labels = [complete_train_labels[i] for i in train_indices]
                else:
                    train_labels = [complete_train_labels[i] for i in random_label[i * 50: (i + 1) * 50]]
                grad_weights, s = self.sess.run([gradients, apply_operation],
                                                feed_dict={self.first_input: train_data, self.y_:
                                                    train_labels})
                self.add_gradients(grad_weights)
            print('training ' + str(
                self.accuracy.eval(session=self.sess, feed_dict={self.first_input: complete_train_data,
                                                                 self.y_: complete_train_labels})))
            print('testing ' + str(
                self.accuracy.eval(session=self.sess, feed_dict={self.first_input: self.val_data, self.y_:
                    self.val_labels})))
            self.save_weights(grad_weights, epoch)
            self.save_gradients(epoch)
        print(str(self.accuracy.eval(session=self.sess, feed_dict={self.first_input: self.test_data, self.y_:
            self.test_labels})))
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
        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[layer]
        self.sess.run(weights[neuron, :].assign(tf.zeros(weights.shape[1], dtype=tf.float64)))

    def test(self, testing=0):
        if testing == 0:
            return self.accuracy.eval(session=self.sess, feed_dict={self.first_input: self.val_data,
                                                                    self.y_: self.val_labels})
        else:
            return self.accuracy.eval(session=self.sess, feed_dict={self.first_input: self.test_data,
                                                                    self.y_: self.val_labels})

    def save_status(self):
        layers = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
        for element in layers:
            self.restore_layers.append(element.eval(session=self.sess))

    def reset(self):
        layers = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
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
