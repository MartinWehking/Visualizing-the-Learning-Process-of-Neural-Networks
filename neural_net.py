from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import pandas as pd
import tensorflow as tf


class ObservableNet:
    def __init__(self, input_units):
        self.previous_input = self.first_input = tf.layers.Input(shape=[input_units])
        self.agg_gradients = None
        self.gradients = pd.DataFrame(columns=['gradient', 'epoch', 'layer'])
        self.weights = pd.DataFrame(columns=['weight', 'epoch', 'layer'])

    def add_layer(self, units, name, type='dense'):
        if type == 'dense':
            self.previous_input = tf.layers.dense(self.previous_input, units, tf.nn.relu, name=name)

    def create_net(self, learning_rate):
        y_ = tf.placeholder(tf.int32)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.previous_input,
                                                                         labels=y_))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(loss)
        apply_operation = optimizer.apply_gradients(gradients)

        correct_prediction = tf.equal(tf.argmax(self.previous_input, axis=1), tf.argmax(y_, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return gradients, apply_operation, y_, accuracy

    def train(self, epochs=30, learning_rate=0.00001):
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        gradients, apply_operation, y_, accuracy = self.create_net(learning_rate)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                print('Starting epoch: ' + str(epoch))
                save_grad_weights = []
                for i in range(50):
                    #ToDO Better method for shuffling
                    batch = mnist.train.next_batch(1000)
                    batch_train, batch_label = batch[0], batch[1]
                    grad_weights = sess.run(gradients, feed_dict={self.first_input: batch_train, y_: batch_label})
                    save_grad_weights = [np.copy(grad_weight) for i, grad_weight in enumerate(grad_weights) if
                                         i % 2 == 0]
                    self.add_gradients(save_grad_weights)
                    sess.run(apply_operation, feed_dict={self.first_input: batch_train, y_: batch_label})
                test = mnist.test.next_batch(9999)
                test_data, test_labels = test[0], test[1]
                print('testing ' + str(accuracy.eval(feed_dict={self.first_input: test_data, y_: test_labels})))
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
        time_vectors = list()
        if prop == 'gradient':
            data = self.gradients
        else:
            data = self.weights

        data_epochs = data.loc[(data['layer'] == layer)][prop].tolist()
        for row in range(len(data_epochs[0])):
            vector = np.array(tuple([entry[row] for entry in data_epochs ]))
            time_vectors.append(vector.T)
        return time_vectors
