import random as rand

import numpy as np
import pandas as pd
import tensorflow as tf


class ObservableNet:
    def __init__(self, input_units):
        self.previous_input = self.first_input = tf.layers.Input(shape=[input_units])
        self.gradients_with_weights = pd.DataFrame()

    def add_layer(self, units, name, type='dense'):
        if type == 'dense':
            self.previous_input = tf.layers.dense(self.previous_input, units, tf.nn.sigmoid, name=name)

    def create_net(self, learning_rate):
        y_ = tf.placeholder(tf.int32)
        y_one_hot = tf.one_hot(y_, 10)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.previous_input,
                                                                         labels=y_one_hot))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(loss)
        apply_operation = optimizer.apply_gradients(gradients)

        correct_prediction = tf.equal(tf.argmax(self.previous_input, 1), tf.argmax(y_one_hot, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return gradients, apply_operation, y_, accuracy

    def train(self, epochs=30, learning_rate=0.1):
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        train_data = mnist.train.images
        train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        train = np.concatenate((train_data, np.matrix(train_labels).T), axis=1)

        test_data = mnist.test.images
        test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

        gradients, apply_operation, y_, accuracy = self.create_net(learning_rate)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                rand.shuffle(train)
                batch_train = train[:, :784]
                batch_label = train[:, 784]
                grad_weights = sess.run(gradients, feed_dict={self.first_input: batch_train, y_: batch_label})
                grad_weights = [grad_weight for i, grad_weight in enumerate(grad_weights) if i % 2 == 0]
                self.save_gradients_with_weights(grad_weights, epoch)
                sess.run(apply_operation, feed_dict={self.first_input: batch_train, y_: batch_label})

            rand.shuffle(train)
            print('training: ' + str(accuracy.eval(feed_dict={self.first_input: test_data, y_: test_labels})))
            print('testing ' + str(accuracy.eval(feed_dict={self.first_input: train_data, y_: train_labels})))

    def save_gradients_with_weights(self, grad_weights, epoch, overwrite=False):
        grad_weight = [(grad_weight[0], grad_weight[1], epoch, i) for i, grad_weight in
                       enumerate(grad_weights)]
        frame = pd.DataFrame(grad_weight, columns=['gradient', 'weight', 'epoch', 'layer'])
        if overwrite:
            self.gradients_with_weights = frame
        else:
            self.gradients_with_weights = self.gradients_with_weights.append(frame)
