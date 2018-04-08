from PyQt5 import QtWidgets, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from sklearn.decomposition import PCA
from matplotlib.ticker import MaxNLocator
from matplotlib import pyplot as plt

import sys
import numpy as np

from neural_net import ObservableNet, sum_columns


class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.observable_net = self.grad_displays = \
            self.weight_displays = self.weights = self.gradients = self.displays = None

        self.l1_from = self.l1_to = self.l2_from = self.l2_to = None

        self.layer = 0
        self.epochs = 50
        self.initialize_observable_net()
        self.vis = 'gradient'

        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.init_ui()
        self.update_texts()

    def initialize_observable_net(self):
        observable_net = ObservableNet(784)
        observable_net.add_layer(512, name='hidden', seed=5034)
        observable_net.add_layer(256, name='hidden2', seed=6456)
        observable_net.add_layer(128, name='hidden3', seed=7675)
        observable_net.add_layer(64, name='hidden4', seed=8345)
        observable_net.add_layer(10, name='output', activation='linear', seed=997)
        observable_net.train(self.epochs)
        self.observable_net = observable_net

        self.weights = observable_net.weights
        self.gradients = observable_net.gradients
        self.grad_displays = self.create_displays([observable_net.create_time_vectors
                                                   ('gradient', layer) for layer in range(5)])
        self.weight_displays = self.create_displays([observable_net.create_time_vectors
                                                     ('weight', layer) for layer in range(5)])

    def init_ui(self):
        self.setMinimumHeight(500)
        self.setMinimumWidth(1000)

        main_group = QtWidgets.QGroupBox('Visualization Settings')
        setting_layout = QtWidgets.QVBoxLayout()
        setting_layout.setAlignment(QtCore.Qt.AlignTop)
        main_group.setLayout(setting_layout)
        h_box = QtWidgets.QHBoxLayout()

        center = QtWidgets.QGroupBox()
        #center.setMinimumWidth((int(self.width() / 3)) * 2)
        #main_group.setMaximumWidth(int(self.width() / 3))

        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.toolbar)
        left.addWidget(self.canvas)
        self.init_bottom(left)
        center.setLayout(left)
        self.init_settings(setting_layout)

        h_box.addWidget(center)
        h_box.addWidget(main_group)

        self.setLayout(h_box)
        self.plot()
        self.show()

    def init_bottom(self, layout):
        show_net = QtWidgets.QPushButton('Neural Network Settings')
        layout.addWidget(show_net)

    def change_layer(self, value):
        self.layer = value
        self.update_texts()
        self.plot()

    def update_texts(self):
        self.l1_from.setText('0')
        self.l2_from.setText('0')
        self.l1_to.setText(str(self.grad_displays[self.layer].shape[0]))
        self.l2_to.setText(str(int(self.grad_displays[self.layer].shape[1] / self.epochs)))

    def change_to_grad(self):
        self.vis = 'gradient'
        self.plot()

    def change_to_weight(self):
        self.vis = 'weight'
        self.plot()

    def adjust_mm(self):
        return

    def change_to_combined(self):
        self.vis = 'combined'
        self.plot()

    def create_displays(self, time_vectors):
        displays = list()
        for layer in range(len(time_vectors)):
            display = list()
            for row in time_vectors[layer]:
                x = row.flatten()
                display.append(x)
            displays.append(np.array(display))
        return displays

    def init_settings(self, layout):
        vis_label = QtWidgets.QLabel('Properties:')
        layout.addWidget(vis_label)

        buttons = QtWidgets.QGroupBox()
        h_box = QtWidgets.QHBoxLayout()
        buttons.setLayout(h_box)
        layout.addWidget(buttons)

        first = QtWidgets.QRadioButton('Gradients')
        first.toggle()
        first.toggled.connect(self.change_to_grad)
        h_box.addWidget(first)

        second = QtWidgets.QRadioButton('Weights')
        second.toggled.connect(self.change_to_weight)
        h_box.addWidget(second)

        third = QtWidgets.QRadioButton('Combination')
        third.toggled.connect(self.change_to_combined)
        h_box.addWidget(third)

        layer_label = QtWidgets.QLabel('Layer:')
        layout.addWidget(layer_label)
        layer_selection = QtWidgets.QComboBox()
        layout.addWidget(layer_selection)

        layer_items = list()
        for item in self.weights['layer'].unique():
            if item == 0:
                layer_items.append('Input - Hidden 1')
            elif item == len(self.weights['layer'].unique()) - 1:
                layer_items.append('Hidden ' + str(item - 1) + ' - Output')
            else:
                layer_items.append('Hidden ' + str(item - 1) + ' - Hidden ' + str(item))
        layer_selection.addItems(layer_items)
        layer_selection.currentIndexChanged.connect(self.change_layer)

        l1_selection = QtWidgets.QGroupBox()
        l1_selection_box = QtWidgets.QHBoxLayout()
        self.l1_from = l1_from_selection = QtWidgets.QLineEdit('0')
        self.l1_to = l1_to_selection = QtWidgets.QLineEdit('784')
        l1_selection.setLayout(l1_selection_box)
        l1_selection_box.addWidget(l1_from_selection)
        l1_selection_box.addWidget(QtWidgets.QLabel(':'))
        l1_selection_box.addWidget(l1_to_selection)

        l2_selection = QtWidgets.QGroupBox()
        l2_selection_box = QtWidgets.QHBoxLayout()
        self.l2_from = l2_from_selection = QtWidgets.QLineEdit('0')
        self.l2_to = l2_to_selection = QtWidgets.QLineEdit('784')
        l2_selection.setLayout(l2_selection_box)
        l2_selection_box.addWidget(l2_from_selection)
        l2_selection_box.addWidget(QtWidgets.QLabel(':'))
        l2_selection_box.addWidget(l2_to_selection)

        layout.addWidget(QtWidgets.QLabel('Show Neurons:'))
        layout.addWidget(QtWidgets.QLabel('Neurons of Y-Axis Layer:'))
        layout.addWidget(l1_selection)
        layout.addWidget(QtWidgets.QLabel('Neurons of X-Axis Layer:'))
        layout.addWidget(l2_selection)

        apply_button = QtWidgets.QPushButton('Apply')
        apply_button.pressed.connect(self.change_size)
        layout.addWidget(apply_button)

        draw_unclustered_vectors = QtWidgets.QPushButton('Draw Time-Vectors')
        draw_unclustered_vectors.pressed.connect(self.plot_time_vectors)
        layout.addWidget(draw_unclustered_vectors)

        adjust = QtWidgets.QPushButton('Adjust Min/Max')
        # adjust.pressed.connect()

    def change_size(self):
        self.plot(l1_from=int(self.l1_from.text()),
                  l1_to=int(self.l1_to.text()), l2_from=int(self.l2_from.text()), l2_to=int(self.l2_to.text()))

    def get_display(self, vis, l1_from, l1_to, l2_from, l2_to):
        if vis == 'gradient':
            display = self.grad_displays[self.layer]
        else:
            display = self.weight_displays[self.layer]
        if l1_from is None:
            l1_from = 0
        if l2_from is None:
            l2_from = 0
        if l1_to is None:
            l1_to = display.shape[0]
        if l2_to is None:
            l2_to = display.shape[1] / self.epochs
        #ToDo l2_from Fix
        if 0 <= l1_to <= display.shape[0] and 0 <= l2_to <= display.shape[1] / self.epochs \
                and 0 <= l1_from <= display.shape[0] and l1_to >= l1_from \
                and l2_to >= l2_from and 0 <= l2_to <= display.shape[1] / self.epochs:
            if l1_from == l1_to:
                l1_to = l1_to + 1
            if l2_from == l2_to:
                l2_to = l2_to + 1
            display = display[l1_from:l1_to, l2_from:int(l2_to) * self.epochs]
        return display

    def plot(self, adjust_min=None, adjust_max=None, l1_from=0, l1_to=None, l2_from=0, l2_to=None):

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if self.layer == len(self.grad_displays) - 1:
            ax.set_xlabel('Output Layer')
        else:
            ax.set_xlabel('Hidden Layer ' + str(self.layer))
        if self.layer > 0:
            ax.set_ylabel('Hidden Layer ' + str(self.layer - 1))
        else:
            ax.set_ylabel('Input Layer')
        ax.clear()

        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # ax.grid(b=True, which='major', color='k', axis='x', linestyle='--')
        if self.vis is not 'combined':
            display = self.get_display(self.vis, l1_from, l1_to, l2_from, l2_to)
            display = ax.imshow(display, aspect='auto', cmap='RdGy', interpolation='None',
                                extent=[0, int(len(display[0]) / self.epochs), len(display), 0], vmin=adjust_min,
                                vmax=adjust_max)

            cb = self.figure.colorbar(display, shrink=0.5)
            cb.set_label(self.vis)
        else:
            display_1 = self.get_display('gradient', l1_from, l1_to, l2_from, l2_to)
            display_2 = self.get_display('weight', l1_from, l1_to, l2_from, l2_to)

            display_1 = ax.imshow(display_1, aspect='auto', cmap='RdGy', interpolation='None',
                                  extent=[0, int(len(display_1[0]) / self.epochs), len(display_1), 0], vmin=adjust_min,
                                  vmax=adjust_max)
            cb = self.figure.colorbar(display_1, shrink=0.5)
            cb.set_label('gradient')
            display_2 = ax.imshow(display_2, aspect='auto', cmap='PuOr', interpolation='None',
                                  extent=[0, int(len(display_2[0]) / self.epochs), len(display_2), 0], vmin=adjust_min,
                                  vmax=adjust_max)
            cb_2 = self.figure.colorbar(display_2, shrink=0.5)
            cb_2.set_label('weight')

        # if self.vis == 'combined':
        #    other_display = self.create_display(other_vectors, l1_from, l1_to, l2_from, l2_to)
        #    other_display = ax.imshow(other_display, aspect='auto', cmap='PuOr', interpolation='None',
        #                              extent=[0, int(len(image[0]) / self.epochs), len(image), 0], vmin=adjust_min,
        #                             vmax=adjust_max)
        self.canvas.draw()

    def plot_time_vectors(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.clear()

        if self.vis == 'gradient':
            tl = self.time_vectors_gradients[self.layer]
        else:
            tl = self.time_vectors_weights[self.layer]

        summed_vectors = sum_columns(tl)
        ax.scatter(summed_vectors[:, 0], summed_vectors[:, 1])
        ax.scatter(summed_vectors[:, 0], summed_vectors[:, 1])
        self.canvas.draw()

    def visualize_time_vectors(self, layer):
        vectors = list()
        for row in self.time_vectors[layer]:
            for vector in row:
                vectors.append(vector)
        representations = PCA().fit_transform(vectors)
        plt.scatter(representations[:, 0], representations[:, 1])


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app_window = Window()
    sys.exit(app.exec_())
