from PyQt5 import QtWidgets, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from sklearn.decomposition import PCA
from matplotlib.ticker import MaxNLocator
from matplotlib import pyplot as plt

import sys
import math
import numpy as np

from neural_net import ObservableNet, sum_columns


class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.observable_net = self.grad_displays = \
            self.weight_displays = self.weights = self.gradients = self.displays = None

        self.l1_from = self.l1_to = self.l2_from = self.l2_to = self.step = self.adjust_value = \
            self.grad_vectors = self.weight_vectors = None

        self.layer = 0
        self.epochs = 36
        self.initialize_observable_net()
        self.ad_value = None
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

        self.grad_vectors = [observable_net.create_time_vectors
                             ('gradient', layer) for layer in range(5)]
        self.weight_vectors = [observable_net.create_time_vectors
                               ('weight', layer) for layer in range(5)]
        self.grad_displays = self.create_displays(self.grad_vectors)
        self.weight_displays = self.create_displays(self.weight_vectors)

    def init_ui(self):
        self.setMinimumHeight(500)
        self.setMinimumWidth(1000)

        main_group = QtWidgets.QGroupBox('Visualization Settings')
        setting_layout = QtWidgets.QVBoxLayout()
        setting_layout.setAlignment(QtCore.Qt.AlignTop)
        main_group.setMaximumWidth(int(self.width() / 3.5))
        main_group.setLayout(setting_layout)
        h_box = QtWidgets.QHBoxLayout()

        center = QtWidgets.QGroupBox()
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
        self.ad_value = float(self.adjust_value.text())
        self.plot()

    def change_to_combined(self):
        self.vis = 'combined'
        self.plot()

    def create_displays(self, time_vectors):
        displays = list()
        for layer in range(len(time_vectors)):
            display = list()
            for row in time_vectors[layer]:
                # row = MinMaxScaler().fit_transform(row)
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
                layer_items.append('Hidden ' + str(item) + ' - Output')
            else:
                layer_items.append('Hidden ' + str(item) + ' - Hidden ' + str(item + 1))
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
        self.l2_to = l2_to_selection = QtWidgets.QLineEdit('512')
        l2_selection.setLayout(l2_selection_box)
        l2_selection_box.addWidget(l2_from_selection)
        l2_selection_box.addWidget(QtWidgets.QLabel(':'))
        l2_selection_box.addWidget(l2_to_selection)

        layout.addWidget(QtWidgets.QLabel('Show Neurons:'))
        layout.addWidget(QtWidgets.QLabel('Neurons of Y-Axis Layer:'))
        layout.addWidget(l1_selection)
        layout.addWidget(QtWidgets.QLabel('Neurons of X-Axis Layer:'))
        layout.addWidget(l2_selection)
        layout.addWidget(QtWidgets.QLabel('Step:'))
        self.step = QtWidgets.QLineEdit('1')

        layout.addWidget(self.step)

        apply_button = QtWidgets.QPushButton('Apply')
        apply_button.pressed.connect(self.change_values)
        layout.addWidget(apply_button)

        draw_unclustered_vectors = QtWidgets.QPushButton('Draw Time-Vectors')
        draw_unclustered_vectors.pressed.connect(self.plot_time_vectors)
        layout.addWidget(draw_unclustered_vectors)

        self.adjust_value = QtWidgets.QLineEdit('')
        layout.addWidget(self.adjust_value)

        adjust = QtWidgets.QPushButton('Adjust')
        adjust.pressed.connect(self.adjust_mm)
        layout.addWidget(adjust)

    def change_values(self):
        self.plot(l1_from=int(self.l1_from.text()),
                  l1_to=int(self.l1_to.text()), l2_from=int(self.l2_from.text()), l2_to=int(self.l2_to.text()),
                  step=int(self.step.text()))
        self.ad_value = None

    def get_display(self, vis, l1_from, l1_to, l2_from, l2_to, step, remove_first=False):
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
        if 0 <= l1_to <= display.shape[0] and 0 <= l2_to <= display.shape[1] / self.epochs \
                and 0 <= l1_from <= display.shape[0] and l1_to >= l1_from \
                and l2_to >= l2_from and 0 <= l2_to <= display.shape[1] / self.epochs:
            if l1_from == l1_to:
                l1_to = l1_to + 1
            if l2_from == l2_to:
                l2_to = l2_to + 1
            display = display[l1_from:l1_to, l2_from * self.epochs:int(l2_to) * self.epochs]
        new_len = self.epochs
        if 1 < step < self.epochs:
            display = np.delete(display,
                                [i for i in range(int(display.shape[1])) if i % step != 0],
                                axis=1)
            new_len = math.ceil(new_len / step)
        if remove_first:
            display = np.delete(display,
                                [i for i in range(int(display.shape[1])) if i % new_len == 0],
                                axis=1)
        return display

    def plot(self, l1_from=0, l1_to=None, l2_from=0, l2_to=None, step=1):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.clear()

        if self.layer == len(self.grad_displays) - 1:
            ax.set_xlabel('Output Layer')
        else:
            ax.set_xlabel('Hidden Layer ' + str(self.layer + 1))
        if self.layer > 0:
            ax.set_ylabel('Hidden Layer ' + str(self.layer))
        else:
            ax.set_ylabel('Input Layer')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        v_min = v_max = None

        if self.vis == 'gradient':
            cmap = 'RdBu_r'
        else:
            cmap = 'viridis'

        if self.vis is not 'combined':
            display = self.get_display(self.vis, l1_from, l1_to, l2_from, l2_to, step)

            if self.ad_value is None:
                if self.vis == 'gradient':
                    max_value = np.abs(np.max(display))
                    min_value = np.abs(np.min(display))
                    max_value = min_value = np.max([min_value, max_value])
                    max_value = np.sign(np.max(display)) * max_value
                    min_value = np.sign(np.min(display)) * min_value
                    v_min = min_value
                    v_max = max_value
            else:
                v_max = self.ad_value
                v_min = (-1) * self.ad_value

            display = ax.imshow(display, aspect='auto', cmap=cmap, interpolation='None',
                                extent=[0, int(len(display[0]) / self.epochs), len(display), 0], vmin=v_min, vmax=v_max)

            cb = self.figure.colorbar(display, shrink=0.5)
            if self.vis == 'gradient':
                cb.set_label('Gradient')
            else:
                cb.set_label('Weight')
        else:
            display_1 = self.get_display('gradient', l1_from, l1_to, l2_from, l2_to, step)
            display_2 = self.get_display('weight', l1_from, l1_to, l2_from, l2_to, step)

            display_1 = ax.imshow(display_1, aspect='auto', cmap='RdBu', interpolation='None',
                                  extent=[0, int(len(display_1[0]) / self.epochs), len(display_1), 0], vmax=v_max,
                                  vmin=v_min)
            cb = self.figure.colorbar(display_1, shrink=0.5)
            cb.set_label('gradient')
            display_2 = ax.imshow(display_2, aspect='auto', cmap='PuOr', interpolation='None',
                                  extent=[0, int(len(display_2[0]) / self.epochs), len(display_2), 0], vmin=v_min,
                                  vmax=v_max)
            cb_2 = self.figure.colorbar(display_2, shrink=0.5)
            cb_2.set_label('weight')
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
