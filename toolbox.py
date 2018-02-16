from PyQt5 import QtWidgets, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from neural_net import normalize_time_vectors
from sklearn.decomposition import PCA
from matplotlib.ticker import MaxNLocator
from matplotlib import pyplot as plt

import sys
import numpy as np

from neural_net import ObservableNet

class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.time_vectors_gradients = None
        self.time_vectors_weights = None
        self.weights = None
        self.gradients = None
        self.initialize_observable_net()
        self.layer = self.epoch = 0
        self.vis = 'gradient'
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.init_ui()

    def initialize_observable_net(self):
        observable_net = ObservableNet(784)
        observable_net.add_layer(784, name='hidden')
        observable_net.add_layer(10, name='output', activation='linear')
        observable_net.train()

        self.weights = observable_net.weights
        self.gradients = observable_net.gradients
        self.time_vectors_gradients = [observable_net.create_time_vectors('gradient', layer) for layer in range(2)]
        self.time_vectors_weights = [observable_net.create_time_vectors('weight', layer) for layer in range(2)]

    def init_ui(self):
        self.setMinimumHeight(400)
        self.setMinimumWidth(800)

        main_group = QtWidgets.QGroupBox('Visualization Settings')
        setting_layout = QtWidgets.QVBoxLayout()
        setting_layout.setAlignment(QtCore.Qt.AlignTop)
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
        show_net = QtWidgets.QPushButton('Show Neural Network')
        layout.addWidget(show_net)

    def change_layer(self, value):
        self.layer = value
        self.plot()

    def change_epoch(self, value):
        self.epoch = value
        self.plot()

    def change_to_grad(self):
        self.vis = 'gradient'
        self.plot()

    def change_to_weight(self):
        self.vis = 'weight'
        self.plot()

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

        layer_label = QtWidgets.QLabel('Layer:')
        layout.addWidget(layer_label)
        layer_selection = QtWidgets.QComboBox()
        layout.addWidget(layer_selection)

        layer_selection.addItems(self.weights['layer'].apply(str).unique())
        layer_selection.currentIndexChanged.connect(self.change_layer)

        l1_selection = QtWidgets.QGroupBox()
        l1_selection_box = QtWidgets.QHBoxLayout()
        l1_from_selection = QtWidgets.QLineEdit('0')
        l1_to_selection = QtWidgets.QLineEdit('784')
        l1_selection.setLayout(l1_selection_box)
        l1_selection_box.addWidget(l1_from_selection)
        l1_selection_box.addWidget(QtWidgets.QLabel(':'))
        l1_selection_box.addWidget(l1_to_selection)

        l2_selection = QtWidgets.QGroupBox()
        l2_selection_box = QtWidgets.QHBoxLayout()
        l2_from_selection = QtWidgets.QLineEdit('0')
        l2_to_selection = QtWidgets.QLineEdit('784')
        l2_selection.setLayout(l2_selection_box)
        l2_selection_box.addWidget(l2_from_selection)
        l2_selection_box.addWidget(QtWidgets.QLabel(':'))
        l2_selection_box.addWidget(l2_to_selection)

        layout.addWidget(QtWidgets.QLabel('Show Neurons:'))
        layout.addWidget(QtWidgets.QLabel('Neurons of Layer 0:'))
        layout.addWidget(l1_selection)
        layout.addWidget(QtWidgets.QLabel('Neurons of Layer 1:'))
        layout.addWidget(l2_selection)

        # epoch_label = QtWidgets.QLabel('Epoch:')
        # layout.addWidget(epoch_label)
        # epoch_selection = QtWidgets.QComboBox()
        # layout.addWidget(epoch_selection)

        # epoch_selection.addItems(self.weights['epoch'].apply(str).unique())
        # epoch_selection.currentIndexChanged.connect(self.change_epoch)

        from_to1 = QtWidgets.QLabel

    def plot_heatmap(self):
        ax = self.figure.add_subplot(111)
        ax.clear()
        self.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)

        if self.vis == 'gradient':
            series = self.gradients.loc[
                (self.gradients['epoch'] == self.epoch) & (self.gradients['layer'] == self.layer)]
        else:
            series = self.weights.loc[
                (self.weights['epoch'] == self.epoch) & (self.weights['layer'] == self.layer)]
        values = series[self.vis].tolist()
        ax.imshow(values[0], interpolation='nearest', aspect='auto')
        self.canvas.draw()

    def plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.clear()
        a = list()
        if self.vis == 'gradient':
            time_vectors = self.time_vectors_gradients
        else:
            time_vectors = self.time_vectors_weights
        epochs = len(time_vectors[self.layer][0][0])
        for row in time_vectors[self.layer]:
            x = row.flatten()
            a.append(x)
        a = np.array(a)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel('Layer ' + str(self.layer))
        ax.set_ylabel('Layer ' + str(self.layer - 1))
        ax.grid(b=True, which='major', color='k', axis='x', linestyle='--')
        ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
        ax.imshow(a, aspect='auto', cmap='RdGy', interpolation='None', extent=[0, int(len(a[0]) / epochs), len(a), 0])
        ax
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
