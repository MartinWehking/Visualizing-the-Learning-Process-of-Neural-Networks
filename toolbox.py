from PyQt5 import QtWidgets, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
import sys

from neural_net import ObservableNet


def initialize_data():
    observable_net = ObservableNet(784)
    # observable_net.add_layer(784, name='hidden')
    observable_net.add_layer(784, name='hidden2')
    observable_net.add_layer(10, name='output')
    observable_net.train()

    return observable_net.weights, observable_net.gradients, \
           [observable_net.create_time_vectors('gradient', layer) for layer in range(2)]


class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.weights, self.gradients, tms = initialize_data()
        self.layer = 0
        self.epoch = 0
        self.vis = 'gradient'
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.init_ui()
        print()

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

        epoch_label = QtWidgets.QLabel('Epoch:')
        layout.addWidget(epoch_label)
        epoch_selection = QtWidgets.QComboBox()
        layout.addWidget(epoch_selection)

        epoch_selection.addItems(self.weights['epoch'].apply(str).unique())
        epoch_selection.currentIndexChanged.connect(self.change_epoch)

    def plot(self):
        ax = self.figure.add_subplot(111)
        if self.vis == 'gradient':
            series = self.gradients.loc[
                (self.gradients['epoch'] == self.epoch) & (self.gradients['layer'] == self.layer)]
        else:
            series = self.weights.loc[
                (self.weights['epoch'] == self.epoch) & (self.weights['layer'] == self.layer)]
        values = series[self.vis].tolist()
        ax.clear()
        ax.imshow(values[0], interpolation='nearest', cmap='RdBu', aspect='auto')
        self.canvas.draw()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app_window = Window()
    sys.exit(app.exec_())

