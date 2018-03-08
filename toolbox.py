from PyQt5 import QtWidgets, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from sklearn.decomposition import PCA
from matplotlib.ticker import MaxNLocator
from matplotlib import pyplot as plt
from networkx import Graph, draw_networkx, get_node_attributes
from functools import partial

import sys
import numpy as np

from neural_net import ObservableNet, cluster_time_vectors, sum_columns


class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.observable_net = None
        self.time_vectors_gradients = None
        self.time_vectors_weights = None
        self.weights = None
        self.gradients = None

        self.normalized_weights = None
        self.normalized_gradients = None

        self.l1_from = None
        self.l1_to = None
        self.l2_from = None
        self.l2_to = None

        self.layer = self.epoch = 0
        self.epsilon = 0.01
        self.initialize_observable_net()
        self.vis = 'gradient'
        self.s_normalized = False

        #self.normalized_weights = normalize_time_vectors(self.time_vectors_weights)
        #self.normalized_gradients = normalize_time_vectors(self.time_vectors_gradients)

        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.init_ui()

    def initialize_observable_net(self):
        observable_net = ObservableNet(784)
        observable_net.add_layer(512, name='hidden')
        observable_net.add_layer(256, name='hidden2')
        observable_net.add_layer(128, name='hidden3')
        observable_net.add_layer(64, name='hidden4')
        observable_net.add_layer(10, name='output', activation='linear')
        observable_net.train()
        self.observable_net = observable_net

        self.weights = observable_net.weights
        self.gradients = observable_net.gradients
        self.time_vectors_gradients = [observable_net.create_time_vectors('gradient', layer) for layer in range(5)]
        self.time_vectors_weights = [observable_net.create_time_vectors('weight', layer) for layer in range(5)]
        #self.normalized_weights = normalize_time_vectors(self.time_vectors_weights)
        #self.normalized_gradients = normalize_time_vectors(self.time_vectors_gradients)

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

    def change_epsilon(self, value):
        print('hi')
        self.epsilon = value
        self.plot_time_vectors()

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
        layout.addWidget(QtWidgets.QLabel('Neurons of Layer 0:'))
        layout.addWidget(l1_selection)
        layout.addWidget(QtWidgets.QLabel('Neurons of Layer 1:'))
        layout.addWidget(l2_selection)

        apply_button = QtWidgets.QPushButton('Apply')
        apply_button.pressed.connect(self.change_size)
        layout.addWidget(apply_button)

        draw_unclustered_vectors = QtWidgets.QPushButton('Draw Time-Vectors')
        draw_unclustered_vectors.pressed.connect(self.plot_time_vectors)
        layout.addWidget(draw_unclustered_vectors)

        layout.addWidget(QtWidgets.QLabel('Clustering Epsilon:'))
        epsilon = QtWidgets.QLineEdit('0.01')
        epsilon.returnPressed.connect(partial(self.change_epsilon, float(epsilon.text())))
        layout.addWidget(epsilon)


        #normalize_button = QtWidgets.QPushButton('Show normalized Time-Vectors')
        #normalize_button.pressed.connect(self.show_normalized)
        #layout.addWidget(normalize_button)

        # epoch_label = QtWidgets.QLabel('Epoch:')
        # layout.addWidget(epoch_label)
        # epoch_selection = QtWidgets.QComboBox()
        # layout.addWidget(epoch_selection)

        # epoch_selection.addItems(self.weights['epoch'].apply(str).unique())
        # epoch_selection.currentIndexChanged.connect(self.change_epoch)

        from_to1 = QtWidgets.QLabel

    def show_normalized(self):
        if self.s_normalized:
            self.s_normalized = False
        else:
            self.s_normalized = True
        self.plot()

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

    def change_size(self):
        self.plot(int(self.l1_from.text()), int(self.l1_to.text()), int(self.l2_from.text()), int(self.l2_to.text()))

    def plot(self, l1_from=0, l1_to=None, l2_from=0, l2_to=None):
        if self.vis == 'gradient':
            if self.s_normalized:
                time_vectors = self.normalized_gradients
            else:
                time_vectors = self.time_vectors_gradients
        else:
            if self.s_normalized:
                time_vectors = self.normalized_weights
            else:
                time_vectors = self.time_vectors_weights

        if l1_to is None:
            l1_to = len(time_vectors[self.layer])
        if l2_to is None:
            l2_to = len(time_vectors[self.layer][0]) * len(time_vectors[self.layer][0][0])
        else:
            l2_to = l2_to * len(time_vectors[self.layer][0][0])

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.clear()
        a = list()
        epochs = len(time_vectors[self.layer][0][0])
        for i, row in enumerate(time_vectors[self.layer]):
            if l1_from <= i <= l1_to:
                x = row.flatten()[l2_from:l2_to]
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
        self.canvas.draw()

    def plot_time_vectors(self, clustered=True):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.clear()

        if self.vis == 'gradient':
            tl = self.time_vectors_gradients[self.layer]
        else:
            tl = self.time_vectors_weights[self.layer]

        summed_vectors = sum_columns(tl)
        pca = PCA(n_components=2).fit_transform(summed_vectors)
        if clustered:
            label = cluster_time_vectors(summed_vectors, epsilon=self.epsilon)
            remove_clusters_evaluate(label, summed_vectors, self.observable_net, self.layer + 1)
            ax.scatter(pca[:, 0], pca[:, 1], c=label)
        else:
            ax.scatter(pca[:, 0], pca[:, 1])
        self.canvas.draw()

    def visualize_time_vectors(self, layer):
        vectors = list()
        for row in self.time_vectors[layer]:
            for vector in row:
                vectors.append(vector)
        representations = PCA().fit_transform(vectors)
        plt.scatter(representations[:, 0], representations[:, 1])


def visualize_embeddings_networkx(embeddings, step=None):
    if step is None:
        step = max(embeddings['step'].tolist())
    vis_embeddings = embeddings[str(step)]
    pca_embeddings = PCA(n_components=2).fit_transform(vis_embeddings.tolist())
    embedding_graph = Graph()
    color_map = []
    for index, embedding in enumerate(pca_embeddings):
        node_number = embeddings['node'].tolist()[index]
        embedding_graph.add_node(node_number, pos=embedding)
        node_step = embeddings['step'].tolist()[index]
        color_map.append(int(10 * node_step))
    # edge_colors = []
    # if original_graph is not None:
    #    for edge in original_graph.edges():
    #       embedding_graph.add_edge(edge[0], edge[1])
    #       if mark_step:
    #           if str(edge[0]) in frequent_nodes or str(edge[1]) in frequent_nodes:
    #               edge_colors.append(0.1)
    #           else:
    #             edge_colors.append(0.2)
    #
    draw_networkx(embedding_graph, pos=get_node_attributes(embedding_graph, 'pos'),
                  node_color=color_map, font_color='r')
    plt.show()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app_window = Window()
    sys.exit(app.exec_())
