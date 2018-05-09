from neural_net import ObservableNet, sum_columns
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from multiprocessing import Process
import pandas as pd
from os import getcwd, path
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import logging

dbscan_params_1 = [1 * 10 ** ((-1) * (i + 1)) for i in range(10)]
dbscan_params_2 = [5 * 10 ** ((-1) * (i + 1)) for i in range(10)]
dbscan_params_3 = [2.5 * 10 ** ((-1) * (i + 1)) for i in range(10)]
dbscan_params_4 = [7.5 * 10 ** ((-1) * (i + 1)) for i in range(10)]
dbscan_params = dbscan_params_1 + dbscan_params_2 + dbscan_params_3 + dbscan_params_4
columns = ['removed_label', 'accuracy', 'summed_vectors', 'label', 'epsilon', 'layer', 'g_w']
dir = getcwd() + '/results.csv'
path_layer = getcwd() + '/grads.csv'
path_layer2 = getcwd() + '/weights.csv'


def remove_clusters_evaluate(label, vectors, observable_net, layer):
    results = list()
    label_set = set(label)
    observable_net.save_status()
    for l in label_set:
        for i, vector in enumerate(vectors):
            if label[i] == l:
                observable_net.remove_neuron(layer + 1, i)
        eval = observable_net.test()
        results.append((l, eval))
        observable_net.reset()
    return results

def remove_specific(label, observable_net, layer, remove_label=0):
    for i, l in enumerate(label):
        if l == remove_label:
            observable_net.remove_neuron(layer + 1, i)



def create_ref_architecture():
    observable_net = ObservableNet(784)
    observable_net.add_layer(512, name='hidden', seed=5034)
    observable_net.add_layer(256, name='hidden2', seed=6456)
    observable_net.add_layer(128, name='hidden3', seed=7675)
    observable_net.add_layer(64, name='hidden4', seed=8345)
    observable_net.add_layer(10, name='output', activation='linear', seed=997)
    test_results = observable_net.train(36)

    return observable_net, test_results


def create_time_vectors():
    create_dataset()
    net, test_results = create_ref_architecture()
    time_vectors_gradients = [net.create_time_vectors('gradient', layer) for layer in range(5)]
    time_vectors_weights = [net.create_time_vectors('weight', layer) for layer in range(5)]

    save_layer(time_vectors_gradients, test_results)
    save_layer(time_vectors_weights, test_results, grads=False)

    return net, time_vectors_gradients, time_vectors_weights


def start_dbscan_evaluation():
    net, time_vectors_gradients, time_vectors_weights = create_time_vectors()
    for epsilon in dbscan_params_1:
        for i, layer in enumerate(time_vectors_gradients[:-1]):
            summed_vectors = sum_columns(layer)
            label = DBSCAN(eps=epsilon).fit_predict(summed_vectors)
            results = remove_clusters_evaluate(label, summed_vectors, net, i)
            if len(results) == 1:
                save_results(results[0][0], 0, summed_vectors, label, epsilon, i, 'g')
            else:
                for result in results:
                    save_results(result[0], result[1], summed_vectors, label, epsilon, i, 'g')

    for epsilon in dbscan_params_1:
        for i, layer in enumerate(time_vectors_weights[:-1]):
            summed_vectors = sum_columns(layer)
            label = DBSCAN(eps=epsilon).fit_predict(summed_vectors)
            if len(set(label)) == 1:
                save_results(label[0], 0, summed_vectors, label, epsilon, i, 'w')
            else:
                results = remove_clusters_evaluate(label, summed_vectors, net, i)
                for result in results:
                    save_results(result[0], result[1], summed_vectors, label, epsilon, i, 'w')


def start_kmeans_evaluation():
    net, time_vectors_gradients, time_vectors_weights = create_time_vectors()
    for i in range(70):
        i = i + 1
        for x, layer in enumerate(time_vectors_gradients[:-1]):
            summed_vectors = sum_columns(layer)
            label = KMeans(n_clusters=i, random_state=3125).fit_predict(summed_vectors)
            results = remove_clusters_evaluate(label, summed_vectors, net, x)
            if len(results) == 1:
                save_results(results[0][0], 0, summed_vectors, label, i, x, 'g')
            else:
                for result in results:
                    save_results(result[0], result[1], summed_vectors, label, i, x, 'g')

    for i in range(70):
        i = i + 1
        for x, layer in enumerate(time_vectors_weights[:-1]):
            summed_vectors = sum_columns(layer)
            label = KMeans(n_clusters=i, random_state=3125).fit_predict(summed_vectors)
            results = remove_clusters_evaluate(label, summed_vectors, net, x)
            if len(results) == 1:
                save_results(results[0][0], 0, summed_vectors, label, i, x, 'w')
            else:
                for result in results:
                    save_results(result[0], result[1], summed_vectors, label, i, x, 'w')


def start_hac_evaluation():
    net, time_vectors_gradients, time_vectors_weights = create_time_vectors()
    for i in range(70):
        i = i + 1
        for x, layer in enumerate(time_vectors_gradients[:-1]):
            summed_vectors = sum_columns(layer)
            label = AgglomerativeClustering(n_clusters=i).fit_predict(summed_vectors)
            results = remove_clusters_evaluate(label, summed_vectors, net, x)
            if len(results) == 1:
                save_results(results[0][0], 0, summed_vectors, label, i, x, 'g')
            else:
                for result in results:
                    save_results(result[0], result[1], summed_vectors, label, i, x, 'g')

    for i in range(70):
        i = i + 1
        for x, layer in enumerate(time_vectors_weights[:-1]):
            summed_vectors = sum_columns(layer)
            label = AgglomerativeClustering(n_clusters=i).fit_predict(summed_vectors)
            results = remove_clusters_evaluate(label, summed_vectors, net, x)
            if len(results) == 1:
                save_results(results[0][0], 0, summed_vectors, label, i, x, 'w')
            else:
                for result in results:
                    save_results(result[0], result[1], summed_vectors, label, i, x, 'w')


def do_hac(i, net, time_vectors_gradients):
    i = i + 1
    for x, layer in enumerate(time_vectors_gradients[:-1]):
        summed_vectors = sum_columns(layer)
        label = AgglomerativeClustering(n_clusters=i).fit_predict(summed_vectors)
        results = remove_clusters_evaluate(label, summed_vectors, net, x)
        if len(results) == 1:
            save_results(results[0][0], 0, summed_vectors, label, i, x, 'g')
        else:
            for result in results:
                save_results(result[0], result[1], summed_vectors, label, i, x, 'g')


def do_kmeans(i, net, time_vectors_gradients):
    i = i + 1
    for x, layer in enumerate(time_vectors_gradients[:-1]):
        summed_vectors = sum_columns(layer)
        label = KMeans(n_clusters=i, random_state=3125).fit_predict(summed_vectors)
        results = remove_clusters_evaluate(label, summed_vectors, net, x)
        if len(results) == 1:
            save_results(results[0][0], 0, summed_vectors, label, i, x, 'g')
        else:
            for result in results:
                save_results(result[0], result[1], summed_vectors, label, i, x, 'g')


def do_dbscan(epsilon, net, time_vectors_weights):
    for i, layer in enumerate(time_vectors_weights[:-1]):
        summed_vectors = sum_columns(layer)
        label = DBSCAN(eps=epsilon).fit_predict(summed_vectors)
        if len(set(label)) == 1:
            save_results(label[0], 0, summed_vectors, label, epsilon, i, 'w')
        else:
            results = remove_clusters_evaluate(label, summed_vectors, net, i)
            for result in results:
                save_results(result[0], result[1], summed_vectors, label, epsilon, i, 'w')


def save_results(removed_label, accuracy, summed_vectors, label, epsilon, layer, g_w):
    new_data = pd.DataFrame([(removed_label, accuracy, summed_vectors, label, epsilon, layer, g_w)], columns=columns)
    with open(dir, 'a') as f:
        new_data.to_csv(f, header=False, index=False)


def save_layer(layers, test, grads=True):
    tuples = list()
    for layer in layers:
        tuples.append((layer, test))
    to_save = pd.DataFrame(tuples, columns=['layer', 'accuracy'])
    if grads:
        to_save.to_csv(path_layer)
    else:
        to_save.to_csv(path_layer2)


def create_dataset():
    dataset = pd.DataFrame(columns=columns)
    dataset.to_csv(dir, index=False)


def best_results(path):
    results = pd.read_csv(path)
    results = results.loc[results['g_w'] == 'g']
    best_acc = results.groupby('layer')['accuracy'].max()
    b_r = pd.DataFrame(columns=results.columns)
    for acc in best_acc:
        b_r = b_r.append(results.loc[acc == results.accuracy])
    b_r = b_r.groupby(['layer', 'g_w'])['epsilon'].min()
    print(b_r)


def reproduce_result(layer, param, net, time_vectors_gradients, time_vectors_weights,
                     grad=1, clustering='dbscan', remove_label=0):
    if clustering == 'dbscan':
        if grad == 1:
            label = DBSCAN(param).fit_predict(sum_columns(time_vectors_gradients[layer]))
        else:
            label = DBSCAN(param).fit_predict(sum_columns(time_vectors_weights[layer]))
    elif clustering == 'kmeans':
        if grad == 1:
            label = KMeans(param).fit_predict(sum_columns(time_vectors_gradients[layer]))
        else:
            label = KMeans(param).fit_predict(sum_columns(time_vectors_weights[layer]))
    elif clustering == 'hac':
        if grad == 1:
            label = AgglomerativeClustering(param).fit_predict(sum_columns(time_vectors_gradients[layer]))
        else:
            label = AgglomerativeClustering(param).fit_predict(sum_columns(time_vectors_weights[layer]))
    else:
        raise ValueError('clustering param unknown')
    if grad == 1:
        remove_specific(label, net, layer, remove_label=remove_label)
    else:
        remove_specific(label, net, layer, remove_label=remove_label)


def merge_all():
    hac_all = pd.read_csv(getcwd() + '/Results/hac.csv')
    for i in [x for x in range(47) if x % 5 == 0]:
        if path.isfile(getcwd() + '/Results17.04/Hac_g' + str(i) + '.csv'):
            hac_all = hac_all.append(pd.read_csv(getcwd() + '/Results17.04/Hac_g' + str(i) + '.csv'))
        if path.isfile(getcwd() + '/Results17.04/Hac_w' + str(i) + '.csv'):
            hac_all = hac_all.append(pd.read_csv(getcwd() + '/Results17.04/Hac_w' + str(i) + '.csv'))
    kmeans_all = pd.read_csv(getcwd() + '/Results/KMeans.csv')
    for i in [x for x in range(47) if x % 5 == 0]:
        if path.isfile(getcwd() + '/Results17.04/KMeans_g' + str(i)):
            kmeans_all = kmeans_all.append(pd.read_csv(getcwd() + '/Results17.04/KMeans_g' + str(i) + '.csv'))
        if path.isfile(getcwd() + '/Results17.04/KMeans_w' + str(i) + '.csv'):
            kmeans_all = kmeans_all.append(pd.read_csv(getcwd() + '/Results17.04/KMeans_w' + str(i) + '.csv'))
    hac_all.to_csv(getcwd() + '/Results/Hac_all.csv')
    kmeans_all.to_csv(getcwd() + '/Results/KMeans_all.csv')

def plot_time_vectors(summed_vectors, clustered=False):

        label = DBSCAN(0.025).fit_predict(summed_vectors)
        summed_vectors = PCA().fit_transform(summed_vectors)
        cmap = []
        for l in label:
            if l == 0:
              cmap.append('r')
            else:
                cmap.append('k')
        plt.scatter(summed_vectors[:, 0], summed_vectors[:, 1], c=cmap)
        plt.show()


if __name__ == "__main__":
    net, tg, tw = create_time_vectors()
    # for i in range(10):
    #    i = i + 20
    #    do_kmeans(i, net, time_vectors_gradients)
    # print('Done')
    # merge_all()

    # start_hac_evaluation()
    #best_results(getcwd() + '/Results/Hac_all.csv')
    #l = 1
    #reproduce_result(0, 0.025, net, tg, tw)
    #reproduce_result(1, 48, net, tg, tw, clustering='hac', remove_label=46)
    #reproduce_result(2, 46, net, tg, tw, clustering='hac', remove_label=22)
    reproduce_result(3, 22, net, tg, tw, clustering='hac', remove_label=18)

    print(net.test(testing=0))
    print(net.test(testing=1))

    #plot_time_vectors(sum_columns(tg[0]))
    #plot_time_vectors(sum_columns(tg[1]))
    #plot_time_vectors(sum_columns(tg[2]))
    #plot_time_vectors(sum_columns(tg[3]))

# best_results(getcwd()+'/Results/Hac_all.csv')
# best_results(getcwd()+'/Results/KMeans_all.csv')
