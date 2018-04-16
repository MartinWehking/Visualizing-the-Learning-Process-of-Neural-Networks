from neural_net import ObservableNet, sum_columns
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from multiprocessing import Process
import pandas as pd
from os import getcwd
import logging

dbscan_params_1 = [1 * 10 ** ((-1) * (i + 1)) for i in range(10)]
dbscan_params_2 = [5 * 10 ** ((-1) * (i + 1)) for i in range(10)]
dbscan_params_3 = [2.5 * 10 ** ((-1) * (i + 1)) for i in range(10)]
dbscan_params_4 = [7.5 * 10 ** ((-1) * (i + 1)) for i in range(10)]
dbscan_params = dbscan_params_1 + dbscan_params_2 + dbscan_params_3 + dbscan_params_4
columns = ['removed_label', 'accuracy', 'summed_vectors', 'label', 'epsilon', 'layer', 'g_w']
path = getcwd() + '/results.csv'
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


def do_hac(i, net, time_vectors_weights):
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

def do_kmeans(i, net, time_vectors_weights):
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
    with open(path, 'a') as f:
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
    dataset.to_csv(path, index=False)


def best_results(path):
    results = pd.read_csv(path)
    best_acc = results.groupby('layer')['accuracy'].max()
    b_r = pd.DataFrame(columns=results.columns)
    for acc in best_acc:
        b_r = b_r.append(results.loc[acc == results.accuracy])
    print(b_r)


def remove_all():
    net, time_vectors_gradients, time_vectors_weights = create_time_vectors()


if __name__ == "__main__":
    net, time_vectors_gradients, time_vectors_weights = create_time_vectors()
    for i in range(5):
        i = i + 0
        do_hac(i, net, time_vectors_weights)
    logging.info('Done')

    # start_hac_evaluation()
    #best_results(getcwd()+'/Results/DBSAN.csv')
    # best_results(getcwd()+'/Results/hac.csv')
    # best_results(getcwd()+'/Results/KMeans.csv')
