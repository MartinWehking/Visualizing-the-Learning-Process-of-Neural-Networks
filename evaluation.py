from neural_net import ObservableNet, sum_columns, cluster_time_vectors
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
import pandas as pd
from os import getcwd

dbscan_params = [1 * 10 ** ((-1) * (i + 1)) for i in range(10)] + [5 * 10 ** ((-1) * (i + 1)) for i in range(10)] \
                + [2.5 * 10 ** ((-1) * (i + 1)) for i in range(10)] + [7.5 * 10 ** ((-1) * (i + 1)) for i in range(10)]
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
                observable_net.remove_neuron(layer, i)
        eval = observable_net.test()
        results.append((l, eval))
        observable_net.reset()
    return results


def create_ref_architecture():
    observable_net = ObservableNet(784)
    observable_net.add_layer(512, name='hidden')
    observable_net.add_layer(256, name='hidden2')
    observable_net.add_layer(128, name='hidden3')
    observable_net.add_layer(64, name='hidden4')
    observable_net.add_layer(10, name='output', activation='linear')
    test_results = observable_net.train(12)

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
    for epsilon in dbscan_params:
        for i, layer in enumerate(time_vectors_gradients):
            summed_vectors = sum_columns(layer)
            label = DBSCAN(eps=epsilon).fit_predict(summed_vectors)
            results = remove_clusters_evaluate(label, summed_vectors, net, i)
            if len(results) == 1:
                save_results(results[0][0], 0, summed_vectors, label, epsilon, i, 'g')
            else:
                for result in results:
                    save_results(result[0], result[1], summed_vectors, label, epsilon, i, 'g')

    for epsilon in dbscan_params:
        for i, layer in enumerate(time_vectors_weights):
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
        for x, layer in enumerate(time_vectors_gradients):
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
        for x, layer in enumerate(time_vectors_weights):
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
        for x, layer in enumerate(time_vectors_gradients):
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
        for x, layer in enumerate(time_vectors_weights):
            summed_vectors = sum_columns(layer)
            label = AgglomerativeClustering(n_clusters=i).fit_predict(summed_vectors)
            results = remove_clusters_evaluate(label, summed_vectors, net, x)
            if len(results) == 1:
                save_results(results[0][0], 0, summed_vectors, label, i, x, 'w')
            else:
                for result in results:
                    save_results(result[0], result[1], summed_vectors, label, i, x, 'w')


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


def load_results(path):
    results = pd.read_csv(path)
    print()


if __name__ == "__main__":
    start_kmeans_evaluation()
    #load_results(getcwd()+'/results.csv')
