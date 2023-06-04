import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
import KMedian
import random

clusters = None


def set_no_of_clusters():
    global clusters
    clusters = 6    # int(input("Enter no of clusters:"))


def create_graph(nodes=[], values=[]):
    # threshold = 0.05
    threshold = 0.02 # float(input("Enter Threshold value:"))
    graph = nx.Graph()
    graph.add_nodes_from(nodes)

    a = 0
    for node in nodes:
        for i in range(len(nodes)):
            if values[a][i] > threshold:
                graph.add_edge(node, nodes[i], weight=values[a][i])
        a += 1

    return graph


def draw_graph(graph=nx.Graph(), graph_dict={}, pos={}):
    colors = []
    for key in graph_dict.keys():
        r = random.randint(0, 100) / 100
        g = random.randint(0, 100) / 100
        b = random.randint(0, 100) / 100
        colors.append([r, g, b])
        lst = list(graph_dict[key])
        # print(lst)
        nx.draw(graph, pos=pos, nodelist=list(graph_dict[key]), with_labels=True, node_color=[[r, g, b]])

    plt.show()


def optimum_k(X=np.array([])):
    # k means determine k
    distortions = []
    K = range(1, 20)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


def cluster_graph(positions={}, cluster=None):

    nodes, locations = zip(*positions.items())

    if cluster is None:
        cluster = KMeans()

    cluster.n_clusters = clusters
    cluster.fit(np.array(locations))

    labels = cluster.labels_

    clustered_dict = {}
    for label, clas in zip(labels, nodes):
        if clustered_dict.get(label) is None:
            clustered_dict[label] = [clas]
        else:
            temp = clustered_dict.get(label)
            temp.append(clas)
            clustered_dict[label] = temp
    return clustered_dict

if __name__ == '__main__':

    input_data = pd.read_csv('data/mig data2016.csv', index_col=0)  # import data
    values = input_data.values
    normalized = normalize(values)
    upper_tri = np.triu(normalized)
    lower_tri = np.tril(normalized)
    cities = input_data.axes[0].values

    for mig_data in [upper_tri, lower_tri]:

        # G = create_graph(cities, normalized)    # method returns the graph
        G = create_graph(cities, mig_data)    # method returns the graph

        pos = nx.spring_layout(G, weight='weight')

        # nodes, locations = zip(*pos.items())

        # Find optimum k value for K
        optimum_k(np.array(list(pos.values())))

        set_no_of_clusters()

        # KMeans using Euclidean Distance
        n_dict = cluster_graph(positions=pos, cluster=KMeans())
        print(n_dict)
        draw_graph(graph=G, graph_dict=n_dict, pos=pos)

        # Kmeans using Manhatten Distance
        kmedian = KMedian.KMedians(k=clusters)
        n_dict1 = cluster_graph(positions=pos, cluster=kmedian)
        print(n_dict1)
        draw_graph(graph=G, graph_dict=n_dict, pos=pos)

        # Kmeans using Fuzzy with Euclidean
        fuzzykmeans = KMedian.FuzzyKMeans(k=clusters)
        n_dict1 = cluster_graph(positions=pos, cluster=fuzzykmeans)
        print(n_dict1)
        draw_graph(graph=G, graph_dict=n_dict, pos=pos)

        # # Kmeans using Fuzzy with Manhatten
        # fuzzykmeansM = KMedian.FuzzyKMeansM(k=cluster_size)
        # n_dict1 = cluster_graph(positions=pos, cluster=fuzzykmeansM)
        # print(n_dict)
        # draw_graph(graph=G, graph_dict=n_dict, pos=pos)

