# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 18:52:15 2017

@author: umut
"""
import random
import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from sklearn.cluster import KMeans
import positionbased
import KMedian
from sklearn.preprocessing import normalize


def draw_weighted_graph(graph=nx.Graph(), graph_dict={}, pos={}):
    colors = []
    for key in graph_dict.keys():
        r = random.randint(0, 100) / 100
        g = random.randint(0, 100) / 100
        b = random.randint(0, 100) / 100
        colors.append([r, g, b])
        lst = list(graph_dict[key])
        print(lst)
        nx.draw(graph, pos=pos, nodelist=list(graph_dict[key]), with_labels=True, node_color=[[r, g, b]])
    plt.show()


def coordinate_generator(positions={}, dc={}):
    vert = 1
    hort = 1
    for cities in positions.values():
        for city in cities:
            x, y = dc[city]
            loc_x = random.randint(-100, 100) / 50
            loc_y = random.randint(-100, 100) / 50
            # dc[city] = [x*10*math.sin(x), y*vert*loc_y]
            dc1[city] = [x * 10 * math.sin(10 * x) + loc_x * vert, x * 10 * math.cos(10 * x) + loc_y * hort]
            hort *= vert
            vert *= -1
    return dc1

if __name__ == '__main__':

    # input_data = pd.read_csv('data/sample.csv', index_col=0)  # import data
    input_data = pd.read_csv('data/mig data2008.csv', index_col=0)  # import data
    values = input_data.values
    values = normalize(values)
    upper_tri = np.triu(values)
    lower_tri = np.tril(values)
    cities = input_data.axes[0].values

    for mig_data in [upper_tri, lower_tri]:
        G = positionbased.create_graph(cities, values)
        # G = nx.Graph(values)  # creates multi directed graphs
        # G = nx.MultiDiGraph(values)  # creates multi directed graphs

        # nx.strongly_connected_components(G)
        coefficients = nx.clustering(G)    # for indrected graphs with this function i can gather the clusters info but for directed graphs
        # val = minmax_scale(list(coefficients.values()), feature_range=(-1, 1))
        dc = {}
        for key in coefficients.keys():
            dc[key] = [coefficients[key], coefficients[key]]

        print(dc)

        # svr = KMeans()
        # positionbased.optimum_k(np.array(list(dc.values())))

        positionbased.set_no_of_clusters()

        dc1 = {}

        pos = positionbased.cluster_graph(dc)
        print("KMeans using Euclidean Distance")
        print(pos)
        dc1 = coordinate_generator(positions=pos, dc=dc)
        # positionbased.draw_graph(G, graph_dict=pos, pos=dc1)
        # lst = nx.nodes(G)
        # plt.show()

        clusters = positionbased.clusters

        # Kmeans using Manhatten Distance
        kmedian = KMedian.KMedians(k=clusters)
        pos = positionbased.cluster_graph(positions=dc, cluster=kmedian)
        print("KMeans using Manhatten Distance")
        print(pos)
        dc1 = coordinate_generator(positions=pos, dc=dc)
        # positionbased.draw_graph(graph=G, graph_dict=pos, pos=dc1)

        # Kmeans using Fuzzy with Euclidean
        fuzzykmeans = KMedian.FuzzyKMeans(k=clusters)
        pos = positionbased.cluster_graph(positions=dc, cluster=fuzzykmeans)
        print("FuzzyKMeans")
        print(pos)
        dc1 = coordinate_generator(positions=pos, dc=dc)
        # positionbased.draw_graph(graph=G, graph_dict=pos, pos=dc1)

