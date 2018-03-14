import numpy as np


def bellman_ford(n, edges, weights):
    """
    Implements the Bellman-Ford algorithm.
    
    :param n: number of edges.
    :param edges: list of edges. Every element of this list is of the form [i,j], where 0 <= i, j <= n-1, which
                              indicates that there is an edge between nodes i and j.
    :param weights: list of weights. weights[j] is associated to edges[j].
    """
    source = 0 # Initialize vertex source arbitrarily.
    distances = np.zeros(n)
    for v in range(1,n):
        distances[v] = np.inf
    
    # Relaxation stage   
    for i in range(n-1):
        for j in range(len(edges)):
            if distances[edges[j][1]] > distances[edges[j][0]] +  weights[j]:
                distances[edges[j][1]] = distances[edges[j][0]] +  weights[j]
    
    # Verification stage
    for j in range(len(edges)):
        if distances[edges[j][1]] > distances[edges[j][0]] +  weights[j]:
            print('A negative cycle was found.')
            return True
    
    print('There are no negative cycles.')
    return False