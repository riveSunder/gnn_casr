
"""
Functional simulation of Life-like CA on graphs
"""

import numpy as np

from sympy import lambdify
import sympy as sp
import pysr
pysr.install()
pysr.julia_helpers.init_julia()
from pysr import PySRRegressor

import torch

import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt

my_cmap = plt.get_cmap("magma")

from functools import reduce


# simple custom activation functions
gaussian = lambda x: torch.exp(- (x**2 / 0.5**2) / 2)  
# soft clip params from Chakazul ;)
soft_clip =lambda x: 1. / (1. + torch.exp(-4 * (x - 0.5))) 


def plot_compare(grid_0, grid_1, my_cmap=plt.get_cmap("magma"), titles=None, vmin=0.0, vmax=1):

    global subplot_0
    global subplot_1

    if type(grid_0) is torch.tensor:
        grid_0 = grid_0.detach().numpy()
    if type(grid_1) is torch.tensor:
        grid_1 = grid_1.detach().numpy()
        
    if titles == None:
        titles = ["CA grid time t", "Neighborhood", "Update", "CA grid time t+1"]

    fig = plt.figure(figsize=(12,6), facecolor="white")
    plt.subplot(121)
    subplot_0 = plt.imshow(grid_0, cmap=my_cmap, vmin=vmin, vmax=vmax, interpolation="nearest") 
    plt.title(titles[0], fontsize=18)

    plt.subplot(122)
    subplot_1 = plt.imshow(grid_1, cmap=my_cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    plt.title(titles[1], fontsize=18)

    plt.tight_layout()

    return fig 

# Graph NN CA functions are based on code from [here](https://github.com/riveSunder/SortaSota/tree/gh-pages/life_like_graphs)
def get_ca_mlp(birth=[3], survival=[2,3]):
    """ 
    return an MLP forward pass function encoding Life-like CA rules
    default to Conway's Game of Life (B3/S23)
    """ 

    wh = torch.ones(18,1)
    bh = -torch.arange(18).reshape(18,1)
    wy = torch.zeros(1,18)

    for bb in birth:
        wy[:, bb] = 1.0 

    for ss in survival:
        wy[:, ss+9] = 1.0 

    def mlp(x):

        hidden = gaussian(torch.matmul(wh, x) + bh)
        #out = np.round(np.dot(wy, hidden))
        out = soft_clip(torch.matmul(wy, hidden))

        return out 

    return mlp 

def get_graph_params():
    """ 
    return an MLP forward pass function encoding Life-like CA rules
    default to Conway's Game of Life (B3/S23)
    """ 
    bh = -torch.rand(18,1) * 18
    wy = torch.rand(1,18)

    return bh, wy 


def ca_graph(length):
        
    # nodes
    # number of nodes is the side of the grid, squared.
    num_nodes = length**2
    nodes = torch.zeros(num_nodes, 1)
    node_indices = np.arange(num_nodes)

    # edges
    num_edges = 8 * num_nodes
    edges = torch.zeros(num_edges, 1)
    
    # senders & receivers
    senders = np.vstack(\
            [node_indices - length -1, \
            node_indices - length, \
            node_indices - length + 1, \
            node_indices - 1, \
            node_indices + 1, \
            node_indices + length - 1, \
            node_indices + length, \
            node_indices + length + 1])
    sender = senders.T.reshape(-1)

    senders = (senders + length**2) % length**2
    receivers = np.repeat(node_indices, 8)

    return (num_nodes, num_edges, nodes, edges, senders, receivers)

def add_puffer(graph_tuple):
    # add puffer for rule B356/S23
    nodes = graph_tuple[2]
    length = int(np.sqrt(graph_tuple[0]))
   
    nodes[2, 0] = 1.0
    nodes[length, 0] = 1.0
    nodes[4 + length, 0] = 1.0

    nodes[2*length, 0] = 1.0
    nodes[1+3*length, 0] = 1.0
    nodes[4*length+2:4*length+4, 0] = 1.0
    nodes[4*length+5, 0] = 1.0
    nodes[5*length+4:5*length+6, 0] = 1.0

    return (graph_tuple[0], nodes, graph_tuple[2], \
            graph_tuple[3], graph_tuple[4], graph_tuple[5])

            
def add_glider(graph_tuple):
    # add Life reflex glider

    nodes = graph_tuple[2]
    length = int(np.sqrt(graph_tuple[0]))

    nodes[0, 0] = 1.0
    nodes[1, 0] = 1.0  
    nodes[2, 0] = 1.0
    nodes[2 + length, 0] = 1.0
    nodes[1 + 2 * length, 0] = 1.0

    return (graph_tuple[0], nodes, graph_tuple[2], \
            graph_tuple[3], graph_tuple[4], graph_tuple[5])


def params_mlp(x, bh, wy):

    wh = torch.ones(18,1)

    hidden = gaussian(torch.matmul(wh, x) + bh)
    out = soft_clip(torch.matmul(wy, hidden))

    return out 

def get_mlp_loss(x, tgt_mlp, bh, wy):

    tgt = tgt_mlp(x.T)
    pred = params_mlp(x.T, bh, wy)

    loss = np.abs((tgt - pred)**2).mean()

    return loss


def get_adjacency(graph_tuple):

    num_nodes = graph_tuple[0]
    num_edges = graph_tuple[1]
    length = int(np.sqrt(graph_tuple[0]))
    
    senders = graph_tuple[4]
    receivers = graph_tuple[5]

    adjacency_matrix = torch.zeros(num_nodes, num_nodes)
    
    for xx in range(receivers.shape[0]):

        for yy in senders[:, receivers[xx]]:
            adjacency_matrix[receivers[xx], yy] = 1.0

    return adjacency_matrix



def get_graph_grid(graph_tuple):

    length = int(np.sqrt(graph_tuple[0]))

    grid = np.zeros((length, length))
    
    for ii, node_state in enumerate(graph_tuple[2]):

        grid[ ii // length, ii % length] = node_state

    return grid

def edge_mlp(x, wch, whn):
    
    # x is a neighbor cell
    # wch is weights from cell to hidden
    # whn is weight from hidden to neighbor value
    h = torch.relu(torch.matmul(x, wch))
    n = torch.matmul(h, whn)
    
    return n

def get_neighbors(adjacency_matrix, x, wch, whn):
    
    num_cells = reduce(lambda a,b: a*b, x.shape)
    my_neighbors = torch.zeros(x.shape[0], x.shape[1])
    
    for adj_x in range(num_cells):
        for adj_y in range(num_cells):
            
            rec_x = adj_x // x.shape[1]
                        
            send_x = adj_y // x.shape[1] 
                        
            if adjacency_matrix[adj_x, adj_y]:
                my_neighbors[rec_x] += edge_mlp(x[send_x].reshape(1,1), wch, whn).squeeze()
                
    return my_neighbors

def full_gnn(adjacency_matrix, x, bh, wy, wch, whn):
    
    neighbors = get_neighbors(adjacency_matrix, x, wch, whn)
    
    #tgt_mlp(((a_matrix @ nodes_0) + 9 * nodes_0).T).T
    new_x = params_mlp((((neighbors) + 9 * x).T), bh, wy).T 
    
    return new_x 

def get_gnn_loss(adjacency_matrix, x, tgt_mlp, bh, wy, wch, whn):

    tgt = tgt_mlp(((adjacency_matrix @ x) + 9 * x).T).T
    pred = full_gnn(adjacency_matrix, x, bh, wy, wch, whn)

    loss = torch.abs((tgt - pred)**2).mean()

    return loss
               
