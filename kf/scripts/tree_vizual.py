#!/usr/bin/python3

#%%
import igraph
from igraph import Graph, EdgeSeq
import plotly.graph_objects as go
import numpy as np
#import gen_tree_node as gtn


def MakeGraphFromNode(listNode):
    G = igraph.Graph()
    
    labels = []
    times = []
    
    for item in listNode:
        id = str(item['Id'])
        labels.append(id)
        times.append(item['Time'])
        G.add_vertex(id)

    distances = []
    for item in listNode:
        idParent = str(item['Id'])
        children = item['Children']
        if children is not None:
            for c in children:
                idChild = str(c['Id'])
                G.add_edge(idParent, idChild)
                distances.append(str(c['Distance']))

    additional_info = [f"TrackID={item['TrackID']}<br>Score={item['Score']}<br>ScoreMax={item['ScoreMax']}<br>DetID={item['DetID']}<br>Time={item['Time']}" for item in listNode]

    return G, distances, labels, times, additional_info

def MakeColorDiff(listNodeOld, listNodeNew):

    idOld = set([])
    for item in listNodeNew:
        idOld.add(item['Id'])

    colors = []
    for item in listNodeOld:
        if item['Id'] in idOld:
            if item['DetID'] == -1:
                colors.append('rgb(100, 100, 100)')
            else:
                colors.append('rgb(97, 117, 193)')
        else:
            if item['DetID'] == -1:
                colors.append('rgb(100, 100, 100)')
            else:
                colors.append('rgb(25, 25, 25)')            

    return colors


def PlotGraphFigure(G, distances, labels, times, additional_info, colors):
    
    lay = G.layout('rt')
    L = len(labels)

    position = {k: lay[k] for k in range(L)}
    Y = times
    M = max(Y)
    Xn = [position[k][0] for k in range(L)]
    Yn = [2 * M - Y[k] for k in range(L)]

    fig = go.Figure()

    es = EdgeSeq(G)
    E = [e.tuple for e in G.es]
    Xe = []
    Ye = []
    for edge in E:
        Xe += [position[edge[0]][0], position[edge[1]][0], None]
        Ye += [2 * M - Y[edge[0]], 2 * M - Y[edge[1]], None]

    # Метки рёбер
    edge_labels = [f'{edge[0]}-{edge[1]}' for edge in E]
    X_edge_labels = [(position[edge[0]][0] + position[edge[1]][0]) / 2 for edge in E]
    Y_edge_labels = [(2 * M - Y[edge[0]] + 2 * M - Y[edge[1]]) / 2 for edge in E]

    # Вычисление углов для рёбер
    angles = []
    for edge in E:
        dx = position[edge[1]][0] - position[edge[0]][0]
        dy = position[edge[1]][1] - position[edge[0]][1]
        angle = np.arctan2(dy, dx) * 180 / np.pi - 180  # Угол в градусах
        angles.append(angle)


    fig.add_trace(go.Scatter(x=Xe,
                            y=Ye,
                            mode='lines',
                            line=dict(color='rgb(210,210,210)', width=1),
                            hoverinfo='none'))

    fig.add_trace(go.Scatter(x=Xn,
                            y=Yn,
                            mode='markers',
                            name='Node',
                            marker=dict(symbol='circle-dot',
                                        size=18,
                                        color=colors,
                                        line=dict(color='rgb(50,50,50)', width=1)),
                            text=additional_info,
                            hoverinfo='text',
                            opacity=0.8))

    for i, label in enumerate(edge_labels):
        fig.add_annotation(
            x=X_edge_labels[i],
            y=Y_edge_labels[i],
            text=distances[i],
            showarrow=False,
            textangle=0, #angles[i],
            font=dict(size=10, color='rgb(50,50,50)')
        )

    for i, label in enumerate(labels):
        fig.add_annotation(
            x=Xn[i],
            y=Yn[i],
            text=label,
            showarrow=False,
            font=dict(size=10, color='rgb(255,255,255)'),
            align='center',
            xanchor='center',
            yanchor='middle',
            bgcolor='rgba(0,0,0,0)'
        ) 

    fig.show()

def PlotGraph(listNode, listNodeNew):
    print("PlotGraph")
    G, distances, labels, times, additional_info = MakeGraphFromNode(listNode)
    colors = MakeColorDiff(listNode, listNodeNew)
    PlotGraphFigure(G, distances, labels, times, additional_info, colors)

# def TestPrint():
#     print("TestPrint")

# TestPrint()

# root = gtn.Node(1.0, 20, 1, 31)

# child1 = gtn.Node(2.0, 200, 1, 20)
# child2 = gtn.Node(3.0, 100, 1, 30)
# child3  = gtn.Node(5.0, 0, 1, -1)

# child1.set_parent(root, 1.)
# child2.set_parent(root, 2.)
# child3.set_parent(child2, 10.)

# listNode = [vars(root), vars(child1), vars(child2), vars(child3)]
# listNodeRemove = [vars(child1), vars(child2), vars(child3)]

# PlotGraph(listNode, listNodeRemove)


#%%
