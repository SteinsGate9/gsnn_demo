# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 20:39:07 2018

@author: hsc
"""
import json
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
from math import ceil
from networkx.readwrite import json_graph
def LoadLabelForImage(attripath, objpath):
    d = {}
    with open(attripath) as f:
        from tqdm import tqdm
        import collections
        import json
        load_list = json.load(f)
        pbar = tqdm(load_list)
        d = {}
        for dic in pbar:
            listofdict = dic["attributes"]
            id = dic["id"]
            for dict in listofdict:
                name = dict["object_names"][0]
                attri_list = dict["attributes"]
                for att in attri_list:
                    try:
                        d[id].append(att)
                    except KeyError as e:
                        d[id] = [att]
            try:
                d[id] = list(set(d[id]))
            except KeyError as e:
                d[id] = []

        with open(objpath) as f:
            load_list = json.load(f)
            pbar = tqdm(load_list)
            for dic in pbar:
                listofdict = dic["objects"]
                id = dic["id"]
                for dict in listofdict:
                    names = dict["names"]
                    for name in names:
                        try:
                            d[id].append(name)
                        except KeyError as e:
                            d[id] = [name]
                try:
                    d[id] = list(set(d[id]))
                except KeyError as e:
                    d[id] = []

        lis = json.load(open('labels.json','r'))
        print(len(d.keys))
        pbar = tqdm(d.keys())
        for i in pbar:
            a = np.zeros(len(lis))
            for name in d[i]:
                try:
                    a[lis.index(name)] = 1
                except:
                    pass
            d[i] = a.tolist()
        json.dump(d,open('labels_dict' + '.json', 'w'), indent=2)

def LoadLabels(attripath, objpath):
    with open(attripath) as f:
        from tqdm import tqdm
        import collections
        import json
        load_list = json.load(f)
        pbar = tqdm(load_list)
        d = {}
        for dic in pbar:
            listofdict = dic["attributes"]
            for dict in listofdict:
                name = dict["object_names"][0]
                attri_list = dict["attributes"]
                for att in attri_list:
                    try:
                        d[str(att)] = dict[str(att)] + 1
                    except KeyError as e:
                        d[str(att)] = 1
        print(d.items())
        print(sorted(d.items(), key=lambda dict: dict[1], reverse=True))
        d = collections.OrderedDict(sorted(d.items(), key=lambda dict: dict[1], reverse=True))
        list = []
        a = 0
        for i,v in d.items():
            if a < 100:
                list.append(i)
            a = a+1
        assert len(list) == 100
        with open(objpath) as f:
            load_list = json.load(f)
            pbar = tqdm(load_list)
            d = {}
            for dic in pbar:
                listofdict = dic["objects"]
                for dict in listofdict:
                    names = dict["names"]
                    for name in names:
                        try:
                            d[str(name)] = d[str(name)] + 1
                        except KeyError as e:
                            d[str(name)] = 1
        d = collections.OrderedDict(sorted(d.items(), key=lambda dict: dict[1], reverse=True))
        a = 0
        for i, v in d.items():
            if a < 200:
                list.append(i)
            a = a + 1
        assert len(list) == 300
        from pycocotools.coco import COCO
        dataDir = '../ms_coco_formatter'
        dataType = 'train2014'
        annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
        coco = COCO(annFile)
        cats = coco.loadCats(coco.getCatIds())
        nms = [cat['name'] for cat in cats]
        a = 0
        for i in nms:
            if i not in list and a < 16:
                list.append(i)
                a = a+1
        assert len(list) == 316
        json.dump(list,open('labels' + '.json', 'w'), indent=2)

def GraphLoader(datapath):
    graph = Graph(nx.Graph())
    graph.load_graph_fromjson(datapath)
    return graph

class Graph(nx.Graph):
    def __init__(self,
                 graph=None,
                 adj=None,
                 node_tags=None,
                 features=None,
                 label=None
                 ):
        self.node_num = None
        self.edge_num = None
        self.node_tages = node_tags
        self.features = features
        self.label = label

        # build graph
        assert graph is not None or adj is not None

        # build graph by nx.graph
        if graph is not None:
            self.node_num = len(graph.nodes())
            self.edge_num = len(graph.edges())
            super(Graph, self).__init__(graph)

        # build graph by adj
        if (adj is not None) and (graph is None):
            edges = np.where(adj > 0)
            x = edges[0]
            y = edges[1]
            edge_list = np.ndarray(shape=(x.size, 2), dtype=int)
            edge_list[:, 0] = x
            edge_list[:, 1] = y
            nx_graph = nx.Graph()
            nx_graph.add_edges_from(list(edge_list))
            self.edge_num = len(nx_graph.edges())
            self.node_num = len(nx_graph.nodes())
            super(Graph, self).__init__(nx_graph)

    def get_edge_pairs(self):
        # create edge pairs
        x, y = zip(*self.edges())
        edge_pairs = np.ndarray(shape=(self.edge_num, 2), dtype=int)
        edge_pairs[:, 0] = x
        edge_pairs[:, 1] = y
        edge_pairs = edge_pairs.flatten()
        return edge_pairs

    def get_adj(self):
        return nx.adjacency_matrix(self).toarray()

    def get_adj_nx(self):
        return nx.adjacency_matrix(self)

    def get_adj_sp(self):
        x, y = zip(*self.edges())
        return sp.coo_matrix((np.ones(len(self.edges())), (x, y)),
                                shape=(len(self.nodes()), len(self.nodes())))

    def get_normalized_adj(self):
        pass

    def add_edge_weight(self, edge, weight=1):
        x,y = edge
        try:
            self[x][y]['weight'] =  self[x][y]['weight'] + 1
        except KeyError as e:
            self.add_edge(x,y,weight=weight)
        return

    def save_graph_tojson(self, save_path):
        json.dump(dict(nodes=[n for n in self.nodes()],
                       edges=[(u, v, self[u][v]) for u, v in self.edges()]),
                  open(save_path+'.json', 'w'), indent=2)

    def load_graph_fromjson(self, save_path):
        G = nx.DiGraph()
        d = json.load(open(save_path))
        G.add_nodes_from(d['nodes'])
        G.add_edges_from(d['edges'])
        super(Graph, self).__init__(G)

    def plot_graph(self):
        plt.subplots(ceil(1), 1, figsize=(15, 3))
        nx.draw(self, with_labels=True, font_weight='bold')
        plt.title("plot")
        plt.axis('on')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def add_attributes(self, data_path):
        with open(data_path) as f:
            from tqdm import tqdm
            import json
            load_list = json.load(f)
            pbar = tqdm(load_list)
            for dict in pbar:
                listofdict = dict["attributes"]
                for dict in listofdict:
                    name = dict["object_names"][0]
                    attri_list = dict["attributes"]
                    for att in attri_list:
                        self.add_edge_weight((name, att))

    def add_relationships(self, data_path):
        with open(data_path) as f:
            from tqdm import tqdm
            import json
            load_list = json.load(f)
            pbar = tqdm(load_list)
            for dict in pbar:
                listofdict = dict["relationships"]
                for dict in listofdict:
                    obj = dict["object"]["name"]
                    sub = dict["subject"]["name"]
                    self.add_edge_weight((obj, sub))

    def filter_graph(self, filter_threshold):
        edges = self.edges()
        new_graph = Graph(nx.Graph())
        from tqdm import tqdm
        for edge in tqdm(edges):
            x = edge[0]
            y = edge[1]
            if self[x][y]['weight'] >= filter_threshold:
                new_graph.add_edge(x, y)
        super(Graph, self).__init__(new_graph)

    def print_info(self, verbose=3):
        assert verbose >= 0 
        print("This graph has %d nodes",len(self.nodes()))
        print("This graph has %d edges",len(self.edges()))
        ver = 0
        for (u, v, wt) in self.edges.data():
            if(ver == verbose):
                break
            print("Example: " +  ' From ' + str(u) + ' To '  + str(v))
            ver = ver + 1

    # filter
    def filter_nodes(self, nodelist):
        sub_graph = Graph(nx.Graph())
        nodes = self.nodes()
        for a in nodelist:
            for b in nodes:
                try:
                    edge = self[a][b]
                    sub_graph.add_edge(edge)
                except KeyError as e:
                    pass
        for a in nodes:
            for b in nodelist:
                try:
                    edge = self[a][b]
                    sub_graph.add_edge(edge)
                except KeyError as e:
                    pass
        return sub_graph
        
    def add_wordnet(self):
        from nltk import word_tokenize as wn
        wn.synset('walk.v.01').entailments()
if __name__ == "main":
    relationship = 'relationships.json'
    attributes = 'attributes.json'
    objects = 'objects.json'
    lis = json.load(open('labels_dict.json','r'))
    print(len(lis.keys()), len(lis['1']))


