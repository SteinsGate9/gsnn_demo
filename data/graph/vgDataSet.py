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
                        att = att.strip().lower()
                        if att != 'id' and att != 'x' and att != 'w' and att != 'h' and att != 'y':
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
                        ##
                        name = name.rstrip().lower()
                        if name:
                            if name[-1] == 's':
                                name = name[:-1]
                        else:
                            continue
                        if name[:2] == 'a ':
                            name = name[2:]
                        if name == 'bu':
                            name = 'bus'
                        if name == 'ski':
                            name = 'skis'
                        if name == 'wine glasse':
                            name = 'wine glass'
                        if name == 'scissor':
                            name = 'scissors'

                        if name == 'carrots':
                            name = 'carrot'
                        if name == 'hotdog':
                            name = 'hot dog'
                        if name == 'plants' or name == 'plant':
                            name = 'potted plant'
                        if name == 'table':
                            name = 'dining table'
                        if name == 'television':
                            name = 'tv'
                        if name == 'bag':
                            name = 'handbag'
                        if name == 'glove' or name == 'gloves':
                            name = 'baseball glove'
                        if name == 'bat':
                            name = 'baseball bat'
                        if name == 'ball':
                            name = 'sports ball'


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
                        att = att.strip().lower()
                        if att != 'id' and att != 'x' and att != 'w' and att != 'h' and att != 'y':
                            d[str(att)] = dict[str(att)] + 1
                    except KeyError as e:
                        d[str(att)] = 1

        assert "x" not in d.items() and 'id' not in d.items() and 'w' not in d.items() and 'h' not in d.items() and 'y' not in d.items()
        d = collections.OrderedDict(sorted(d.items(), key=lambda dict: dict[1], reverse=True))
        list = []
        a = 0
        for i,v in d.items():
            if a < 100:
                list.append(i)
            a = a + 1
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
                        name = name.rstrip().lower()
                        if name:
                            if name[-1] == 's':
                                name = name[:-1]
                        else:
                            continue
                        if name[:2] == 'a ':
                            name = name[2:]
                        if name == 'bu':
                            name = 'bus'
                        if name == 'ski':
                            name = 'skis'
                        if name == 'wine glasse':
                            name = 'wine glass'
                        if name == 'scissor':
                            name = 'scissors'

                        if name == 'carrots':
                            name = 'carrot'
                        if name == 'hotdog':
                            name = 'hot dog'
                        if name == 'plants' or name == 'plant':
                            name = 'potted plant'
                        if name == 'table':
                            name = 'dining table'
                        if name == 'television':
                            name = 'tv'
                        if name == 'bag':
                            name = 'handbag'
                        if name == 'glove' or name == 'gloves':
                            name = 'baseball glove'
                        if name == 'bat':
                            name = 'baseball bat'
                        if name == 'ball':
                            name = 'sports ball'
                        try:
                            d[str(name)] = d[str(name)] + 1
                        except KeyError as e:
                            d[str(name)] = 1

        d = collections.OrderedDict(sorted(d.items(), key=lambda dict: dict[1], reverse=True))
        d2 = collections.OrderedDict(sorted(d.items(), key=lambda dict: dict[1], reverse=False))
        print(d2)
        a = 0
        for i, v in d.items():
            if a < 200:
                list.append(i)
            a = a + 1
        assert len(list) == 300
        from pycocotools.coco import COCO
        dataDir = '../coco'
        dataType = 'train2014'
        annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
        coco = COCO(annFile)
        cats = coco.loadCats(coco.getCatIds())
        nms = [cat['name'] for cat in cats]

        for i in nms:
            if i not in list:
                list.append(i)

        json.dump(list,open('labels' + '.json', 'w'), indent=2)
        print(len(list))

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

    def add_edge_weight(self, edge, type, weight=1):
        x,y = edge
        try:
            if type == 'att':
                self[x][y]['weight'] =  self[x][y]['weight'] + 1
            elif type == 'obj':
                self[x][y]['weight'] = self[x][y]['weight'] + 1
                self[y][x]['weight'] = self[y][x]['weight'] + 1
        except KeyError as e:
            if type == 'att':
                self.add_edge(x,y,weight=weight)
            elif type == 'obj':
                self.add_edge(x, y, weight=weight)
                self.add_edge(y, x, weight=weight)
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
                    name_list = dict["object_names"]
                    attri_list = dict["attributes"]
                    for name in name_list:
                        name = name.rstrip().lower()

                        if name:
                            if name[-1] == 's':
                                name = name[:-1]
                        else:
                            continue
                        if name[:2] == 'a ':
                            name = name[2:]
                        if name == 'bu':
                            name = 'bus'
                        if name == 'ski':
                            name = 'skis'
                        if name == 'wine glasse':
                            name = 'wine glass'
                        if name == 'scissor':
                            name = 'scissors'
                        if name == 'carrots':
                            name = 'carrot'
                        if name == 'hotdog':
                            name = 'hot dog'
                        if name == 'plants' or name == 'plant':
                            name = 'potted plant'
                        if name == 'table':
                            name = 'dining table'
                        if name == 'television':
                            name = 'tv'
                        if name == 'bag':
                            name = 'handbag'
                        if name == 'glove' or name == 'gloves':
                            name = 'baseball glove'
                        if name == 'bat':
                            name = 'baseball bat'
                        if name == 'ball':
                            name = 'sports ball'

                        for att in attri_list:
                            att = att.strip().lower()
                            if att and att != 'id' and att != 'x' and att != 'w' and att != 'h' and att != 'y':
                                self.add_edge_weight((name, att) ,'att')

    def add_relationships(self, data_path):
        with open(data_path) as f:
            from tqdm import tqdm
            import json
            load_list = json.load(f)
            pbar = tqdm(load_list)
            for dict in pbar:
                listofdict = dict["relationships"]
                for dict in listofdict:
                    o = dict["object"]["name"]
                    s = dict["subject"]["name"]
                    o = o.rstrip().lower()
                    if o:
                        if o[-1] == 's':
                            o = o[:-1]
                    else:
                        continue
                    if o[:2] == 'a ':
                        o = o[2:]
                    if o == 'bu':
                        o = 'bus'
                    if o == 'ski':
                        o = 'skis'
                    if o == 'wine glasse':
                        o = 'wine glass'
                    if o == 'scissor':
                        o = 'scissors'
                    if o == 'carrots':
                        o = 'carrot'
                    if o == 'hotdog':
                        o = 'hot dog'
                    if o == 'plants' or o == 'plant':
                        o = 'potted plant'
                    if o == 'table':
                        o = 'dining table'
                    if o == 'television':
                        o = 'tv'
                    if o == 'bag':
                        o = 'handbag'
                    if o == 'glove' or o == 'gloves':
                        o = 'baseball glove'
                    if o == 'bat':
                        o = 'baseball bat'
                    if o == 'ball' or 'soccer ball' or 'soccer':
                        o = 'sports ball'

                    s = s.rstrip().lower()

                    if s:
                        if s[-1] == 's':
                            s = s[:-1]
                    else:
                        continue
                    if s[:2] == 'a ':
                        s = s[2:]
                    if s == 'bu':
                        s = 'bus'
                    if s == 'ski':
                        s = 'skis'
                    if s == 'wine glasse':
                        s = 'wine glass'
                    if s == 'scissor':
                        s = 'scissors'
                    if s == 'carrots':
                        s = 'carrot'
                    if s == 'hotdog':
                        s = 'hot dog'
                    if s == 'plants' or s == 'plant':
                        s = 'potted plant'
                    if s == 'table':
                        s = 'dining table'
                    if s == 'television':
                        s = 'tv'
                    if s == 'bag':
                        s = 'handbag'
                    if s == 'glove' or s == 'gloves':
                        s = 'baseball glove'
                    if s == 'bat':
                        s = 'baseball bat'
                    if s == 'ball':
                        s = 'sports ball'
                    self.add_edge_weight((o, s),'obj')

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
if __name__ == "__main__":
    relationship = 'relationships.json'
    attributes = 'attributes.json'
    objects = 'objects.json'
    #

    # LoadLabels(attributes,objects)
    # graph = Graph(nx.Graph())
    # graph.add_relationships(relationship)
    # graph.add_attributes(attributes)
    # graph.filter_graph(150)
    # graph.save_graph_tojson('filtered_graph_v2')
    # LoadLabelForImage(attributes,objects)


    
    true = np.asarray(['___background__', u'person', u'bicycle', u'car', u'motorcycle', u'airplane', u'bus', u'train', u'truck', u'boat', u'traffic light', u'fire hydrant', u'stop sign', u'parking meter', u'bench', u'bird', u'cat', u'dog', u'horse', u'sheep', u'cow', u'elephant', u'bear', u'zebra', u'giraffe', u'backpack', u'umbrella', u'handbag', u'tie', u'suitcase', u'frisbee', u'skis', u'snowboard', u'sports ball', u'kite', u'baseball bat', u'baseball glove', u'skateboard', u'surfboard', u'tennis racket', u'bottle', u'wine glass', u'cup', u'fork', u'knife', u'spoon', u'bowl', u'banana', u'apple', u'sandwich', u'orange', u'broccoli', u'carrot', u'hot dog', u'pizza', u'donut', u'cake', u'chair', u'couch', u'potted plant', u'bed', u'dining table', u'toilet', u'tv', u'laptop', u'mouse', u'remote', u'keyboard', u'cell phone', u'microwave', u'oven', u'toaster', u'sink', u'refrigerator', u'book', u'clock', u'vase', u'scissors', u'teddy bear', u'hair drier', u'toothbrush'])
  

    g = GraphLoader('filtered_graph_v2.json')
    n = g.nodes()
    print(n)
    # true = list((
    #             '__background__', u'person', u'bicycle', u'car', u'motorcycle', u'airplane', u'bus', u'train', u'truck',
    #             u'boat', u'traffic light', u'fire hydrant', u'stop sign', u'parking meter', u'bench', u'bird', u'cat',
    #             u'dog', u'horse', u'sheep', u'cow', u'elephant', u'bear', u'zebra', u'giraffe', u'backpack',
    #             u'umbrella', u'handbag', u'tie', u'suitcase', u'frisbee', u'skis', u'snowboard', u'sports ball',
    #             u'kite', u'baseball bat', u'baseball glove', u'skateboard', u'surfboard', u'tennis racket', u'bottle',
    #             u'wine glass', u'cup', u'fork', u'knife', u'spoon', u'bowl', u'banana', u'apple', u'sandwich',
    #             u'orange', u'broccoli', u'carrot', u'hot dog', u'pizza', u'donut', u'cake', u'chair', u'couch',
    #             u'potted plant', u'bed', u'dining table', u'toilet', u'tv', u'laptop', u'mouse', u'remote', u'keyboard',
    #             u'cell phone', u'microwave', u'oven', u'toaster', u'sink', u'refrigerator', u'book', u'clock', u'vase',
    #             u'scissors', u'teddy bear', u'hair drier', u'toothbrush'))
    #
    # #
    # lisnotinnodes = list(filter(lambda x:x not in n,lis))
    # truenotinlist = list(filter(lambda x: x not in lis, true))
    truenotingraph = list(filter(lambda x: x not in n, true))
    # print([i if i in  true else None for i in lisnotinnodes])
    # print(truenotinlist)
    print(truenotingraph)
    # print(list(sorted(n)))
    # g.print_info()
    # lis = json.load(open('labels_dict.json','r'))
    # print(len(lis.keys()), len(lis['1']))


