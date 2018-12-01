# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 19:28:05 2018

@author: hsc
"""
import json
import torch
import numpy as np
import torch.nn as nn
from functools import reduce
import operator
import torch.nn.functional as F
import torchvision.models as models
from data.graph.vgDataSet import *

def adjust_learning_rate(optimizer, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']

class finetune(torch.nn.Module):
    """vgg16 before fc7"""
    def __init__(self,vgg):
        super(finetune, self).__init__()
        self.features = vgg.features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout()
            )
        # for p in self.parameters():
        #     p.requires_grad = Fal
        self.finetune = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.finetune(x)
        return x




class PipeLine(torch.nn.Module):
    def __init__(self, opt, dataloader):
        # set opt
        self.opt = opt
        self.dataloader = dataloader
        super(PipeLine, self).__init__()

        # set vgg 16
        vgg16 = models.vgg16(pretrained=False)
        vgg16.load_state_dict(torch.load('/home/nesa320/huangshicheng/gitforwork/my_gsnn/models/vgg16-397923af.pth'))
        pretrained_dict = vgg16.state_dict()
        self.vgg16 = finetune(vgg16)
        model_dict = self.vgg16.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.vgg16.load_state_dict(model_dict)

        # set gsnn
        self.gsnn = GSNN(opt)

        # set output layer
        self.output = nn.Sequential(
            nn.Linear((4096+80+321*5),
                self.opt.vg_objects+self.opt.vg_attributes+self.opt.coco_cats),
            nn.Dropout(),
            nn.Sigmoid())

    def forward(self, image, concat, annotation):
        """forward pipeline"""
        # send in data
        gsnn_data = self.gsnn(annotation)
        vgg16_data = self.vgg16(image)
        frcnn_data = concat
        print(gsnn_data.shape, vgg16_data.shape, frcnn_data.shape)
        concat = torch.cat([gsnn_data, vgg16_data, frcnn_data], 1)

        # output
        output = self.output(concat)
        return  output

    def _train(self):
        """train this fucker"""
        self.train()

        # cuda
        if self.opt.cuda:
            self.cuda()
        i = 0

        # set optimizers
        vgg16_features_params = list(map(id, self.vgg16.features.parameters()))
        vgg16_classifier_params = list(map(id, self.vgg16.classifier.parameters()))
        gsnn_params = list(map(id, self.gsnn.parameters()))
        other_params = filter(lambda p: id(p) not in vgg16_classifier_params + vgg16_features_params + gsnn_params,
                              self.parameters())
        optimizer_adam = torch.optim.Adam([
            {'params': self.gsnn.parameters(), "lr": self.opt.lr,"weight_decay": self.opt.weight_decay}
        ])
        optimizer_sgd = torch.optim.SGD(
            [{'params': other_params},
             {'params': self.vgg16.features.parameters(), "lr": 0.005},
             {'params': self.vgg16.classifier.parameters(),"lr": 0.005}],
            lr=self.opt.lr,
            momentum=self.opt.momentum,
            weight_decay=self.opt.weight_decay)

        # begin training
        from tqdm import tqdm
        for epoch in range(self.opt.epochs):
            # decay lr
            if epoch % (self.opt.lr_decay_step + 1) == 0:
                adjust_learning_rate(optimizer_adam, self.opt.lr_decay_rate)
                adjust_learning_rate(optimizer_sgd, self.opt.lr_decay_rate)

            # begin
            pbar = tqdm(self.dataloader)
            for image, label, concat, annotation in pbar:
                # cuda
                if self.opt.cuda:
                    image = image.cuda()
                    annotation = annotation.cuda()
                    concat = concat.cuda()
                    label = label.cuda()

                # grad
                image.requires_grad_()
                annotation.requires_grad_()
                concat.requires_grad_()

                # get result
                output = self.forward(image, concat, annotation)
                loss = F.binary_cross_entropy(output, label)

                # set optimizers
                optimizer_sgd.zero_grad()
                optimizer_adam.zero_grad()
                loss.backward()

                # optimizers
                optimizer_adam.step()
                optimizer_sgd.step()

                # verbose
                i = i + 1
                if self.opt.verbose and i%len(pbar) == 500:
                    print('[%d/%d][%d/%d] Loss: %.4f' % (epoch, self.opt.epochs, i, len(dataloader), loss.data[0]))

        return




import torch
import torch.nn as nn
from libs.utils import AttrProxy



class Propogator(nn.Module):
    """
    GGNN dense matrix
    """
    def __init__(self, state_dim, node_num, edge_type_num):
        super(Propogator, self).__init__()

        self.node_num = node_num
        self.edge_type_num = edge_type_num

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.Sigmoid()
        )
        self.old_gate = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.Tanh()
        )


    def forward(self, state_in, state_cur, A):
        # [n 2n]
        A_in = A[:, :, :self.node_num*self.edge_type_num]
        # [n,H]
        a_in = torch.bmm(A_in, state_in)
        #
        print(A_in.shape, state_in.shape)
        #state_in = torch.view(1, ,)
        #A_in = A[]

        #
        # [n, 3H]
        a = torch.cat((a_in, state_cur),2)
        # [n, H]
        r = self.reset_gate(a)
        # [n, H]
        z = self.update_gate(a)
        # [n.3H]
        for_h_hat = torch.cat((a_in, r*state_cur),2)
        h_hat = self.old_gate(for_h_hat)
        # [n, H]
        output = (1 - z) * state_cur + z * h_hat
        return output


class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, opt):
        super(GGNN, self).__init__()

        self.state_dim = opt.state_dim
        self.annotation_dim = opt.annotation_dim
        self.edge_type_num = opt.edge_type_num
        self.node_num = opt.node_num
        self.n_steps = opt.n_steps
        self.graph = GraphLoader(opt.graph_dir)
        self.A = self.graph.get_adj()
        for i in range(self.edge_type_num):
            # incoming and outgoing edge embedding
            in_fc = nn.Linear(self.state_dim, self.state_dim)
            self.add_module("in_{}".format(i), in_fc)

        self.in_fcs = AttrProxy(self, "in_")

        # Propogation Model
        self.propogator = Propogator(self.state_dim, self.node_num, self.edge_type_num)

        # Output Model
        # Question here, why there's no sigmoid?
        self.out = nn.Sequential(
            nn.Linear(self.state_dim + self.annotation_dim, self.state_dim),
            nn.Tanh(),
            nn.Linear(self.state_dim, 1)
        )

        self._initialization()


    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)


  #  net(init_input, annotation, adj_matrix)
    def forward(self, prop_state, annotation, node_num):
            # init
            in_states = []
            out_states = []
            self.node_num = node_num

            # propogate
            for i in range(self.edge_type_num):
                # embedding for different edge_types
                # the reason separating in and out is that
                # to concat them later, basically a trick.
                in_states.append(self.in_fcs[i](prop_state))
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.view(-1, self.node_num*self.edge_type_num, self.state_dim)
            prop = self.propogator(in_states, prop_state, self.A)

            # output
            #join_state = torch.cat((prop_state, annotation), 2)
            #output = self.out(join_state)
            return prop

    def batch_forward(self, prop_state, annotation, node_num):
        pass
class GSNN(torch.nn.Module):

    def __init__(self, opt):
        super(GSNN, self).__init__()
        self.opt = opt
        self.state_dim = opt.state_dim
        self.annotation_dim = opt.annotation_dim
        self.edge_type_num = opt.edge_type_num
        self.node_num = opt.node_num
        self.n_steps = opt.n_steps
        self.important_loss = None

        # set graph
        self.graph =  GraphLoader(opt.graph_dir)
        self.adj = self.graph.get_adj()
        self.labels = json.load(open(self.opt.label2_dir, 'r'))
        self.n = list(self.graph.nodes)
        self.label_mask = list([self.n.index(i) if i in self.n else -1 for i in self.labels])
        self.pure_label_mask = set(self.label_mask)
        self.pure_label_mask.remove(-1)

        # variables
        self.node_bias = torch.zeros(opt.node_num).float().unsqueeze(0).unsqueeze(2).requires_grad_()
        if self.opt.cuda:
            self.node_bias = self.node_bias.cuda()
        self.important_net = nn.Linear(self.state_dim+self.annotation_dim, 1)
        self.propogation_net = GGNN(opt=opt)
        self.output_net = nn.Linear(self.state_dim+self.annotation_dim+1, self.state_dim)

        # init
        self._initialize()


    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)


    def _initialize_importantnet(self, annotation, adj):
        anno = torch.where(annotation > 0, torch.full_like(annotation, 1), annotation)
        cur_batch = np.tile(np.arange(adj.shape[0]), annotation.shape[0]).reshape(annotation.shape[0],-1)[(anno.data.cpu().numpy() != 0)]
        # batch trans
        for i in range(2):
            old_batch = cur_batch
            cur_batch = get_adj_nodes_batch(adj, cur_batch)
            cur_add_batch = list([set(cur_batch[i]) - set(old_batch[i]) for i in range(len(cur_add_batch))])
            cur_add_batch_new = list([i,j] for j in cur_add_batch[i] for i in range(len(cur_add_batch)))
            reduce(operator.add, cur_add_batch_new)
            anno[cur_add_batch_new] = self.opt.importance_factor**(i+1)

        return anno

    def forward(self, annotation):

        # 假设annotation是1000多维的
        anno = self._initialize_importantnet(annotation, self.graph.get_adj()).unsqueeze(2)
        anno_copy = anno.clone()
        padding = torch.zeros(len(anno), self.opt.node_num, self.opt.state_dim - self.opt.annotation_dim).float()
        if self.opt.cuda:
            padding = padding.cuda()
        init_input = torch.cat((anno, padding), 2)

        # other stuff
        important = None
        current_nodes = None

        # set up adj
        sub_graph, current_nodes_expand, current = SetUpAdj(self.adj, important, current_nodes, anno)
        anno = anno[:, current_nodes_expand, :]
        prop_state = init_input[:, current_nodes_expand, :]
        self.propogation_net.A = torch.FloatTensor(sub_graph).unsqueeze(0)
        self.propogation_net.A = torch.cat((self.propogation_net.A,self.propogation_net.A),dim=2)

        for i_step in range(self.n_steps):
            # padding prop_state
            output = self.propogation_net.forward(prop_state, anno, len(current_nodes_expand))
            init_input[:,current_nodes_expand,:] = prop_state
            forimportant = torch.cat((output, anno),2)
            important  = self.important_net(forimportant).squeeze(0).squeeze(1)

            # update adj# 取important 向量除了已有的前5
            sub_graph, current_nodes_expand, current = SetUpAdj(self.adj, important, current, anno)

            # update state
            anno = anno_copy[:, current_nodes_expand, :]
            prop_state = init_input[:, current_nodes_expand, :]
            self.propogation_net.A = torch.FloatTensor(sub_graph).unsqueeze(0)
            self.propogation_net.A = torch.cat((self.propogation_net.A, self.propogation_net.A), dim=2)

        # output net
        output = self.propogation_net.forward(prop_state, anno, len(current_nodes_expand))
        forimportant = torch.cat((output, anno), 2)

        # masked fuck
        mask = list(set([current_nodes_expand.index(i) if i in current else -1 for i in current_nodes_expand]))
        mask.remove(-1)
        forimportant = forimportant[:, mask, :]
        node_bias = self.node_bias[:, current, :]
        foroutput = torch.cat((forimportant, node_bias), 2)
        output = self.output_net(foroutput)

        # padding and stretching -> 323x5
        assert len(current) <= self.opt.label_len
        target = torch.zeros(self.opt.batch_size, self.opt.label_len, self.state_dim)

        # decode label
        mask = list(set([i if self.label_mask[i] in current else -1 for i in range(self.opt.label_len)]))
        mask.remove(-1)
        mask2 = list(set([current.index(i) if i in self.pure_label_mask else -1 for i in current]))
        mask2.remove(-1)

        # output
        target[:, mask, :] = output[:, mask2, :]
        output = target
        assert output.shape[1] == self.opt.label_len
        output = output.view(self.opt.batch_size,self.opt.label_len*self.state_dim,-1).squeeze(2)
        assert output.shape[0] == self.opt.batch_size
        return output

def SetUpAdj(graph, important_list, current_nodes_list, annotation_list):
    # set subgraph and expanded_nodes
    if current_nodes_list is None:
        for i in len(annotation_list)
        # 16 z 1
        mask = annotation.data.cpu().numpy() == 1
        # 16 z 1
        current = np.arange(0,graph.shape[0],1)[mask.squeeze(0).squeeze(1)]
        # 16 x+a
        current_nodes_expand = get_adj_nodes(graph, current)
        # check
        assert len(current_nodes_expand) > len(current)
        # 16 x+a+a
        sub_graph = get_sub_graph(graph,current_nodes_expand)

    else:
        im = important.data
        imr = im.sort(0,True)[1].numpy()
        #imr = list(reversed(np.argsort(im)))
        a = 0
        current_nodes_expand = current_nodes
        for i in imr:
            if i not in current_nodes:
                current_nodes_expand.append(i)
            a = a + 1
            if a >= 5:
                break
        current = np.array(sorted(current_nodes_expand))
        current_nodes_expand = get_adj_nodes(graph,current)
        sub_graph = get_sub_graph(graph,current_nodes_expand)

    current = current.tolist()
    return sub_graph, current_nodes_expand, current

def get_batch_graph_nodes(graph_list):
    first = [i.nodes() for i in graph_list]
    return first

def get_adj_nodes_batch(graph, nodes):
    A = graph
    re = list([set(nodes[i]) for i in range(nodes.shape[0])])
    for j in range(nodes.shape[0]):
        for i in nodes[j]:
            yes = np.where(A[i] > 0)[0]
            for nei in yes:
                assert A[i][nei] > 0
                re[j].add(nei)
    return [sorted(list(i)) for i in re]

def get_adj_nodes_batch(graph, nodes_list):
    A = graph
    re = set()
def get_sub_graph(graph, current):
    mask = np.array([True if x in current else False for x in range(graph.shape[0])])
    re = np.transpose(np.transpose(graph[mask])[mask])
    return re

