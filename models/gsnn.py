# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 19:28:05 2018

@author: hsc
"""
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def adjust_learning_rate(optimizer, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']

class finetune(torch.nn.Module):
    """vgg16 before fc7"""
    def __init__(self):
        super(finetune, self).__init__()
        self.features = vgg16 = models.vgg16(pretrained=True).features
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
    def __init__(self, opt):
        # set opt
        self.opt = opt
        super(PipeLine, self).__init__()

        # set vgg 16
        vgg16 = models.vgg16(pretrained=True)
        pretrained_dict = vgg16.state_dict()
        self.vgg16 = finetune()
        model_dict = self.vgg16.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.vgg16.load_state_dict(model_dict)
        self.graph = None

        # set gsnn
        self.gsnn = GSNN(opt, graph=self.graph)

        # set frcnn
        self.frcnn = None

        # set output layer
        self.output = nn.Sequential(
            nn.Linear((4096+80+316*5),
                self.opt.vg_objects+self.opt.vg_attributes+self.opt.coco_cats),
            nn.Dropout(),
            nn.Sigmoid())

    def forward(self, image):
        """forward pipeline"""
        # send in data
        gsnn_data = self.gsnn(image)
        vgg16_data = self.vgg16(image)
        frcnn_data = self.frcnn(image)
        concat = torch.cat([gsnn_data, vgg16_data, frcnn_data], 1)

        # output
        output = self.output(concat)
        return  output

    def _train(self, dataloader):
        """train this fucker"""
        self.train()
        i = 0

        # set optimizers
        vgg16before_params = list(
            map(id, self.vgg16.features.parameters() + self.vgg16.classifier.parameters()))
        gsnn_params = list(map(id, self.gsnn.parameters()))
        other_params = filter(lambda p: id(p) not in vgg16before_params + gsnn_params,
                              self.parameters())
        optimizer_adam = torch.optim.Adam([
            {'params': self.gsnn.parameters(),
             "lr": self.opt.lr,
             "momentum": self.opt.momentum,
             "weight_decay": self.opt.weight_decay
             }
        ])
        optimizer_sgd = torch.optim.SGD(
            [{'params': other_params},
             {'params': self.vgg16.features.parameters() + self.vgg16.classifier.parameters(), "lr": 0.005}],
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
            pbar = tqdm(dataloader)
            for image, labels in pbar:
                i = i + 1
                output = self.forward(image)
                loss = F.binary_cross_entropy(output, labels)

                # set optimizers
                optimizer_sgd.zero_grad()
                optimizer_adam.zero_grad()
                loss.backward()

                # optimizers
                optimizer_adam.step()
                optimizer_sgd.step()

                # verbose
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
        # [n 2n]
       # A_out = A[:, :, self.node_num*self.edge_type_num:]
        # [n,H]
        a_in = torch.bmm(A_in, state_in)
        # [n,H]
       # a_out = torch.bmm(A_in, state_in)
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
    def __init__(self, opt, graph):
        super(GGNN, self).__init__()

        self.state_dim = opt.state_dim
        self.annotation_dim = opt.annotation_dim
        self.edge_type_num = opt.edge_type_num
        self.node_num = opt.node_num
        self.n_steps = opt.n_steps
        self.A = graph.get_adj()
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
    def forward(self, prop_state, annotation):
            in_states = []
            out_states = []
            for i in range(self.edge_type_num):
                # embedding for different edge_types
                # the reason separating in and out is that
                # to concat them later, basically a trick.
                in_states.append(self.in_fcs[i](prop_state))
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.view(-1, self.node_num*self.edge_type_num, self.state_dim)
            prop_state = self.propogator(in_states, prop_state, self.A)

            join_state = torch.cat((prop_state, annotation), 2)
            output = self.out(join_state)
            output = output.sum(2)
            return output


class GSNN(torch.nn.Module):

    def __init__(self, opt, graph):
        super(GSNN, self).__init__()
        self.opt = opt
        self.state_dim = opt.state_dim
        self.annotation_dim = opt.annotation_dim
        self.edge_type_num = opt.edge_type_num
        self.node_num = opt.node_num
        self.n_steps = opt.n_steps
        self.node_bias = torch.zeros().float().requires_grad()
        self.graph = graph
        self.adj = None
        self.important_loss = None

        self.important_net = nn.Linear(self.state_dim+self.annotation_dim, self.node_num)
        self.propogation_net = GGNN(opt=opt, graph=self.graph)
        self.output_net = nn.Linear(self.state_dim+self.annotation_dim+1, self.state_dim)
        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, annotation):

        padding = torch.zeros(len(annotation), self.opt.node_num, self.opt.state_dim - self.opt.annotation_dim).double()
        init_input = torch.cat((annotation, padding), 1)
        current_nodes = []
        important = None
        forimportant = None
        nodes = self.graph.nodes()

        # set up adj
        current_nodes, sub_graph, current_nodes_expand = SetUpAdj(self.opt, self.graph, important, current_nodes, annotation)
        adj = sub_graph.get_adj()
        prop_state = init_input[sorted([nodes.index(i) for i in current_nodes_expand])]


        for i_step in range(self.n_steps):
            # padding prop_state
            prop_state = padding(prop_state)
            output = self.propogation_net(prop_state)
            forimportant = torch.cat((output, annotation),1)
            important  = self.important_net(forimportant)
            # 取important 向量除了已有的前5
            # update adj
            current_nodes, sub_graph, current_nodes_expand = SetUpAdj(self.opt, self.graph, important, current_nodes, annotation)
            adj = sub_graph.get_adj()
            # update state
            prop_state = prop_state[sorted([nodes.index(i) for i in current_nodes_expand])]

        foroutput = torch.cat((forimportant, self.node_bias), 1)
        output = self.output_net(foroutput)
        assert output.shape[0] <= 316
        # padding and stretching -> 316x5
        temp = None
        for i in range(len(current_nodes)):
            if current_nodes[i] in nodes:
                if i == 0:
                    temp = output[0]
                else:
                    temp = torch.cat(temp, output[i])
            else:
                if i == 0:
                    temp = torch.zeros(len(self.opt.state_dim))
                else:
                    temp = torch.cat((temp, torch.zeros(len(self.opt.state_dim))))
        output = output._requires_grad()
        return output

def SetUpAdj(opt, graph, important, current_nodes, annotation):
    sub_graph = None
    current = current_nodes
    nodes = graph.nodes()
    if len(current_nodes) == 0:
        mask = np.where(annotation)
        current = nodes[mask]
        sub_graph = graph.filter_nodes(current)
    else:
        im = important.data.numpy()
        imr = list(reversed(np.argsort(im)))
        a = 0
        for i in imr:
            if i not in current_nodes:
                current.append(nodes[imr[i]])
            a = a + 1
            if a >= 5:
                break
        sub_graph = graph.filter_nodes(current)
    current_nodes_expand = sub_graph.nodes()
    assert set(current_nodes_expand) > set(current)

    return current, sub_graph, current_nodes_expand





