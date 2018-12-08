# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 19:28:05 2018

@author: hsc
"""
import os
import json
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from data.graph.vgDataSet import *
from operator import itemgetter
def adjust_learning_rate(optimizer, decay=0.9):
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
            nn.Dropout(),
            nn.Linear(4096, 4096),
            )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x




class PipeLine(torch.nn.Module):
    def __init__(self, opt, dataloader, testloader):
        # set opt
        self.opt = opt
        self.dataloader = dataloader
        self.testloader = testloader
        super(PipeLine, self).__init__()

        # set vgg 16
        vgg16 = models.vgg16(pretrained=False)
        vgg16.load_state_dict(torch.load('/home/nesa320/huangshicheng/gitforwork/my_gsnn/models/vgg/vgg16-397923af.pth'))
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
            nn.Linear((4096+opt.label_num+opt.state_dim*opt.label_len),
                self.opt.vg_objects+self.opt.vg_attributes+self.opt.coco_cats),
            nn.Dropout(),
            nn.Sigmoid())

        # cuda
        if self.opt.cuda:
            self.cuda()

    def _attack_adj(self, attack_num=50):
        self.eval()
        pbar = tqdm(self.testloader)
        i = 0
        l = 0
        np.set_printoptions(threshold=np.nan)
        # np.set_printoptions(threshold='nan')  # 全部输出
        for image, label, concat, annotation in pbar:
            i += 1
            # check image
            if image.shape[1] == 1:
                continue

            # cuda
            if self.opt.cuda:
                image = image.cuda()
                annotation = annotation.cuda()
                concat = concat.cuda()
                label = label.cuda()

            # get result
            output, adj = self.forward(image, concat, annotation, None)
            if type(output) == int:
                continue
            loss = F.binary_cross_entropy(output, label)
            print("loss1 = ", loss.data)
            loss.backward()

            # get grad
            grad = self.gsnn.grads[0].reshape(1, -1)
            print(grad.abs().max()*10000)
            salient_n = (grad.where(grad == 0, grad.abs()) * 10000).sort(1, True)[1].cpu().numpy()[0]
            adj_shape = adj.shape
            adj = adj.view(1, -1)[0]
            print(adj.shape)
            # print(grad.cpu().data.numpy())
            for i in range(attack_num):
                print("node %d number %d", (salient_n[i]/1278, salient_n[i]%1278), 10000*grad[0][salient_n[i]])
                adj[salient_n[i]] = 1-adj[salient_n[i]]

            adj = adj.reshape(adj_shape)
            print(adj.shape)
            # get result
            output, adj = self.forward(image, concat, annotation, adj)
            if type(output) == int:
                continue
            loss = F.binary_cross_entropy(output, label)
            self.graph = GraphLoader(self.opt.graph_dir)
            self.labels = json.load(open(self.opt.label2_dir, 'r'))
            self.n = list(self.graph.nodes)
            self.label_mask = list([self.n.index(i) if i in self.n else -1 for i in self.labels])
            self.pure_label_mask = set(self.label_mask)
            self.pure_label_mask.remove(-1)
            mask = annotation.data.cpu().numpy() > 0.5
            output = output.data.cpu().numpy() > 0.8
            output = np.arange(0, 316, 1)[output.squeeze(0)].tolist()
            current = np.arange(0, annotation.shape[1], 1)[mask.squeeze(0)].tolist()
            fuck = list(filter(lambda a: a in list(self.pure_label_mask), current))
            annotation_ = list(map(lambda a: self.label_mask.index(a),fuck))
            mask = label.data.cpu().numpy() == 1
            label_ = np.arange(0, 316, 1)[mask.squeeze(0)].tolist()
            print("true = ", label_)
            print("anno = ", annotation_)
            print("anno1 = ",current)
            print("out = ",output)



            # compute loss
            l += loss
            if i > 1:
                print("loss2 = ", l.data)
                return l.data

    def _attack_feature(self, attack_num=50):
        self.eval()
        pbar = tqdm(self.testloader)
        i = 0
        l = 0
        np.set_printoptions(threshold=np.nan)
        #np.set_printoptions(threshold='nan')  # 全部输出
        for image, label, concat, annotation in pbar:
            i += 1
            # check image
            if image.shape[1] == 1:
                continue

            # cuda
            if self.opt.cuda:
                image = image.cuda()
                annotation = annotation.cuda()
                concat = concat.cuda()
                label = label.cuda()

            # get result
            output,feature = self.forward(image, concat, annotation, None)
            if type(output) == int:
                continue
            loss = F.binary_cross_entropy(output, label)
            loss.backward()
            grad = self.gsnn.grads[0].reshape(1,-1)
            salient_n = (grad.where(grad==0,grad.abs())*10000).sort(1, True)[1].cpu().numpy()[0]
            feature_shape = feature.shape
            feature = feature.view(1,-1)[0]
            for i in range(attack_num):
                feature[salient_n[i]] = 0
            feature = feature.reshape(feature_shape)

            # get result
            output, feature = self.forward(image, concat, annotation, feature)
            if type(output) == int:
                continue
            loss = F.binary_cross_entropy(output, label)

            # compute loss
            l += loss
            if i > 1:
                return l.data

        
    def _test(self, n=100):
        self.eval()
        pbar = tqdm(self.testloader)
        i=0
        l=0
        for image, label, concat, annotation in pbar:
            i += 1
            # check image
            if image.shape[1] == 1:
                continue

            # cuda
            if self.opt.cuda:
                image = image.cuda()
                annotation = annotation.cuda()
                concat = concat.cuda()
                label = label.cuda()

            # get result
            output = self.forward(image, concat, annotation, None)
            if type(output) == int:
                continue
            loss = F.binary_cross_entropy(output, label).data
            print(loss)
            l+=loss
            if i > n:
                print("loss = ",l.data)
                return l.data

    def forward(self, image, concat, annotation, feat):
        """forward pipeline"""
        # send in data
        gsnn_data,feature = self.gsnn(annotation, feat)
        vgg16_data = self.vgg16(image)
        frcnn_data = concat
        if type(gsnn_data) == int:
            return -1,-1
        concat = torch.cat([gsnn_data, vgg16_data, frcnn_data], 1)

        # output
        output = self.output(concat)
        if self.opt.attack_type != 'none':
            return output, feature
        else:
            return output, -1

    def _train(self):
        """train this fucker"""
        self.train()
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
             {'params': self.vgg16.features.parameters(), "lr": self.opt.lr*0.1},
             {'params': self.vgg16.classifier.parameters(),"lr": self.opt.lr}],
            lr=self.opt.lr,
            momentum=self.opt.momentum,
            weight_decay=self.opt.weight_decay)

        # begin training
        epoch = self.opt.checkpoint_epoch
        epoch_whole = self.opt.epochs - self.opt.checkpoint_epoch
        for eps in range(epoch_whole):

            # decay lr
            if eps+epoch % (self.opt.lr_decay_step + 1) == 0:
                adjust_learning_rate(optimizer_adam, self.opt.lr_decay_rate)
                adjust_learning_rate(optimizer_sgd, self.opt.lr_decay_rate)

            # begin
            pbar = tqdm(self.dataloader)
            for image, label, concat, annotation in pbar:

                # check image
                if image.shape[1] == 1:
                    continue


                # self.graph = GraphLoader(self.opt.graph_dir)
                # self.labels = json.load(open(self.opt.label2_dir, 'r'))
                # self.n = list(self.graph.nodes)
                # self.label_mask = list([self.n.index(i) if i in self.n else -1 for i in self.labels])
                # self.pure_label_mask = set(self.label_mask)
                # self.pure_label_mask.remove(-1)
                # mask = annotation.data.cpu().numpy() > 0.5
                # current = np.arange(0, annotation.shape[1], 1)[mask.squeeze(0)].tolist()
                # fuck = list(filter(lambda a: a in list(self.pure_label_mask), current))
                # annotation_ = list(map(lambda a: self.label_mask.index(a),fuck))
                # mask = label.data.cpu().numpy() == 1
                # label_ = np.arange(0, 316, 1)[mask.squeeze(0)].tolist()
                # print("true = ", label_)
                # print("anno = ", annotation_)
                # print("anno1 = ",current)


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
                output, feature = self.forward(image, concat, annotation)
                if type(output) == int:
                    continue
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
                if self.opt.verbose and i%2000 == 0:
                    print('[%d/%d][%d/%d] Loss: %.4f' % (eps+epoch, self.opt.epochs, i, len(pbar), self.test(100)))

                if i%20000 == 0:
                    # save training model
                    save_name = os.path.join('/home/nesa320/huangshicheng/gitforwork/my_gsnn/models/checkpoints',
                                             'gsnn_{}_{}.pth'.format(eps+epoch,i))
                    torch.save({
                        'state_dict': self.state_dict(),
                    }, save_name)


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

        # init net
        self._initialization()


    def _initialization(self):
        # init net
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

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
        self.grads = []

        # set graph
        self.graph =  GraphLoader(opt.graph_dir)
        self.adj = torch.FloatTensor(self.graph.get_adj())
        if self.opt.cuda:
            self.adj = self.adj.cuda()
        if self.opt.attack_type== 'adj':
            self.adj = self.adj.requires_grad_()
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

        # cuda
        if self.opt.cuda:
            self.cuda()

    def _initialize(self):
        # init net
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.1, 0.5)
                m.bias.data.fill_(0)

    def _initialize_importantnet(self, annotation, adj):
        # init net
        anno = torch.where(annotation > 0, torch.full_like(annotation, 1), annotation)
        cur = np.arange(adj.shape[0])[(anno.data.cpu().numpy() != 0).squeeze(0)]
        for i in range(2):
            old = cur
            cur = get_adj_nodes(adj, cur)
            cur_add = list(set(cur) - set(old))
            anno[:, cur_add] = self.opt.importance_factor**(i+1)
        return anno

    def get_grad_feature(self,grad):
        # get grad for attack
        try:
            self.grads[0] = grad
        except:
            self.grads.append(grad)

    def get_grad_adj(self,grad):
        # get grad for attack
        try:
            self.grads[0] = grad
        except:
            self.grads.append(grad)

    def forward(self, annotation, feature):
        # suppose annotation_dim =1000
        if sum(annotation[0].cpu().data.numpy()) == 0:
            return -1,-1
        anno = self._initialize_importantnet(annotation, self.adj).unsqueeze(2)
        anno_copy = anno.clone()
        padding = torch.zeros(len(anno), self.opt.node_num, self.opt.state_dim - self.opt.annotation_dim).float()
        if self.opt.cuda:
            padding = padding.cuda()
        init_input = torch.cat((anno, padding), 2)

        # other stuff
        important = None
        current_nodes = None

        # set up adj
        if self.opt.attack_type== 'adj' and type(feature) == torch.Tensor:
            self.adj = feature
        sub_graph, current_nodes_expand, current = SetUpAdj(self.adj, important, current_nodes, anno, None)
        anno = anno[:,current_nodes_expand,:]
        prop_state = init_input[:,current_nodes_expand,:]
        self.propogation_net.A = sub_graph.unsqueeze(0)
        self.propogation_net.A = torch.cat((self.propogation_net.A,self.propogation_net.A),dim=2)

        for i_step in range(self.n_steps):
            # padding prop_state
            output = self.propogation_net.forward(prop_state, anno, len(current_nodes_expand))
            init_input[:, current_nodes_expand, :] = output
            forimportant = torch.cat((output, anno),2)
            important  = self.important_net(forimportant).squeeze(0).squeeze(1)

            # update adj get top 5
            sub_graph, current_nodes_expand, current = SetUpAdj(self.adj, important, current, anno, current_nodes_expand)

            # update state
            anno = anno_copy[:, current_nodes_expand, :]
            prop_state = init_input[:, current_nodes_expand, :]
            self.propogation_net.A = sub_graph.unsqueeze(0)
            self.propogation_net.A = torch.cat((self.propogation_net.A, self.propogation_net.A), dim=2)

        # output net
        output_1 = self.propogation_net.forward(prop_state, anno, len(current_nodes_expand))
        if self.opt.attack_type == 'feature' and type(feature) == torch.Tensor:
            output_1 = feature
        forimportant = torch.cat((output_1, anno), 2)
        
        # masked fuck
        mask = sorted(list(set([current_nodes_expand.index(i) if i in current else -1 for i in current_nodes_expand])))
        try:
            mask.remove(-1)
        except:
            pass

        # out
        forimportant = forimportant[:, mask, :]
        node_bias = self.node_bias[:, current, :]
        foroutput = torch.cat((forimportant, node_bias), 2)
        output = self.output_net(foroutput)

        # padding and stretching -> 323x5
        target = torch.zeros(self.opt.batch_size, self.opt.label_len, self.state_dim)
        if self.opt.cuda:
            target = target.cuda()

        # decode label
        mask = list(set([i if self.label_mask[i] in current else -1 for i in range(self.opt.label_len)]))
        mask2 = list(set([current.index(i) if i in self.pure_label_mask else -1 for i in current]))
        try:
            mask.remove(-1)
            mask2.remove(-1)
        except:
            pass
        decorated = list(map(itemgetter(0), sorted([(i,self.label_mask[i]) for i in mask], key = itemgetter(1))))

        # output
        target[:, decorated, :] = output[:, mask2, :]
        target = target.view(self.opt.batch_size,self.opt.label_len*self.state_dim,-1).squeeze(2)
        if self.opt.attack_type== 'feature':
            output_1.register_hook(self.get_grad_feature)
            return target,output_1
        elif self.opt.attack_type == 'adj':
            self.adj.register_hook(self.get_grad_adj)
            return target,self.adj
        else:
            return target,-1

def SetUpAdj(graph, important, current_nodes, annotation, expand):
    # set subgraph and expanded_nodes
    if current_nodes is None:
        mask = annotation.data.cpu().numpy() == 1
        current = np.arange(0,graph.shape[0],1)[mask.squeeze(0).squeeze(1)].tolist()
        current_nodes_expand = get_adj_nodes(graph, current)
        sub_graph = get_sub_graph(graph,current_nodes_expand)

    # update due to important
    else:
        imr = important.data.cpu().sort(0,True)[1].numpy().tolist()
        a = 0
        current_nodes_expand = current_nodes
        for i in imr:
            if expand[i] not in current_nodes:
                current_nodes_expand.append(expand[i])
                a = a + 1
            if a >= 5:
                break
        current_nodes_expand = sorted(current_nodes_expand)
        current = current_nodes_expand.copy()
        current_nodes_expand = get_adj_nodes(graph, current)
        sub_graph = get_sub_graph(graph, current_nodes_expand)

    return sub_graph, current_nodes_expand, current

def get_batch_graph_nodes(graph_list):
    # get batch_graph TBD
    first = [i.nodes() for i in graph_list]
    return first

def get_adj_nodes(graph, nodes):
    # get adj nodes
    re = set(nodes)
    yes = set(torch.nonzero(graph[nodes])[:,1].cpu().data.numpy())
    return sorted(list(re|yes))

def get_sub_graph(graph, current):
    # get small graph
    re = ((graph[current,:].transpose(1,0))[current,:]).transpose(1,0)
    return re

