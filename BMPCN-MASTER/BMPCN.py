from importlib.metadata import requires
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch
import math

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, num_sample, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(1, in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(1, num_sample, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(dim=-1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_features, adj):
        n_batch = input_features.shape[0]
        total_weight = self.weight.repeat(n_batch, 1, 1)
        support = torch.bmm(input_features, total_weight)
        output = torch.bmm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class SubGraphEncoder(nn.Module):
    def __init__(self, nfeat, nhid, num_sample, dropout):
        super(SubGraphEncoder, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 2 * nhid, 1)
        self.gc2 = GraphConvolution(2 * nhid, nhid, num_sample)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        adj = adj.transpose(1, 2)
        x = self.gc2(x, adj)

        return x

class imageToWord(nn.Module):
    def __init__(self, in_c, base_c, dropout=0.0):
        super(imageToWord, self).__init__()
        self.in_c = in_c
        self.base_c = base_c
        self.dropout = dropout

        layer_list = []
        layer_list += [nn.Conv2d(in_channels=self.in_c, out_channels=self.base_c * 2, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c * 2),
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]

        layer_list += [nn.Conv2d(in_channels=self.base_c * 2, out_channels=self.base_c, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c),
                       nn.LeakyReLU()]

        self.proto_node_transform = nn.Sequential(*layer_list)

    def forward(self, proto_node):
        vec_num = proto_node.shape[-1]
        batch_num = proto_node.shape[0]

        proto_node = proto_node.view(-1, self.in_c, vec_num)

        # 放入 D2P 网络
        node_feat = self.proto_node_transform(proto_node.unsqueeze(-1))
        node_feat = node_feat.transpose(1, 2).squeeze(-1)
        node_feat = node_feat.mean(dim=1)
        node_feat = node_feat.view(batch_num, -1, self.base_c)

        return node_feat

class PointSimilarity(nn.Module):
    def __init__(self, in_c, base_c, dropout=0.0):
        super(PointSimilarity, self).__init__()

        self.in_c = in_c
        self.base_c = base_c
        self.dropout = dropout

        layer_list = []
        
        layer_list += [nn.Conv2d(in_channels=self.in_c, out_channels=self.base_c * 2, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c * 2),
                       nn.LeakyReLU()]
        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]

        layer_list += [nn.Conv2d(in_channels=self.base_c * 2, out_channels=self.base_c, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c),
                       nn.LeakyReLU()]
        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]

        layer_list += [nn.Conv2d(in_channels=self.base_c, out_channels=1, kernel_size=1)]
        self.point_sim_transform = nn.Sequential(*layer_list)

    def forward(self, vp_last_gen, ep_last_gen, distance_metric):
        vp_i = vp_last_gen.unsqueeze(2)
        vp_j = torch.transpose(vp_i, 1, 2)
        if distance_metric == 'l2':
            vp_similarity = (vp_i - vp_j) ** 2
        elif distance_metric == 'l1':
            vp_similarity = torch.abs(vp_i - vp_j)
        trans_similarity = torch.transpose(vp_similarity, 1, 3)
        ep_ij = torch.sigmoid(self.point_sim_transform(trans_similarity))

        # normalization
        diagonal_mask = 1.0 - torch.eye(vp_last_gen.size(1)).unsqueeze(0).repeat(vp_last_gen.size(0), 1, 1).to(ep_last_gen.get_device())
        ep_last_gen *= diagonal_mask  
        ep_last_gen_sum = torch.sum(ep_last_gen, -1, True)
        ep_ij = F.normalize(ep_ij.squeeze(1) * ep_last_gen, p=1, dim=-1) * ep_last_gen_sum
        diagonal_reverse_mask = torch.eye(vp_last_gen.size(1)).unsqueeze(0).to(ep_last_gen.get_device())
        ep_ij += (diagonal_reverse_mask + 1e-6)
        ep_ij /= torch.sum(ep_ij, dim=2).unsqueeze(-1)
        node_similarity_l2 = -torch.sum(vp_similarity, 3)  # [25,10,10]
        return ep_ij, node_similarity_l2

class LinearTransformation(nn.Module):
    def __init__(self, in_c, out_c):
        super(LinearTransformation, self).__init__()

        self.p2d_transform = nn.Sequential(*[nn.Linear(in_features=in_c, out_features=out_c, bias=True), nn.LeakyReLU()])
        self.out_c = out_c

    def forward(self, point_edge, distribution_node):
        meta_batch = point_edge.size(0)
        num_sample = point_edge.size(1)

        distribution_node = torch.cat([point_edge[:, :, :self.out_c], distribution_node], dim=2)
        distribution_node = distribution_node.view(meta_batch * num_sample, -1)

        distribution_node = self.p2d_transform(distribution_node)
        distribution_node = distribution_node.view(meta_batch, num_sample, -1)

        return distribution_node

class DistributionSimilarity(nn.Module):
    def __init__(self, in_c, base_c, dropout=0.0):
        super(DistributionSimilarity, self).__init__()
        self.in_c = in_c
        self.base_c = base_c
        self.dropout = dropout
        layer_list = []

        layer_list += [nn.Conv2d(in_channels=self.in_c, out_channels=self.base_c * 2, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c * 2),
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]

        layer_list += [nn.Conv2d(in_channels=self.base_c * 2, out_channels=self.base_c, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c),
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]

        layer_list += [nn.Conv2d(in_channels=self.base_c, out_channels=1, kernel_size=1)]
        self.point_sim_transform = nn.Sequential(*layer_list)

    def forward(self, vd_curr_gen, ed_last_gen, distance_metric):
        vd_i = vd_curr_gen.unsqueeze(2)
        vd_j = torch.transpose(vd_i, 1, 2)
        if distance_metric == 'l2':
            vd_similarity = (vd_i - vd_j) ** 2
        elif distance_metric == 'l1':
            vd_similarity = torch.abs(vd_i - vd_j)
        trans_similarity = torch.transpose(vd_similarity, 1, 3)
        ed_ij = torch.sigmoid(self.point_sim_transform(trans_similarity))

        # normalization
        diagonal_mask = 1.0 - torch.eye(vd_curr_gen.size(1)).unsqueeze(0).repeat(vd_curr_gen.size(0), 1, 1).to(ed_last_gen.get_device())
        ed_last_gen *= diagonal_mask 
        ed_last_gen_sum = torch.sum(ed_last_gen, -1, True)
        ed_ij = F.normalize(ed_ij.squeeze(1) * ed_last_gen, p=1, dim=-1) * ed_last_gen_sum
        diagonal_reverse_mask = torch.eye(vd_curr_gen.size(1)).unsqueeze(0).to(ed_last_gen.get_device())
        ed_ij += (diagonal_reverse_mask + 1e-6)
        ed_ij /= torch.sum(ed_ij, dim=2).unsqueeze(-1)

        return ed_ij

class PointUpdate(nn.Module):
    def __init__(self, in_c, base_c, dropout=0.0):
        super(PointUpdate, self).__init__()
        self.in_c = in_c
        self.base_c = base_c
        self.dropout = dropout

        layer_list = []
        layer_list += [nn.Conv2d(in_channels=self.in_c, out_channels=self.base_c * 2, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c * 2),
                       nn.LeakyReLU()]

        layer_list += [nn.Conv2d(in_channels=self.base_c * 2, out_channels=self.base_c, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c),
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]

        self.point_node_transform = nn.Sequential(*layer_list)

    def forward(self, distribution_edge, point_node):
        meta_batch = point_node.size(0)
        num_sample = point_node.size(1)

        diag_mask = 1.0 - torch.eye(num_sample).unsqueeze(0).repeat(meta_batch, 1, 1).to(distribution_edge.get_device())
        edge_feat = F.normalize(distribution_edge * diag_mask, p=1, dim=-1)

        aggr_distribution_feat = torch.bmm(edge_feat, point_node)
        node_feat = torch.cat([point_node, aggr_distribution_feat], -1).transpose(1, 2)

        node_feat = self.point_node_transform(node_feat.unsqueeze(-1))
        node_feat = node_feat.transpose(1, 2).squeeze(-1)

        return node_feat

class YuyiPointUpdate(nn.Module):
    def __init__(self, in_c, base_c, dropout=0.0):
        super(YuyiPointUpdate, self).__init__()
        self.in_c = in_c
        self.base_c = base_c
        self.dropout = dropout

        layer_list = []
        layer_list += [nn.Conv2d(in_channels=self.in_c, out_channels=self.base_c * 2, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c * 2),
                       nn.LeakyReLU()]

        layer_list += [nn.Conv2d(in_channels=self.base_c * 2, out_channels=self.base_c, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c),
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]

        self.point_node_transform = nn.Sequential(*layer_list)

    def forward(self, distribution_edge, instance_edge, point_node):
        meta_batch = point_node.size(0)
        num_sample = point_node.size(1)

        diag_mask = 1.0 - torch.eye(num_sample).unsqueeze(0).repeat(meta_batch, 1, 1).to(distribution_edge.get_device())
        edge_feat = F.normalize(distribution_edge * diag_mask, p=1, dim=-1)

        aggr_feat = torch.bmm(edge_feat, point_node)
        edge_instance = F.normalize(instance_edge * diag_mask, p=1, dim=-1)
        aggr_instance = torch.bmm(edge_instance, point_node)
        node_feat = torch.cat([aggr_feat, aggr_instance], -1).transpose(1, 2)

        node_feat = self.point_node_transform(node_feat.unsqueeze(-1))
        node_feat = node_feat.transpose(1, 2).squeeze(-1)

        return node_feat

class BMPCN(nn.Module):
    def __init__(self, num_generations, dropout, num_way, num_shot, num_support_sample, num_sample, sample_num_list, point_metric, distribution_metric, img_to_word):

        super(BMPCN, self).__init__()
        self.generation = num_generations
        self.dropout = dropout 
        self.num_support_sample = num_support_sample 
        self.num_sample = num_sample 
        self.sample_num_list = sample_num_list
        self.point_metric = point_metric
        self.distribution_metric = distribution_metric 
        self.num_way = num_way 
        self.num_shot = num_shot

        self.image_to_word = img_to_word

        for l in range(self.generation):
            point_node_update = PointUpdate(128 * 2, 128, dropout=self.dropout if l < self.generation - 1 else 0.0)
            point_edge_update = PointSimilarity(128, 128, dropout=self.dropout if l < self.generation - 1 else 0.0)

            local_distribution_edge_update = DistributionSimilarity(num_way, num_way, dropout=self.dropout if l < self.generation - 1 else 0.0)
            local_distribution_node_update = LinearTransformation(2 * num_way, num_way)

            yuyi_point_node_update = YuyiPointUpdate(300 * 2, 300, dropout=self.dropout if l < self.generation - 1 else 0.0)
            yuyi_point_edge_update = PointSimilarity(300, 300, dropout=self.dropout if l < self.generation - 1 else 0.0)

            yuyi_distribution_edge_update = DistributionSimilarity(num_way, num_way, dropout=self.dropout if l < self.generation - 1 else 0.0)
            yuyi_distribution_node_update = LinearTransformation(2 * num_way, num_way)

            sub_graph_encoder = SubGraphEncoder(128, 128, self.num_sample, dropout=self.dropout if l < self.generation - 1 else 0.0)
            sub_graph_edge_update = PointSimilarity(128, 128, dropout=self.dropout if l < self.generation - 1 else 0.0)

            self.add_module('point_node_update{}'.format(l), point_node_update)
            self.add_module('point_edge_update{}'.format(l), point_edge_update)
            self.add_module('local_distribution_node_update{}'.format(l), local_distribution_node_update)
            self.add_module('local_distribution_edge_update{}'.format(l), local_distribution_edge_update)
            self.add_module('yuyi_point_node_update{}'.format(l), yuyi_point_node_update)
            self.add_module('yuyi_point_edge_update{}'.format(l), yuyi_point_edge_update)
            self.add_module('yuyi_distribution_node_update{}'.format(l), yuyi_distribution_node_update)
            self.add_module('yuyi_distribution_edge_update{}'.format(l), yuyi_distribution_edge_update)
            self.add_module('sub_graph_encoder{}'.format(l), sub_graph_encoder)
            self.add_module('sub_graph_edge_update{}'.format(l), sub_graph_edge_update)

    def init_similarity(self, nodes):
        edge_init = []
        for raw in nodes:
            sim = []
            for point in raw:
                sim.append(F.sigmoid(nn.CosineSimilarity(dim=-1)(point, raw)))
            sim = torch.stack(sim)
            edge_init.append(sim)
        edge_init = torch.stack(edge_init)
        edge_num = edge_init.shape[-1]

        for j in range(edge_num):
            edge_init[:, j, j] = 1.0

        return edge_init

    def get_proto_node_with_query(self, node_feat, edge_feat, sample_num):
        num_supports = self.num_support_sample
        n_batch = node_feat.shape[0]
        query_node = node_feat[:, num_supports:, :]

        # here we find out whether query num per class == 1
        # if query num per class = 1  we can use "torch.argmax(query_similarity, dim=-2)" which can be faster
        # else "query_similarity.topk(sample_num, dim=-1, largest=True)" can be used to find topK
        query_num = self.num_sample - self.num_support_sample
        if query_num > self.num_way:
            query_similarity = edge_feat[:, :num_supports, num_supports:]
            _, index = query_similarity.topk(sample_num, dim=-1, largest=True)
        else:
            query_similarity = edge_feat[:, num_supports:, :num_supports]
            index = torch.argmax(query_similarity, dim=-2)

        protos = []  # prototype
        for batch_index in range(0, n_batch):
            num = 0
            task_proto = [[], [], [], [], []]
            for query_index in index[batch_index]:
                current_index = num // self.num_shot
                task_proto[current_index].append(query_node[batch_index][query_index])
                num += 1
                if num % self.num_shot == 0:
                    task_proto[current_index] = torch.stack(task_proto[current_index])
            task_proto = torch.stack(task_proto)
            protos.append(task_proto)

        protos = torch.stack(protos).view(n_batch, self.num_way, sample_num * self.num_shot, -1)
        support_node = node_feat[:, :num_supports, :]
        support_node = support_node.view(n_batch, self.num_way, self.num_shot, -1)
        protos = torch.cat((protos, support_node), dim=-2)
        protos = protos.mean(dim=-2)

        return protos

    def get_proto_distribution(self, protos, node_feat):
        num_samples = self.num_sample

        proto_distribution = None  # [25,10,5]   node-proto distribution
        for sample_index in range(0, num_samples):
            current_nodes = node_feat[:, sample_index, :]  # [25,128]
            current_nodes = current_nodes.unsqueeze(1)  # [25,1,128]
            sim = nn.CosineSimilarity(dim=-1)(current_nodes, protos)  # [25,5]
            sim = sim.unsqueeze(1)  # [25,1,5]
            if proto_distribution is None:
                proto_distribution = sim
            else:
                proto_distribution = torch.cat((proto_distribution, sim), dim=-2)

        return proto_distribution  # [25,10,5]

    def forward(self, point_node, middle_node, local_proto_distribution_node, task_vectors):
        point_edge_similarities = []  # point graph edge loss
        point_node_similarities = []  # point graph node loss
        yuyi_edge_similarities = []  # yuyi graph edge loss
        subGraph_edge_similarities = []  # subGraph edge loss

        yuyi_point_node = self.image_to_word(middle_node)
        yuyi_point_edge = self.init_similarity(yuyi_point_node)
        yuyi_proto_distribution_node = local_proto_distribution_node

        for l in range(self.generation):

            point_edge, node_similarity_l2 = self._modules['point_edge_update{}'.format(l)](point_node, yuyi_point_edge, self.point_metric)
            yuyi_point_edge, _ = self._modules['yuyi_point_edge_update{}'.format(l)](yuyi_point_node, point_edge, self.point_metric)

            subGraph_edge = point_edge.unsqueeze(2).view(-1, 1, self.num_sample)
            subGraph_node = point_node.unsqueeze(1).repeat(1, self.num_sample, 1, 1).view(-1, self.num_sample, 128)
            subGraph_features = self._modules['sub_graph_encoder{}'.format(l)](subGraph_node, subGraph_edge)
            subGraph_features = subGraph_features.mean(dim=1).view(-1, self.num_sample, 128)
            if l == 0:
                subGraph_sim_edge = self.init_similarity(subGraph_features)
            subGraph_sim_edge, _ = self._modules['sub_graph_edge_update{}'.format(l)](subGraph_features, subGraph_sim_edge, self.point_metric)

            local_proto_node = self.get_proto_node_with_query(point_node, subGraph_sim_edge, self.sample_num_list[l])
            local_proto_edge = self.get_proto_distribution(local_proto_node, point_node)
            local_proto_distribution_node = self._modules['local_distribution_node_update{}'.format(l)](local_proto_edge, local_proto_distribution_node)  # [25,10,5]/[25,30,5]

            if l == 0:
                local_proto_distribution_edge = self.init_similarity(local_proto_distribution_node)
            local_proto_distribution_edge = self._modules['local_distribution_edge_update{}'.format(l)](local_proto_distribution_node, local_proto_distribution_edge, self.distribution_metric)  # [25,10,10]/[25,30,30]

            point_node = self._modules['point_node_update{}'.format(l)](local_proto_distribution_edge, point_node)

            yuyi_proto_edge = self.get_proto_distribution(task_vectors, yuyi_point_node)
            yuyi_proto_distribution_node = self._modules['yuyi_distribution_node_update{}'.format(l)](yuyi_proto_edge, yuyi_proto_distribution_node)  # [25,10,5]/[25,30,5]

            if l == 0:
                yuyi_proto_distribution_edge = self.init_similarity(yuyi_proto_distribution_node)
            yuyi_proto_distribution_edge = self._modules['yuyi_distribution_edge_update{}'.format(l)](yuyi_proto_distribution_node, yuyi_proto_distribution_edge, self.distribution_metric)  # [25,10,10]/[25,30,30]

            yuyi_point_node = self._modules['yuyi_point_node_update{}'.format(l)](yuyi_proto_distribution_edge, yuyi_point_edge, yuyi_point_node)  # [25,10,256]/[25,30,256]

            point_edge_similarities.append(point_edge)
            point_node_similarities.append(node_similarity_l2)
            yuyi_edge_similarities.append(yuyi_point_edge)
            subGraph_edge_similarities.append(subGraph_sim_edge)

        return point_edge_similarities, point_node_similarities, yuyi_edge_similarities, subGraph_edge_similarities