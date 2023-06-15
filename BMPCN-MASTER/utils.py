import os
import logging
import torch
import shutil

def allocate_tensors():
    tensors = dict()
    tensors['support_data'] = torch.FloatTensor()
    tensors['support_label'] = torch.LongTensor()
    tensors['query_data'] = torch.FloatTensor()
    tensors['query_label'] = torch.LongTensor()
    tensors['task_vectors'] = torch.FloatTensor()
    return tensors


def set_tensors(tensors, batch):
    # support_data => [1,25,5,3,84,84]    support_label => [1,25,5]
    support_data, support_label, query_data, query_label, task_vectors = batch
    tensors['support_data'].resize_(support_data.size()).copy_(support_data)
    tensors['support_label'].resize_(support_label.size()).copy_(support_label)
    tensors['query_data'].resize_(query_data.size()).copy_(query_data)
    tensors['query_label'].resize_(query_label.size()).copy_(query_label)
    tensors['task_vectors'].resize_(task_vectors.size()).copy_(task_vectors)

def set_logging_config(logdir):
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logging.basicConfig(format="[%(asctime)s] [%(name)s] %(message)s",
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(logdir, 'log.txt')),
                                  logging.StreamHandler(os.sys.stdout)])

def save_checkpoint(state, is_best, exp_name):
    torch.save(state, os.path.join('{}'.format(exp_name), 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join('{}'.format(exp_name), 'checkpoint.pth.tar'),
                        os.path.join('{}'.format(exp_name), 'model_best.pth.tar'))

def adjust_learning_rate(optimizers, lr, iteration, dec_lr_step, lr_adj_base):
    new_lr = lr * (lr_adj_base ** (int(iteration / dec_lr_step)))
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

def label2edge(label, device):
    # get size
    num_samples = label.size(1)
    # reshape
    label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
    label_j = label_i.transpose(1, 2)
    # compute edge
    edge = torch.eq(label_i, label_j).float().to(device) 
    return edge

# one_hot_encode
def one_hot_encode(num_classes, class_idx, device):
    return torch.eye(num_classes)[class_idx].to(device)

def preprocessing(num_ways, num_shots, num_queries, batch_size, device):
    # set size of support set, query set and total number of data in single task
    num_supports = num_ways * num_shots   # 5
    num_samples = num_supports + num_queries * num_ways  # 10

    # set edge mask (to distinguish support and query edges)
    support_edge_mask = torch.zeros(batch_size, num_samples, num_samples).to(device)  # [25,10,10]
    support_edge_mask[:, :num_supports, :num_supports] = 1
    query_edge_mask = 1 - support_edge_mask
    evaluation_mask = torch.ones(batch_size, num_samples, num_samples).to(device)  # [25,10,10]

    return num_supports, num_samples, query_edge_mask, evaluation_mask


def initialize_nodes_edges(batch, num_protos, tensors, batch_size, num_queries, num_ways, device):
    # allocate data in this batch to specific variables
    set_tensors(tensors, batch)
    support_data = tensors['support_data'].squeeze(0)    # [25,5,3,84,84]
    support_label = tensors['support_label'].squeeze(0)  # [25,5]
    query_data = tensors['query_data'].squeeze(0)        # [25,5,3,84,84]
    query_label = tensors['query_label'].squeeze(0)      # [25,5]
    task_vectors = tensors['task_vectors'].squeeze(0)    # [25,5,300]

    node_lgd_init_support = one_hot_encode(num_protos, support_label.long(), device)  # [25,5,5] support set 
    node_lgd_init_query = (torch.ones([batch_size, num_queries * num_ways, num_protos])
                          * torch.tensor(1. / num_protos)).to(device)
    node_feature_lgd = torch.cat([node_lgd_init_support, node_lgd_init_query], dim=1)

    all_data = torch.cat([support_data, query_data], 1)  # [25,10,3,84,84]  
    all_label = torch.cat([support_label, query_label], 1)  # [25,10]
    all_label_in_edge = label2edge(all_label, device)  # [25,10,10]

    return support_data, support_label, query_data, query_label, task_vectors, all_data, all_label_in_edge, node_feature_lgd


def backbone_two_stage_initialization(full_data, encoder):

    last_layer_data_temp = []
    second_last_layer_data_temp = []
    for data in full_data.chunk(full_data.size(1), dim=1):
        encoded_result = encoder(data.squeeze(1))

        last_layer_data_temp.append(encoded_result[0])
        second_last_layer_data_temp.append(encoded_result[1])

    # last_layer_data: (batch_size, num_samples, embedding dimension)
    last_layer_data = torch.stack(last_layer_data_temp, dim=1)

    # second_last_layer_data: (batch_size, num_samples, embedding dimension)
    second_last_layer_data = torch.stack(second_last_layer_data_temp, dim=1)

    return last_layer_data, second_last_layer_data




