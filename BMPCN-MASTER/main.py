from backbone import ConvNet, ResNet12
from BMPCN import BMPCN, imageToWord
from utils import set_logging_config, adjust_learning_rate, save_checkpoint, allocate_tensors, preprocessing, \
    initialize_nodes_edges, backbone_two_stage_initialization, one_hot_encode
from dataloader import MiniImagenet, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
import random
import logging
import argparse
import imp


class BMPCNTrainer(object):
    def __init__(self, enc_module, gnn_module, data_loader, log, arg, config, best_step):
        self.arg = arg
        self.config = config
        self.eval_opt = config['eval_config']

        # 初始化 使用的 tensor 放在cuda上面
        self.tensors = allocate_tensors()
        for key, tensor in self.tensors.items():
            self.tensors[key] = tensor.to(self.arg.device)

        # set backbone and DPGN
        self.enc_module = enc_module.to(arg.device)
        self.gnn_module = gnn_module.to(arg.device)

        # set logger
        self.log = log

        # get data loader
        self.data_loader = data_loader

        # set parameters
        self.module_params = list(self.enc_module.parameters()) + list(self.gnn_module.parameters())

        # set loss
        self.edge_loss = nn.BCELoss(reduction='none')
        self.pred_loss = nn.CrossEntropyLoss(reduction='none')
        self.proto_loss = nn.CrossEntropyLoss()

        # initialize other global variables
        self.global_step = best_step
        self.best_step = best_step
        self.val_acc = 0
        self.test_acc = 0

    def eval(self, partition='test', log_flag=True):

        num_supports, num_samples, query_edge_mask, evaluation_mask = preprocessing(
            self.eval_opt['num_ways'],
            self.eval_opt['num_shots'],
            self.eval_opt['num_queries'],
            self.eval_opt['batch_size'],
            self.arg.device)

        query_edge_loss_generations = []
        query_node_cls_acc_generations = []
        # main training loop, batch size is the number of tasks
        for current_iteration, batch in tqdm(enumerate(self.data_loader[partition]())):
            # initialize nodes and edges for dual graph model
            num_protos = 5
            support_data, support_label, query_data, query_label, task_vectors, all_data, all_label_in_edge, node_feature_lgd = initialize_nodes_edges(
                batch,
                num_protos,
                self.tensors,
                self.eval_opt['batch_size'],
                self.eval_opt['num_queries'],
                self.eval_opt['num_ways'],
                self.arg.device)

            # set as eval mode
            self.enc_module.eval()
            self.gnn_module.eval()

            last_layer_data, second_last_layer_data = backbone_two_stage_initialization(all_data, self.enc_module)  # [25,10,128]

            # run the DPGN model
            point_similarity, _, _, _ = self.gnn_module(last_layer_data, second_last_layer_data, node_feature_lgd, task_vectors)

            query_node_cls_acc_generations, query_edge_loss_generations = \
                self.compute_eval_loss_pred(query_edge_loss_generations,
                                            query_node_cls_acc_generations,
                                            all_label_in_edge,
                                            point_similarity,
                                            query_edge_mask,
                                            evaluation_mask,
                                            num_supports,
                                            support_label,
                                            query_label)

        # logging
        if log_flag:
            self.log.info('------------------------------------')
            self.log.info('step : {}  {}_edge_loss : {}  {}_node_acc : {}'.format(
                self.global_step, partition,
                np.array(query_edge_loss_generations).mean(),
                partition,
                np.array(query_node_cls_acc_generations).mean()))

            self.log.info('evaluation: total_count=%d, accuracy: mean=%.2f%%, std=%.2f%%, ci95=%.2f%%' %
                          (current_iteration,
                           np.array(query_node_cls_acc_generations).mean() * 100,
                           np.array(query_node_cls_acc_generations).std() * 100,
                           1.96 * np.array(query_node_cls_acc_generations).std()
                           / np.sqrt(float(len(np.array(query_node_cls_acc_generations)))) * 100))
            self.log.info('------------------------------------')

        return np.array(query_node_cls_acc_generations).mean()

    def compute_eval_loss_pred(self,
                               query_edge_losses,
                               query_node_accs,
                               all_label_in_edge,
                               point_similarities,
                               query_edge_mask,
                               evaluation_mask,
                               num_supports,
                               support_label,
                               query_label):

        point_similarity = point_similarities[-1]
        full_edge_loss = self.edge_loss(1 - point_similarity, 1 - all_label_in_edge)

        pos_query_edge_loss = torch.sum(
            full_edge_loss * query_edge_mask * all_label_in_edge * evaluation_mask) / torch.sum(
            query_edge_mask * all_label_in_edge * evaluation_mask)
        neg_query_edge_loss = torch.sum(
            full_edge_loss * query_edge_mask * (1 - all_label_in_edge) * evaluation_mask) / torch.sum(
            query_edge_mask * (1 - all_label_in_edge) * evaluation_mask)

        # weighted loss for balancing pos/neg
        query_edge_loss = pos_query_edge_loss + neg_query_edge_loss

        # prediction
        query_node_pred = torch.bmm(
            point_similarity[:, num_supports:, :num_supports],
            one_hot_encode(self.eval_opt['num_ways'], support_label.long(), self.arg.device))

        # test accuracy
        query_node_acc = torch.eq(torch.max(query_node_pred, -1)[1], query_label.long()).float().mean()

        query_edge_losses += [query_edge_loss.item()]
        query_node_accs += [query_node_acc.item()]

        return query_node_accs, query_edge_losses


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--config', type=str, default=os.path.join('.', 'config', '5way_5shot_convnet_mini-imagenet_3query.py'))
    parser.add_argument('--checkpoint_dir', type=str, default=os.path.join('.', 'checkpoints'))
    parser.add_argument('--num_gpu', type=int, default=1, help='number of gpu')
    parser.add_argument('--display_step', type=int, default=100)
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--log_dir', type=str, default=os.path.join('.', 'logs'))
    parser.add_argument('--dataset_root', type=str, default='/home/cmz/workspace/datasets/mini_imagenet_3pickles')
    parser.add_argument('--seed', type=int, default=222)
    parser.add_argument('--mode', type=str, default='eval')

    args_opt = parser.parse_args()
    config_file = args_opt.config

    # Set train and test datasets and the corresponding data loaders
    config = imp.load_source("", config_file).config
    eval_opt = config['eval_config']

    args_opt.exp_name = '{}way_{}shot_{}_{}_{}query'.format(eval_opt['num_ways'],
                                                    eval_opt['num_shots'],
                                                    config['backbone'],
                                                    config['dataset_name'],
                                                    eval_opt['num_queries'])

    set_logging_config(os.path.join(args_opt.log_dir, args_opt.exp_name))
    logger = logging.getLogger('main')

    # Load the configuration params of the experiment
    logger.info('Launching experiment from: {}'.format(config_file))
    logger.info('Generated logs will be saved to: {}'.format(args_opt.log_dir))
    logger.info('Generated checkpoints will be saved to: {}'.format(args_opt.checkpoint_dir))
    print()

    logger.info('-------------command line arguments-------------')
    logger.info(args_opt)
    print()
    logger.info('-------------configs-------------')
    logger.info(config)

    np.random.seed(args_opt.seed)
    torch.manual_seed(args_opt.seed)
    torch.cuda.manual_seed_all(args_opt.seed)
    random.seed(args_opt.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if config['dataset_name'] == 'mini-imagenet':
        dataset = MiniImagenet
        print('Dataset: MiniImagenet')
    else:
        logger.info('Invalid dataset:')
        exit()

    cifar_flag = False

    if config['backbone'] == 'convnet':
        enc_module = ConvNet(emb_size=config['emb_size'], cifar_flag=cifar_flag)
        print('Backbone: ConvNet')
    elif config['backbone'] == 'resnet12':
        enc_module = ResNet12(emb_size=config['emb_size'], cifar_flag=cifar_flag)
        print('Backbone: ResNet12')
    else:
        logger.info('Invalid backbone')
        exit()

    dataset_test = dataset(root=args_opt.dataset_root, partition='test')

    test_loader = DataLoader(dataset_test,
                             num_tasks=eval_opt['batch_size'],
                             num_ways=eval_opt['num_ways'],
                             num_shots=eval_opt['num_shots'],
                             num_queries=eval_opt['num_queries'],
                             epoch_size=eval_opt['iteration'],
                             )
    data_loader = {'test': test_loader}
    need_Pretrain = True 
    img_to_word = imageToWord(512, 300, 0.2)
    if need_Pretrain:
        pretrained_dict = torch.load('./pretrain/conv4-mini.pth')
        # pretrained_dict = torch.load('./pretrain/conv4-mini.pth', map_location={'cuda:0':'cuda:1'})
        pretrained_dict = pretrained_dict['model_sd']
        pretrained_encoder_dict = {k[8:]: v for k, v in pretrained_dict.items() if 'encoder.' + k[8:] in pretrained_dict}
        pretrained_imageToWord_dict = {k[12:]: v for k, v in pretrained_dict.items() if 'imageToWord.' + k[12:] in pretrained_dict}
        enc_module.load_state_dict(pretrained_encoder_dict)
        img_to_word.load_state_dict(pretrained_imageToWord_dict)

    gnn_module = BMPCN(config['num_generation'],
                      eval_opt['dropout'],
                      eval_opt['num_ways'],
                      eval_opt['num_shots'],
                      eval_opt['num_ways'] * eval_opt['num_shots'],
                      eval_opt['num_ways'] * eval_opt['num_shots'] + eval_opt['num_ways'] * eval_opt['num_queries'],
                      eval_opt['sample_num_list'],
                      config['point_distance_metric'],
                      config['distribution_distance_metric'],
                      img_to_word
                      )

    # multi-gpu configuration
    [print('GPU: {}  Spec: {}'.format(i, torch.cuda.get_device_name(i))) for i in range(args_opt.num_gpu)]

    if args_opt.num_gpu > 1:
        print('Construct multi-gpu model ...')
        enc_module = nn.DataParallel(enc_module, device_ids=range(args_opt.num_gpu), dim=0)
        gnn_module = nn.DataParallel(gnn_module, device_ids=range(args_opt.num_gpu), dim=0)
        print('done!\n')

    temp_dir = os.path.join(args_opt.checkpoint_dir, args_opt.exp_name)
    if not os.path.exists(temp_dir):
        logger.info('no checkpoint for model')
        exit()
    else:
        if not os.path.exists(os.path.join(args_opt.checkpoint_dir, args_opt.exp_name, 'model_best.pth.tar')):
            logger.info('no checkpoint for model')
            exit()
        else:
            logger.info('find a checkpoint, loading checkpoint from {}'.format(
                os.path.join(args_opt.checkpoint_dir, args_opt.exp_name)))
            best_checkpoint = torch.load(os.path.join(args_opt.checkpoint_dir, args_opt.exp_name, 'model_best.pth.tar'))

            logger.info('best model pack loaded')
            best_step = best_checkpoint['iteration']
            enc_module.load_state_dict(best_checkpoint['enc_module_state_dict'])
            gnn_module.load_state_dict(best_checkpoint['gnn_module_state_dict'])

    # create trainer
    trainer = BMPCNTrainer(enc_module=enc_module,
                          gnn_module=gnn_module,
                          data_loader=data_loader,
                          log=logger,
                          arg=args_opt,
                          config=config,
                          best_step=best_step)

    # here we only provide eval mode to prove our model's effectiveness
    # we will release the full version of BMPCN's source code soon after our paper successfully published 
    if args_opt.mode == 'eval':
        trainer.eval()
    else:
        print('select a mode')
        exit()


if __name__ == '__main__':
    main()
    # 5way 1shot resnet  mini 1query 77.93
    # 5way 5shot resnet  mini 1query 92.55

    # 5way 1shot convnet mini 1query 76.37
    # 5way 1shot convnet mini 2query 73.43
    # 5way 1shot convnet mini 3query 73.80

    # 5way 5shot convnet mini 1query 93.66
    # 5way 5shot convnet mini 2query 85.06
    # 5way 5shot convnet mini 3query 84.68