from collections import OrderedDict

config = OrderedDict()
config['dataset_name'] = 'mini-imagenet'
config['backbone'] = 'resnet12'
config['emb_size'] = 128
config['num_generation'] = 5
config['point_distance_metric'] = 'l2'
config['distribution_distance_metric'] = 'l2'


eval_opt = OrderedDict()
eval_opt['num_ways'] = 5
eval_opt['num_shots'] = 5
eval_opt['batch_size'] = 2
eval_opt['iteration'] = 5000
eval_opt['sample_num_list'] = [1,1,1,1,1]
eval_opt['num_queries'] = 1
eval_opt['dropout'] = 0.1

config['eval_config'] = eval_opt
