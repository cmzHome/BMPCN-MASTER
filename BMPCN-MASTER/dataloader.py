from __future__ import print_function
from PIL import Image as pil_image
import random
import os
import numpy as np
import pickle
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchnet as tnt


class MiniImagenet(data.Dataset):
    """
    preprocess the MiniImageNet dataset
    """

    def __init__(self, root, partition='train', category='mini'):
        super(MiniImagenet, self).__init__()
        self.root = root
        self.partition = partition
        self.data_size = [3, 84, 84]
        self.dataset_name = 'mini'

        # set normalizer
        mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)  # 标准化

        # set transformer
        if self.partition == 'train':
            self.transform = transforms.Compose([transforms.RandomCrop(84, padding=4),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ColorJitter(brightness=.1,
                                                                        contrast=.1,
                                                                        saturation=.1,
                                                                        hue=.1),
                                                 lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])
        else:  # 'val' or 'test' ,
            self.transform = transforms.Compose([lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])
        print('Loading {} ImageNet dataset -phase {}'.format(category, partition))

        # load data
        dataset_path = os.path.join(self.root, 'mini_imagenet_%s.pickle' % self.partition)
        with open(dataset_path, 'rb') as handle:
            pack = pickle.load(handle, encoding='latin1')

        data = pack['data']
        labels = pack['labels']

        temp = {}
        index = 0
        for label in labels:
            if label not in temp.keys():
                temp[label] = []
            temp[label].append(data[index])
            index += 1

        data = temp
        self.full_class_list = list(data.keys())
        self.data, self.labels = data2datalabel(data)
        self.label2ind = buildLabelIndex(self.labels)

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        image_data = pil_image.fromarray(np.uint8(img))
        image_data = image_data.resize((self.data_size[2], self.data_size[1]))
        return image_data, label

    def __len__(self):
        return len(self.data)

class DataLoader:

    def __init__(self, dataset, num_tasks, num_ways, num_shots, num_queries, epoch_size, num_workers=8, batch_size=1):

        self.dataset = dataset
        self.num_tasks = num_tasks
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_queries = num_queries
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.data_size = dataset.data_size
        self.full_class_list = dataset.full_class_list
        self.label2ind = dataset.label2ind
        self.transform = dataset.transform
        self.phase = dataset.partition
        self.is_eval_mode = (self.phase == 'test') or (self.phase == 'val')

        if dataset.dataset_name == 'mini':
            with open('./yuyiPackage/mini_imagenet_yuyi.pickle', 'rb') as handle:
                self.yuyi = pickle.load(handle)

    def get_class_data(self, iter_idx):
        index = self.label2ind[iter_idx]
        all_data = [self.transform(self.dataset[i][0]) for i in index]
        all_data = torch.stack(all_data)
        return all_data, iter_idx

    def get_task_batch(self):
        # init task batch data
        support_data, support_label, query_data, query_label = [], [], [], []
        for _ in range(self.num_ways * self.num_shots):
            data = np.zeros(shape=[self.num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[self.num_tasks],
                             dtype='float32')
            support_data.append(data)
            support_label.append(label)
        for _ in range(self.num_ways * self.num_queries):
            data = np.zeros(shape=[self.num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[self.num_tasks],
                             dtype='float32')
            query_data.append(data)
            query_label.append(label)

        # for each task
        task_vectors = []
        for t_idx in range(self.num_tasks):
            task_class_list = random.sample(self.full_class_list, self.num_ways)

            word_vectors = self.get_class_yuyi_vectors(task_class_list, self.num_shots)
            task_vectors.append(word_vectors)

            # for each sampled class in task
            for c_idx in range(self.num_ways):
                data_idx = random.sample(self.label2ind[task_class_list[c_idx]], self.num_shots + self.num_queries)
                class_data_list = [self.dataset[img_idx][0] for img_idx in data_idx]
                for i_idx in range(self.num_shots):
                    # set data
                    support_data[i_idx + c_idx * self.num_shots][t_idx] = self.transform(class_data_list[i_idx])
                    support_label[i_idx + c_idx * self.num_shots][t_idx] = c_idx
                # load sample for query set
                for i_idx in range(self.num_queries):
                    query_data[i_idx + c_idx * self.num_queries][t_idx] = \
                        self.transform(class_data_list[self.num_shots + i_idx])
                    query_label[i_idx + c_idx * self.num_queries][t_idx] = c_idx
        support_data = torch.stack([torch.from_numpy(data).float() for data in support_data], 1)
        support_label = torch.stack([torch.from_numpy(label).float() for label in support_label], 1)
        query_data = torch.stack([torch.from_numpy(data).float() for data in query_data], 1)
        query_label = torch.stack([torch.from_numpy(label).float() for label in query_label], 1)
        task_vectors = torch.stack(task_vectors)  # [n_batch, 5, 300]

        # support_data => [25,5,3,84,84]  support_label => [25,5]  query_data => [25,5,3,84,84]  query_label => [25,5]  task_class => [25,5]
        return support_data, support_label, query_data, query_label, task_vectors

    def get_iterator(self, epoch=0):
        rand_seed = epoch
        random.seed(rand_seed)
        np.random.seed(rand_seed)

        def load_function(iter_idx):
            support_data, support_label, query_data, query_label, task_vectors = self.get_task_batch()
            return support_data, support_label, query_data, query_label, task_vectors

        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=range(self.epoch_size), load=load_function)
        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=(False if self.is_eval_mode else True))
        return data_loader

    def get_class_yuyi_vectors(self, x_shot_classes, num_shots):
        class_ids = self.yuyi['class_attribute_id_dict']
        class_vectors = self.yuyi['vectors']

        total_yuyi_vectors = []
        word_dim = 300
        for c in x_shot_classes:
            id = class_ids[c][-1]
            vector = class_vectors[id]
            vector = torch.from_numpy(np.array(vector)).float()

            if num_shots > 1:
                vector = vector.unsqueeze(0).repeat(num_shots,1)

            total_yuyi_vectors.append(vector)

        total_yuyi_vectors = torch.stack(total_yuyi_vectors)
        return total_yuyi_vectors.view(-1, word_dim)

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return self.epoch_size // self.batch_size


def data2datalabel(ori_data):
    data = []
    label = []
    for c_idx in ori_data:
        for i_idx in range(len(ori_data[c_idx])):
            data.append(ori_data[c_idx][i_idx])
            label.append(c_idx)
    return data, label

def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)
    return label2inds
