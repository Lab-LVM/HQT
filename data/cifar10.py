# reference by https://github.com/luoxiao12/CIMON/blob/main/data/cifar10.py

import os, sys, pickle
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as dataset
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from data.transform import train_transforms, test_transforms, Onehot, encode_onehot, train_aug_transforms



def load_data(root, num_query, num_train, batch_size, num_workers, hash_model, mean, std, img_size, scale,
              get_feature=False, hqt_label=None, bihalf_loader=True):
    """
    Load cifar10 dataset.

    Args
        root(str): Path of dataset.
        num_query(int): Number of query data points.
        num_train(int): Number of training data points.
        batch_size(int): Batch size.
        num_workers(int): Number of loading data threads.
        hash_name(str): Hashing Model Name.
        mean(float): dataset mean
        std(float): dataset std
        scale(float): Scale of RandomResizeCrop
        get_feature(bool): True: Feature extraction
        hqt(bool): Ture: Using HQT pseudo labels.

    Returns
        query_dataloader, train_dataloader, retrieval_dataloader(torch.evaluate.data.DataLoader): Data loader.
    """
    if hash_model == "bihalf":
        train_transform = train_transforms(mean, std, img_size)
    else:
        train_transform = train_aug_transforms(mean, std, img_size, scale)
    query_transform = test_transforms(mean, std, img_size)
    if bihalf_loader:
        train_dataset, query_dataset, retrieval_dataset = parsedata()
    else:
        CIFAR10.init(root, num_query, num_train)
        train_dataset = CIFAR10('train', transform=train_transform, target_transform=None)
        query_dataset = CIFAR10('query', transform=query_transform, target_transform=Onehot())
        retrieval_dataset = CIFAR10('database', transform=query_transform, target_transform=Onehot())
    if get_feature:
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=False,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=num_workers,
        )
        return train_dataloader
    if hqt_label is not None:
        train_dataset.targets = hqt_label

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )
    query_dataloader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
      )
    retrieval_dataloader = DataLoader(
        retrieval_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )
    return train_dataloader, query_dataloader, retrieval_dataloader

class CIFAR10(Dataset):
    """
    Cifar10 dataset.
    """
    @staticmethod
    def init(root, num_query, num_train):
        data_list = ['data_batch_1',
                     'data_batch_2',
                     'data_batch_3',
                     'data_batch_4',
                     'data_batch_5',
                     'test_batch',
                     ]
        base_folder = 'cifar-10-batches-py'

        data = []
        targets = []

        for file_name in data_list:
            file_path = os.path.join(root, base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                data.append(entry['data'])
                if 'labels' in entry:
                    targets.extend(entry['labels'])
                else:
                    targets.extend(entry['fine_labels'])

        data = np.vstack(data).reshape(-1, 3, 32, 32)
        data = data.transpose((0, 2, 3, 1))  # convert to HWC
        targets = np.array(targets)

        # Sort by class
        sort_index = targets.argsort()
        data = data[sort_index, :]
        targets = targets[sort_index]

        # (num_query / number of class) query images per class
        # (num_train / number of class) train images per class
        query_per_class = num_query // 10
        train_per_class = num_train // 10

        # Permutate index (range 0 - 6000 per class)
        perm_index = np.random.permutation(data.shape[0] // 10)
        query_index = perm_index[:query_per_class]
        train_index = perm_index[query_per_class: query_per_class + train_per_class]

        query_index = np.tile(query_index, 10)
        train_index = np.tile(train_index, 10)
        inc_index = np.array([i * (data.shape[0] // 10) for i in range(10)])
        query_index = query_index + inc_index.repeat(query_per_class)
        train_index = train_index + inc_index.repeat(train_per_class)
        list_query_index = [i for i in query_index]
        retrieval_index = np.array(list(set(range(data.shape[0])) - set(list_query_index)), dtype=np.int32)

        # Split data, targets
        CIFAR10.QUERY_IMG = data[query_index, :]
        CIFAR10.QUERY_TARGET = targets[query_index]
        CIFAR10.TRAIN_IMG = data[train_index, :]
        CIFAR10.TRAIN_TARGET = targets[train_index]
        CIFAR10.RETRIEVAL_IMG = data[retrieval_index, :]
        CIFAR10.RETRIEVAL_TARGET = targets[retrieval_index]

    def __init__(self, mode='train',
                 transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        if mode == 'train':
            self.data = CIFAR10.TRAIN_IMG
            self.targets = CIFAR10.TRAIN_TARGET
        elif mode == 'query':
            self.data = CIFAR10.QUERY_IMG
            self.targets = CIFAR10.QUERY_TARGET
        else:
            self.data = CIFAR10.RETRIEVAL_IMG
            self.targets = CIFAR10.RETRIEVAL_TARGET

        self.onehot_targets = encode_onehot(self.targets, 10)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        img_aug_1 = self.transform(img)
        img_aug_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_aug_1, img_aug_2, target, index

    def __len__(self):
        return len(self.data)

    def get_onehot_targets(self):
        """
        Return one-hot encoding targets.
        """
        return torch.FloatTensor(self.onehot_targets)


class Bihalf_CIFAR10(dataset.CIFAR10):
    def __init__(self, root, train, transform, target_transform, download=False):
        super().__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        img1 = img
        img2 = img

        return img1, img2, target, index


def parsedata():
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = Bihalf_CIFAR10(root='./dataset',
                                  train=True,
                                  transform=train_transform,
                                  target_transform=None,
                                  download=True)

    test_dataset = Bihalf_CIFAR10(root='./dataset',
                                 train=False,
                                 transform=test_transform,
                                 target_transform=Onehot())

    database_dataset = Bihalf_CIFAR10(root='./dataset',
                                     train=False,
                                     transform=test_transform,
                                     target_transform=Onehot())

    # Re-Construct training, query and database set
    X = train_dataset.data
    L = np.array(train_dataset.targets)

    X = np.concatenate((X, test_dataset.data))
    L = np.concatenate((L, np.array(test_dataset.targets)))

    first = True

    for label in range(10):
        index = np.where(L == label)[0]

        N = index.shape[0]
        np.random.seed(0)
        perm = np.random.permutation(N)
        index = index[perm]

        data = X[index[0:1000]]
        labels = L[index[0:1000]]
        if first:
            test_L = labels
            test_data = data
        else:
            test_L = np.concatenate((test_L, labels))
            test_data = np.concatenate((test_data, data))

        data = X[index[1000:6000]]
        labels = L[index[1000:6000]]
        if first:
            dataset_L = labels
            data_set = data
        else:
            dataset_L = np.concatenate((dataset_L, labels))
            data_set = np.concatenate((data_set, data))

        data = X[index[1000:1500]]
        labels = L[index[1000:1500]]
        if first:
            train_L = labels
            train_data = data
        else:
            train_L = np.concatenate((train_L, labels))
            train_data = np.concatenate((train_data, data))

        first = False
        train_dataset.data = train_data
        train_dataset.targets = train_L.astype(np.int32)
        test_dataset.data = test_data
        test_dataset.targets = (test_L).astype(np.int32)
        database_dataset.data = data_set
        database_dataset.targets = (dataset_L).astype(np.int32)

    return train_dataset, test_dataset, database_dataset