import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

from .datasets import dataset_folder
# from eval_config import DATASET_PATHS


def get_dataset(opt):
    dset_lst = []
    for cls in opt.classes:
        root = opt.dataroot + '/' + cls
        dset = dataset_folder(opt, root)
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)

def get_my_dataset(opt):
    dataset = []
    for dataset in DATASET_PATHS:
        opt.mode = 'filename'
        real_dataset = dataset_folder(opt, dataset['real_path'])
        dataset.append(real_dataset)
        
        opt.mode = 'filename'
        fake_dataset = dataset_folder(opt, dataset['fake_path'])
        dataset.append(fake_dataset)
        print(f"Loaded datasets for {dataset['key']}: Real - {dataset['real_path']}, Fake - {dataset['fake_path']}")
    
    return torch.utils.data.ConcatDataset(datasets)

def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler


def create_dataloader(opt):
    # print(f'opt.datapath: {opt.dataroot}')
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = get_dataset(opt)
    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=int(opt.num_threads))
    return data_loader


def create_my_dataloader(opt):
    # print(f'opt.datapath: {opt.dataroot}')
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = get_my_dataset(opt)
    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=int(opt.num_threads))
    return data_loader