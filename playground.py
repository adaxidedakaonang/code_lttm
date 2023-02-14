# import torch
# from torch.utils.data.dataset import ConcatDataset
# import random
# import math
# import torch
# from torch.utils.data.sampler import RandomSampler
# from Sampler import BalancedBatchSchedulerSampler


# class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
#     """
#     iterate over tasks and provide a random batch per task in each mini-batch
#     """
#     def __init__(self, dataset, batch_size):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.number_of_datasets = len(dataset.datasets)
#         self.largest_dataset_size = max([len(cur_dataset.samples) for cur_dataset in dataset.datasets])

#     def __len__(self):
#         return self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets)

#     def __iter__(self):
#         samplers_list = []
#         sampler_iterators = []
#         for dataset_idx in range(self.number_of_datasets):
#             cur_dataset = self.dataset.datasets[dataset_idx]
#             sampler = RandomSampler(cur_dataset)
#             samplers_list.append(sampler)
#             cur_sampler_iterator = sampler.__iter__()
#             sampler_iterators.append(cur_sampler_iterator)

#         push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
#         step = self.batch_size * self.number_of_datasets
#         samples_to_grab = self.batch_size
#         # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
#         epoch_samples = self.largest_dataset_size * self.number_of_datasets

#         final_samples_list = []  # this is a list of indexes from the combined dataset
#         for _ in range(0, epoch_samples, step):
#             for i in range(self.number_of_datasets):
#                 cur_batch_sampler = sampler_iterators[i]
#                 cur_samples = []
#                 for _ in range(samples_to_grab):
#                     try:
#                         cur_sample_org = cur_batch_sampler.__next__()
#                         cur_sample = cur_sample_org + push_index_val[i]
#                         cur_samples.append(cur_sample)
#                     except StopIteration:
#                         # got to the end of iterator - restart the iterator and continue to get samples
#                         # until reaching "epoch_samples"
#                         sampler_iterators[i] = samplers_list[i].__iter__()
#                         cur_batch_sampler = sampler_iterators[i]
#                         cur_sample_org = cur_batch_sampler.__next__()
#                         cur_sample = cur_sample_org + push_index_val[i]
#                         cur_samples.append(cur_sample)
#                 final_samples_list.extend(cur_samples)

#         return iter(final_samples_list)


# class MyFirstDataset(torch.utils.data.Dataset):
#     def __init__(self):
#         # dummy dataset
#         # self.samples = torch.cat((-torch.ones(5), torch.ones(5)))
#         self.samples = torch.ones(100)

#     def __getitem__(self, index):
#         # change this to your samples fetching logic
#         return self.samples[index]

#     def __len__(self):
#         # change this to return number of samples in your dataset
#         return self.samples.shape[0]


# class MySecondDataset(torch.utils.data.Dataset):
#     def __init__(self):
#         # dummy dataset
#         # self.samples = torch.cat((torch.ones(50) * 5, torch.ones(5) * -5))
#         self.samples = torch.ones(500)*5

#     def __getitem__(self, index):
#         # change this to your samples fetching logic
#         return self.samples[index]

#     def __len__(self):
#         # change this to return number of samples in your dataset
#         return self.samples.shape[0]


# class MyConcateDataset(torch.utils.data.Dataset):
#     def __init__(self, *datasets):
#         self.datasets = datasets

#     def __getitem__(self, index):
#         p = random.random()
#         if p > 0.5:
#             return self.datasets[0][index%len(self.datasets[0])]
#         else:
#             return self.datasets[1][index%len(self.datasets[1])]

#     def __len__(self):
#         return max(len(d) for d in self.datasets)
        

# first_dataset = MyFirstDataset()
# second_dataset = MySecondDataset()
# concat_dataset = ConcatDataset([first_dataset, second_dataset])
# my_concate_dataset = MyConcateDataset(first_dataset, second_dataset)

# if __name__ == "__main__":

#     batch_size = 8

#     # dataloader with BatchSchedulerSampler
#     dataloader = torch.utils.data.DataLoader(dataset=concat_dataset,
#                                             sampler=BalancedBatchSchedulerSampler(dataset=concat_dataset,
#                                                                         batch_size=batch_size),
#                                             batch_size=batch_size,
#                                             shuffle=False)

#     for inputs in dataloader:
#         print(inputs)

####################################################################################################################

from delete_tfcord_redundant_v2 import psnr
import os
from PIL import Image
import numpy as np
dir_path = r"C:\ADAXI\Replay_Data_tfrecord/18_out"
img1 = os.path.join(dir_path, "image/01735.jpg")
img2 = os.path.join(dir_path, "image/08420.jpg")
img1_ = Image.open(img1)
img2_ = Image.open(img2)
img1_np = np.array(img1_)
img2_np = np.array(img2_)
print(psnr(img1_np, img2_np))
