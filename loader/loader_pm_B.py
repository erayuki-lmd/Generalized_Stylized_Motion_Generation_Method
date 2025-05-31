import os
import sys
import random
import torch
import numpy as np
import argparse
import glob

from torch.utils.data import Dataset, DataLoader
# from ..utils.animation_data import AnimationData
from utils.load_skeleton import Skel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
content_labels = ["walk", "run", "jump", "punch", "kick"]
style_labels = ["angry", "childlike", "depressed", "old", "proud", "sexy", "strutting"]

class MotionDataset_S(Dataset):
    def __init__(self, args, subset_name="train", dataset_list=["style"]):
        super(MotionDataset_S, self).__init__()

        self.args = args
        self.skel = Skel()
        # if subset_name == "train_2":
        #     data_list = glob.glob("dataset/train_2/direction/*.npy")
        # elif subset_name == "test_2":
        #     data_list = glob.glob("dataset/test_2/direction/*.npy")

        self.joint_direction, self.joint_positions, self.joint_velocities, self.styles, self.contents, self.roots, self.contacts = [], [], [], [], [], [], []

        data_count = 0

        for dataset in dataset_list:
            data_dir = os.path.join("dataset",subset_name,dataset)
            data_list = glob.glob(os.path.join(data_dir,"direction","*.npy"))

            for path in data_list:
          
                data_tmp = np.load(path)
                self.joint_direction.append(data_tmp)
                data_tmp = np.load(os.path.join(data_dir,"position",path.split("/")[-1]))
                self.joint_positions.append(data_tmp)
                data_tmp = np.load(os.path.join(data_dir,"velocity",path.split("/")[-1]))
                self.joint_velocities.append(data_tmp)
                data_tmp = np.load(os.path.join(data_dir,"style",path.split("/")[-1]))
                self.styles.append(data_tmp)
                data_tmp = np.load(os.path.join(data_dir,"content",path.split("/")[-1]))
                self.contents.append(data_tmp)

                roots = np.zeros((len(data_tmp),4))
                self.roots.append(roots)

                data_count += 1

        self.dim_dict = {
            "direction": self.joint_direction[0].shape[-1],
            "position": self.joint_positions[0].shape[-1],
            "velocity": self.joint_velocities[0].shape[-1],
            "style": self.styles[0].shape[-1],
            "content": self.contents[0].shape[-1],
            "root": self.roots[0].shape[-1]
        }


        self.len = data_count
        print(self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        episode_length = self.styles[index].shape[0]
        data = {
            "direction": torch.FloatTensor(self.joint_direction[index]).to(device),
            "position": torch.FloatTensor(self.joint_positions[index]).to(device),
            "velocity": torch.FloatTensor(self.joint_velocities[index]).to(device),
            "content": torch.FloatTensor(self.contents[index]).to(device),
            "root": torch.FloatTensor(self.roots[index]).to(device),
            "style": torch.FloatTensor(self.styles[index]).to(device),
            "content_index": torch.LongTensor(np.argwhere(self.contents[index][0] == 1)[0]).to(device)
        }

        return data


class MotionDatasetLack_S(Dataset):
    def __init__(self, args, subset_name="train", dataset_list=["style"], lack_index = -1):
        super(MotionDatasetLack_S, self).__init__()

        self.args = args
        self.lack_index = lack_index
        self.skel = Skel()
        # if subset_name == "train_2":
        #     data_list = glob.glob("dataset/train_2/direction/*.npy")
        # elif subset_name == "test_2":
        #     data_list = glob.glob("dataset/test_2/direction/*.npy")

        self.joint_direction, self.joint_positions, self.joint_velocities, self.styles, self.contents, self.roots, self.contacts = [], [], [], [], [], [], []

        data_count = 0

        for dataset in dataset_list:
            data_dir = os.path.join("dataset",subset_name,dataset)
            data_list = glob.glob(os.path.join(data_dir,"direction","*.npy"))

            for path in data_list:
                data_tmp = np.load(os.path.join(data_dir,"style",path.split("/")[-1]))
                if 1 in data_tmp:
                    style_index = np.argmax(data_tmp)
                else:
                    style_index = 7
                if style_index == lack_index:
                    continue
                data_tmp = np.load(path)
                self.joint_direction.append(data_tmp)
                data_tmp = np.load(os.path.join(data_dir,"position",path.split("/")[-1]))
                self.joint_positions.append(data_tmp)
                data_tmp = np.load(os.path.join(data_dir,"velocity",path.split("/")[-1]))
                self.joint_velocities.append(data_tmp)
                data_tmp = np.load(os.path.join(data_dir,"style",path.split("/")[-1]))
                self.styles.append(data_tmp)
                data_tmp = np.load(os.path.join(data_dir,"content",path.split("/")[-1]))
                self.contents.append(data_tmp)

                roots = np.zeros((len(data_tmp),4))
                self.roots.append(roots)

                data_count += 1

        self.dim_dict = {
            "direction": self.joint_direction[0].shape[-1],
            "position": self.joint_positions[0].shape[-1],
            "velocity": self.joint_velocities[0].shape[-1],
            "style": self.styles[0].shape[-1],
            "content": self.contents[0].shape[-1],
            "root": self.roots[0].shape[-1]
        }


        self.len = data_count
        print(self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        episode_length = self.styles[index].shape[0]
        data = {
            "direction": torch.FloatTensor(self.joint_direction[index]).to(device),
            "position": torch.FloatTensor(self.joint_positions[index]).to(device),
            "velocity": torch.FloatTensor(self.joint_velocities[index]).to(device),
            "content": torch.FloatTensor(self.contents[index]).to(device),
            "root": torch.FloatTensor(self.roots[index]).to(device),
            "style": torch.FloatTensor(self.styles[index]).to(device),
            "content_index": torch.LongTensor(np.argwhere(self.contents[index][0] == 1)[0]).to(device)
        }

        return data