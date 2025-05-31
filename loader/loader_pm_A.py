import os
import sys
import random
import torch
import numpy as np
import argparse
import glob

from torch.utils.data import Dataset, DataLoader
from utils.animation_data import AnimationData
from utils.load_skeleton import Skel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
content_labels = ["walk", "run", "jump", "punch", "kick"]
style_labels = ["angry", "childlike", "depressed", "old", "proud", "sexy", "strutting"]

class MotionDataset(Dataset):
    def __init__(self, args, subset_name="train", dataset_list=["style","dance"]):
        super(MotionDataset, self).__init__()

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


                style_label = np.zeros((len(data_tmp),len(style_labels)))
                content_label = np.zeros((len(data_tmp),len(content_labels)))
                roots = np.zeros((len(data_tmp),4))


                self.styles.append(style_label)
                self.contents.append(content_label)
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
        output_style = np.zeros(len(style_labels))
        output_style = np.tile([output_style], (episode_length, 1))
        data = {
            "direction": torch.FloatTensor(self.joint_direction[index]).to(device),
            "position": torch.FloatTensor(self.joint_positions[index]).to(device),
            "velocity": torch.FloatTensor(self.joint_velocities[index]).to(device),
            "content": torch.FloatTensor(self.contents[index]).to(device),
            "root": torch.FloatTensor(self.roots[index]).to(device),
            "input_style": torch.FloatTensor(self.styles[index]).to(device),
            "transferred_style": torch.FloatTensor(output_style).to(device),
            # "content_index": torch.LongTensor(np.argwhere(self.contents[index][0] == 1)[0]).to(device)
        }

        return data


class MotionDataset_Ex(Dataset):
    def __init__(self, args, subset_name="train", dataset_list=["style","dance"],ratio=20):
        super(MotionDataset_Ex, self).__init__()

        self.args = args
        self.skel = Skel()
        # if subset_name == "train_2":
        #     data_list = glob.glob("dataset/train_2/direction/*.npy")
        # elif subset_name == "test_2":
        #     data_list = glob.glob("dataset/test_2/direction/*.npy")

        self.joint_direction, self.joint_positions, self.joint_velocities, self.styles, self.contents, self.roots, self.contacts = [], [], [], [], [], [], []

        data_count = 0

        for dataset in dataset_list:
            if dataset == "style":
                data_dir = os.path.join("dataset",subset_name,dataset)
                data_list = glob.glob(os.path.join(data_dir,"direction","*.npy"))
            else:
                data_dir = os.path.join("dataset",subset_name+"_"+str(int(ratio)),dataset)
                print(data_dir)
                data_list = glob.glob(os.path.join(data_dir,"direction","*.npy"))

            for path in data_list:
                    
                data_tmp = np.load(path)
                self.joint_direction.append(data_tmp)
                data_tmp = np.load(os.path.join(data_dir,"position",path.split("/")[-1]))
                self.joint_positions.append(data_tmp)
                data_tmp = np.load(os.path.join(data_dir,"velocity",path.split("/")[-1]))
                self.joint_velocities.append(data_tmp)


                style_label = np.zeros((len(data_tmp),len(style_labels)))
                content_label = np.zeros((len(data_tmp),len(content_labels)))
                roots = np.zeros((len(data_tmp),4))


                self.styles.append(style_label)
                self.contents.append(content_label)
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
        output_style = np.zeros(len(style_labels))
        output_style = np.tile([output_style], (episode_length, 1))
        data = {
            "direction": torch.FloatTensor(self.joint_direction[index]).to(device),
            "position": torch.FloatTensor(self.joint_positions[index]).to(device),
            "velocity": torch.FloatTensor(self.joint_velocities[index]).to(device),
            "content": torch.FloatTensor(self.contents[index]).to(device),
            "root": torch.FloatTensor(self.roots[index]).to(device),
            "input_style": torch.FloatTensor(self.styles[index]).to(device),
            "transferred_style": torch.FloatTensor(output_style).to(device),
            # "content_index": torch.LongTensor(np.argwhere(self.contents[index][0] == 1)[0]).to(device)
        }

        return data