import argparse
import json
import os
import math
import torch
import time
import numpy as np
import os
import matplotlib.pyplot as plt 
from statistics import mean
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import Dataset, DataLoader
from models.model_pm import RecurrentStylization,style_Encoder
from utils.utils import make_dir, get_style_name, create_logger
from utils.other_postprocess import save_bvh_from_network_output, remove_fs
from utils.utils import *
from utils.load_skeleton import Skel
import cv2

style_name = "depressed_run_1.npy"
motion_name = "gBR_sBM_cAll_d05_mBR0_ch02.npy"

style_model_name = "RESULT/models/pm/style_10.pt"
motion_model_name = "RESULT/models/pm/model_200.pt"

exp_dir = "EXP/exp_1/pm/"
os.makedirs(exp_dir,exist_ok=True)

style_weight = 0.8


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
content_labels = ["walk", "run", "jump", "punch", "kick"]
style_labels = ["angry", "childlike", "depressed", "old", "proud", "sexy", "strutting"]



class MiniData(Dataset):
    def __init__(self, args, style_path, motion_path):
        super(MiniData, self).__init__()

        self.args = args
        self.skel = Skel()
        self.style_path = style_path
        self.motion_path = motion_path

        self.style_direction, self.style_positions, self.style_velocities, self.style_roots, self.motion_direction, self.motion_positions, self.motion_velocities, self.motion_roots = [], [], [], [], [], [], [], []

        data_count = 1

        data_tmp = np.load("dataset/test/style/direction/"+self.style_path)
        self.style_direction.append(data_tmp)
        data_tmp = np.load("dataset/test/style/position/"+self.style_path)
        self.style_positions.append(data_tmp)
        data_tmp = np.load("dataset/test/style/velocity/"+self.style_path)
        self.style_velocities.append(data_tmp)
        roots = np.zeros((len(data_tmp),4))
        self.style_roots.append(roots)

        data_tmp = np.load("dataset/test/motion/direction/"+self.motion_path)
        self.motion_direction.append(data_tmp)
        data_tmp = np.load("dataset/test/motion/position/"+self.motion_path)
        self.motion_positions.append(data_tmp)
        data_tmp = np.load("dataset/test/motion/velocity/"+self.motion_path)
        self.motion_velocities.append(data_tmp)
        roots = np.zeros((len(data_tmp),4))
        self.motion_roots.append(roots)


        self.dim_dict = {
            "direction": self.style_direction[0].shape[-1],
            "position": self.style_positions[0].shape[-1],
            "velocity": self.style_velocities[0].shape[-1],
            "root": self.style_roots[0].shape[-1]
        }


        self.len = data_count
        # print(self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        style_data = {
            "direction": torch.FloatTensor(self.style_direction[index]).to(device),
            "position": torch.FloatTensor(self.style_positions[index]).to(device),
            "velocity": torch.FloatTensor(self.style_velocities[index]).to(device),
            "root": torch.FloatTensor(self.style_roots[index]).to(device)
        }
        motion_data = {
            "direction": torch.FloatTensor(self.motion_direction[index]).to(device),
            "position": torch.FloatTensor(self.motion_positions[index]).to(device),
            "velocity": torch.FloatTensor(self.motion_velocities[index]).to(device),
            "root": torch.FloatTensor(self.motion_roots[index]).to(device)
        }

        return style_data,motion_data


class ArgParserTrain(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # setup hyperparameters
        self.add_argument('--n_epoch', type=int, default=2000)
        self.add_argument('--episode_length', type=int, default=24)
        self.add_argument("--seed", default=88, type=int)
        self.add_argument('--test_freq', type=int, default=10)
        self.add_argument('--log_freq', type=int, default=30)
        self.add_argument('--save_freq', type=int, default=10)
        self.add_argument('--style_num', type=int, default=7)
        self.add_argument('--content_num', type=int, default=5)
        self.add_argument("--data_path", default='./data/xia.npz')
        self.add_argument("--work_dir", default="./"+exp_dir+"/")
        self.add_argument("--load_dir", default=None, type=str)
        self.add_argument("--tag", default='', type=str)
        self.add_argument("--train_classifier", default=False, action='store_true')

        # training hyperparameters
        self.add_argument("--batch_size", default=16, type=int)
        self.add_argument("--w_reg", default=128, type=int)
        self.add_argument("--dis_lr", default=5e-6, type=float)
        self.add_argument("--gen_lr", default=1e-5, type=float)
        self.add_argument("--cla_lr", default=1e-4, type=float)
        self.add_argument("--perceptual_loss", default=False, action='store_true')

        # model hyperparameters
        self.add_argument("--no_pos", default=True, action='store_true')
        self.add_argument("--no_vel", default=False, action='store_true')
        self.add_argument("--encoder_layer_num", type=int, default=2)
        self.add_argument("--decoder_layer_num", type=int, default=4)
        self.add_argument("--discriminator_layer_num", type=int, default=4)
        self.add_argument("--classifier_layer_num", type=int, default=5)
        self.add_argument("--latent_dim", type=int, default=32)
        self.add_argument("--neutral_layer_num", type=int, default=4)
        self.add_argument("--style_layer_num", type=int, default=6)
        self.add_argument("--feature_dim", type=int, default=16)

        #for_sh
        self.add_argument("--motion_data", type=str, default=motion_name)
        self.add_argument("--style_data", type=str, default=style_name)
        self.add_argument("--style_weight", type=float, default=style_weight)


def main():
    args = ArgParserTrain().parse_args()
    tester = Tester(args)
    tester.gen()

class Tester():
    def __init__(self, args):
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)

        self.style_name = args.style_data
        self.motion_name = args.motion_data
        self.style_weight = args.style_weight

        self.data = MiniData(args, self.style_name, self.motion_name)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.data, batch_size=1, shuffle=False)


        self.model = RecurrentStylization(args, self.data.dim_dict,False).to(device)
        self.model.load_state_dict(torch.load(motion_model_name),strict=False)

        self.style_model = style_Encoder(args, self.data.dim_dict).to(device)
        self.style_model.load_state_dict(torch.load(style_model_name),strict=False)


    def gen(self):
        self.model.eval()

        for i, (style_tmp, motion_tmp) in enumerate(self.dataloader):
            # print(motion_tmp["direction"].shape)
            style_style = self.style_model(style_tmp["direction"], style_tmp["position"], style_tmp["velocity"])
            print(style_style.shape)
            origin_style = self.style_model(motion_tmp["direction"], motion_tmp["position"], motion_tmp["velocity"])
            style_feature = style_style * (self.style_weight) + origin_style * (1-self.style_weight)
            reconstructed_motion = self.model.forward_gen(motion_tmp["direction"], motion_tmp["position"], motion_tmp["velocity"], style_feature=style_feature)
            break
        reconstructed_motion = reconstructed_motion["position"].squeeze(0)
        reconstructed_motion = reconstructed_motion.cpu().detach().numpy()
        gt = motion_tmp["position"][0].cpu().detach().numpy().copy()
        style = style_tmp["position"][0].cpu().detach().numpy().copy()
        np.save(exp_dir+"/"+self.motion_name[:-4]+"-by-"+self.style_name[:-4],reconstructed_motion.reshape((len(reconstructed_motion),-1,3)))

        # make_videos(reconstructed_motion,gt,style)

if __name__ == "__main__":
    main()
