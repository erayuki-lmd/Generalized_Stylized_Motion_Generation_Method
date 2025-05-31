import torch
import torch.nn as nn
import time
from torch import autograd
from utils.other_kinematics import ForwardKinematics

from utils.load_skeleton import Skel
import numpy as np
import copy

def conv_layer(kernel_size, in_channels, out_channels, pad_type='replicate'):
    def zero_pad_1d(sizes):
        return nn.ConstantPad1d(sizes, 0)

    if pad_type == 'reflect':
        pad = nn.ReflectionPad1d
    elif pad_type == 'replicate':
        pad = nn.ReplicationPad1d
    elif pad_type == 'zero':
        pad = zero_pad_1d

    pad_l = (kernel_size - 1) // 2
    pad_r = kernel_size - 1 - pad_l
    return nn.Sequential(pad((pad_l, pad_r)), nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size))


class style_Encoder(nn.Module):
    def __init__(self, args, dim_dict):
        super().__init__()
        layers = []
        input_size = dim_dict["direction"] + dim_dict["position"] + dim_dict["velocity"]
        for _ in range(args.encoder_layer_num):
            layers.append(nn.Linear(input_size, input_size // 4 * 2))
            layers.append(nn.ReLU())
            input_size = input_size // 4 * 2

        layers.append(nn.Linear(input_size, 8))
        layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

        self.softmax = nn.Softmax()

    def forward(self, direction, position, velocity):
        feature = self.layers(torch.cat((direction, position, velocity), dim=-1))
        feature = feature.mean(dim=1)
        feature = self.softmax(feature)
        return feature


class content_Encoder(nn.Module):
    def __init__(self, args, dim_dict):
        super().__init__()
        layers = []
        input_size = dim_dict["direction"] + dim_dict["position"] + dim_dict["velocity"]
        for _ in range(args.encoder_layer_num):
            layers.append(nn.Linear(input_size, input_size // 4 * 2))
            layers.append(nn.ReLU())
            input_size = input_size // 4 * 2

        layers.append(nn.Linear(input_size, 8))
        layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

        self.lstm = nn.LSTM(input_size=8, hidden_size=8, num_layers=args.neutral_layer_num, batch_first=True)
        self.relu = nn.ReLU()

    def forward(self, direction, position, velocity):
        feature = self.layers(torch.cat((direction, position, velocity), dim=-1))
        feature,_ = self.lstm(feature)
        feature = self.relu(feature)
        return feature.mean(dim=1)



class Encoder(nn.Module):
    def __init__(self, args, dim_dict):
        super().__init__()
        layers = []
        input_size = dim_dict["direction"] + dim_dict["position"] + dim_dict["velocity"]
        for _ in range(args.encoder_layer_num):
            layers.append(nn.Linear(input_size, input_size // 4 * 2))
            layers.append(nn.ReLU())
            input_size = input_size // 4 * 2

        layers.append(nn.Linear(input_size, args.latent_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, direction, position, velocity):
        return self.layers(torch.cat((direction, position, velocity), dim=-1))


class ResidualAdapter(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.neutral_branch = nn.LSTM(input_size=args.latent_dim + 8, hidden_size=args.latent_dim + 8,
                                      num_layers=args.neutral_layer_num, batch_first=True)
        self.neutral_init_hidden_state = nn.Parameter(
            torch.zeros(args.neutral_layer_num, args.content_num, args.latent_dim))
        self.neutral_init_cell_state = nn.Parameter(
            torch.zeros(args.neutral_layer_num,args.content_num, args.latent_dim))

    def forward(self, latent_code, test_time=False):
        neutral_output, (hn, cn) = self.neutral_branch(latent_code)
        return neutral_output

class Decoder(nn.Module):
    def __init__(self, args, dim_dict):
        super().__init__()
        current_dim = args.latent_dim + 8
        self.fk = ForwardKinematics()
        self.args = args
        layers = []
        for _ in range(args.decoder_layer_num - 1):
            layers.append(nn.Linear(current_dim, int((current_dim * 1.5 // 2) * 2)))
            layers.append(nn.ReLU())
            current_dim = int((current_dim * 1.5 // 2) * 2)

        self.features = nn.Sequential(*layers)
        self.direction_layer = nn.Linear(current_dim, dim_dict["direction"])
        if not args.no_pos: self.position_layer = nn.Linear(current_dim, dim_dict["position"])
        if not args.no_vel: self.velocity_layer = nn.Linear(current_dim, dim_dict["velocity"])

    def forward(self, latent_code, test_time=False):
        #print(latent_code.shape)
        features = self.features(latent_code)
        #print(torch.cat((latent_code,style), dim=-1).shape)
        output_dict = {}
        output_dict["direction"] = self.direction_layer(features)
        batch_size, length, direction_dim = output_dict["direction"].shape
        #print(length)
        direction_norm = torch.norm(output_dict["direction"].view(batch_size, length, -1, 3), dim=-1, keepdim=True)
        output_dict["direction"] = output_dict["direction"].view(batch_size, length, -1, 3) / direction_norm
        output_dict["direction"] = output_dict["direction"].view(batch_size, length, -1)
        
        if test_time:
            return output_dict

        if not self.args.no_pos:
            output_dict["position"] = self.position_layer(features)
        else:
            output_dict["position"] = self.fk.forwardX(output_dict["direction"].permute(0, 2, 1)).permute(0, 2, 1)
        if not self.args.no_vel:
            output_dict["velocity"] = self.velocity_layer(features)
        else:
            velocity = output_dict["position"][:, 1:, :] - output_dict["position"][:, :-1, :]
            velocity_last = (2 * velocity[:, -1, :] - velocity[:, -2, :]).unsqueeze(1)
            output_dict["velocity"] = torch.cat((velocity, velocity_last), dim=1)

        return output_dict


class Generator(nn.Module):
    def __init__(self, args, dim_dict, pre_train):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__()
        self.args = args
        self.encoder = Encoder(args, dim_dict)
        self.ra = ResidualAdapter(args)
        self.decoder = Decoder(args, dim_dict)
        self.pre_train = pre_train
        ###
        self.fk = ForwardKinematics()
        ###

    def forward(self, direction, position, velocity, test_time=False, style_feature=None):
        batch_size, length, _ = direction.shape
        start_time = time.time()
        ###
        direction = direction.reshape(-1, direction.shape[-1])
        position = position.reshape(-1, position.shape[-1])
        velocity = velocity.reshape(-1, velocity.shape[-1])
        encoded_data = self.encoder(direction, position, velocity)
        encoded_data = encoded_data.view(batch_size, length, -1)
        if self.pre_train:
            style_code = torch.zeros(8).to(self.device)
            style_code = style_code.repeat((batch_size, length,1))
        else:
            style_code = torch.zeros((batch_size,length,8)).to(self.device)
            for i in range(batch_size):
                style_code[i] = style_feature[i].repeat((length,1))
        encoded_data = torch.cat((encoded_data,style_code),dim=2)
        latent_code = self.ra(encoded_data, test_time=test_time)
        output = self.decoder(latent_code, test_time=test_time)
        
        return output
    
    


class Discriminator(nn.Module):
    def __init__(self, args, dim_dict):
        super().__init__()
        layers = []
        current_size = dim_dict["position"] + dim_dict["velocity"]
        dummy_data = torch.zeros(1, current_size, args.episode_length)
        for _ in range(args.discriminator_layer_num):
            layers.append(conv_layer(3, current_size, (current_size // 3) * 2))
            layers.append(nn.LeakyReLU())
            current_size = (current_size // 3) * 2
        self.features = nn.Sequential(*layers)

        self.last_layer = conv_layer(3, current_size, args.feature_dim)

        self.temporal_attention = nn.Linear(64, self.last_layer(self.features(dummy_data)).shape[-1])
        self.feature_attention = nn.Linear(64, args.feature_dim)

    def forward(self, direction, position, velocity, compute_grad):
        input_data = torch.cat((position, velocity), dim=-1).permute(0, 2, 1)
        if compute_grad: input_data.requires_grad_()
        features = self.last_layer(self.features(input_data))

        combined_features = features.sum(dim=1)
        final_score = combined_features.sum(dim=-1)
        grad = None
        if compute_grad:
            batch_size = final_score.shape[0]
            grad = autograd.grad(outputs=final_score.mean(),
                                 inputs=input_data,
                                 create_graph=True,
                                 retain_graph=True,
                                 only_inputs=True)[0]
            grad = (grad ** 2).sum() / batch_size
        return final_score, grad


class RecurrentStylization(nn.Module):
    def __init__(self, args, dim_dict, pre_train=False):
        super().__init__()
        self.generator = Generator(args, dim_dict,pre_train)
        self.discriminator = Discriminator(args, dim_dict)

    def forward_gen(self, direction, position, velocity, test_time=False, style_feature=None):
        return self.generator(direction, position, velocity, test_time=test_time, style_feature=style_feature)

    def forward_dis(self, direction, position, velocity, compute_grad=False):
        return self.discriminator(direction, position, velocity, compute_grad=compute_grad)

    def forward(self, direction, position, velocity, root, style_feature=None):
        generated_motion = self.forward_gen(direction, position, velocity, test_time=False, style_feature=style_feature)
        score = \
            self.forward_dis(generated_motion["direction"], generated_motion["position"], generated_motion["velocity"])[0]
        return generated_motion, score