import argparse
import json
import os
import torch
import time
import numpy as np
import os
from statistics import mean
from torch.utils.tensorboard import SummaryWriter

from models.model_pm import RecurrentStylization,style_Encoder
from utils.utils import make_dir, get_style_name, create_logger
from loader.loader_pm_A import MotionDataset
from loader.loader_pm_B import MotionDataset_S
from utils.other_postprocess import save_bvh_from_network_output, remove_fs
from utils.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

exp_dir = "RESULT/models/pm"
content_labels = ["walk", "run", "jump", "punch", "kick"]
style_labels = ["angry", "childlike", "depressed", "old", "proud", "sexy", "strutting"]
utilizing_datasets = ["style", "motion"]


body_weight = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
w_ave = float(sum(body_weight)/25)

class ArgParserTrain(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # setup hyperparameters
        self.add_argument('--n_epoch', type=int, default=500)
        self.add_argument('--episode_length', type=int, default=24)
        self.add_argument("--seed", default=88, type=int)
        self.add_argument('--test_freq', type=int, default=100)
        self.add_argument('--log_freq', type=int, default=30)
        self.add_argument('--save_freq', type=int, default=100)
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


def main():
    args = ArgParserTrain().parse_args()
    trainer = Trainer(args)
    if args.train_classifier:
        trainer.train_classifier()
    else:
        trainer.train()

class Trainer():
    def __init__(self, args):
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)

        self.num_batch = args.batch_size

        exp_name = "train"
        experiment_dir = os.path.join(args.work_dir, exp_name)

        if os.path.isdir(experiment_dir):
            mini_number = 1
            while True:
                experiment_dir = os.path.join(args.work_dir, exp_name+"_ver"+str(mini_number))
                if os.path.isdir(experiment_dir):
                    mini_number += 1
                else:
                    break

        os.makedirs(experiment_dir)
        self.video_dir = make_dir(os.path.join(experiment_dir, 'video'))
        self.model_dir = make_dir(os.path.join(experiment_dir, 'model'))
        self.args = args
        self.logger = create_logger(experiment_dir)

        with open(os.path.join(experiment_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)
        
        self.dataset = MotionDataset(args,dataset_list=utilizing_datasets)
        self.test_dataset = MotionDataset(args, subset_name="test",dataset_list=utilizing_datasets)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=args.batch_size, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=1, shuffle=True)
        
        self.dataset_SC = MotionDataset_S(args)
        self.test_dataset_SC = MotionDataset_S(args, subset_name="test")
        self.dataloader_SC = torch.utils.data.DataLoader(dataset=self.dataset_SC, batch_size=args.batch_size, shuffle=True)
        self.testloader_SC = torch.utils.data.DataLoader(dataset=self.test_dataset_SC, batch_size=1, shuffle=True)

        self.model = RecurrentStylization(args, self.dataset.dim_dict,False).to(device)
        # self.model.load_state_dict(torch.load('./'+exp_dir+'/motion_500.pt'),strict=False)
        self.loss = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        self.criterion = torch.nn.MSELoss()
        self.gen_optimizer = torch.optim.Adam(self.model.generator.parameters(), lr=args.gen_lr)

        self.style_model = style_Encoder(args, self.dataset.dim_dict).to(device)
        self.style_model.load_state_dict(torch.load('./'+exp_dir+'/style_10.pt'),strict=False)


        self.reset_loss_dict()

    def train_gen(self, train_data):

        for i, style_train in enumerate(self.dataloader_SC):
            style_motion = style_train
            break

        #style_extruction
        style_feature = self.style_model(style_motion["direction"], style_motion["position"], style_motion["velocity"])

        # reconstruction
        reconstructed_motion = self.model.forward_gen(train_data["direction"], train_data["position"], train_data["velocity"], style_feature=style_feature)

        # reconstruction
        self_style = self.style_model(train_data["direction"], train_data["position"], train_data["velocity"])
        reconstructed_motion_wo_style = self.model.forward_gen(train_data["direction"], train_data["position"], train_data["velocity"], style_feature=self_style)

        # fake motion
        fake_motion, fake_score = self.model(train_data["direction"], train_data["position"], train_data["velocity"],train_data["root"], style_feature=style_feature)

        # reconstruct style
        reconstruct_style = self.style_model(reconstructed_motion["direction"], reconstructed_motion["position"], reconstructed_motion["velocity"])

        style_loss = 0
        for i in range(len(style_feature)):
            style_loss -= self.loss(style_feature[i],reconstruct_style[i])

        return self.compute_generator_loss(fake_motion, fake_score, style_loss, train_data, reconstructed_motion_wo_style)

    def compute_generator_loss(self, fake_motion, fake_score, style_loss, train_data=None, reconstructed_motion_wo_style=None):

        # if fake_motion != None:
        #     adversarial_loss = self.criterion(fake_score, torch.zeros_like(fake_score))
        # else:
        adversarial_loss = torch.tensor(0).to(device)

        if train_data != None:
            direction_loss = self.direction_difference(reconstructed_motion_wo_style["direction"], train_data["direction"]) * 0.05
            position_loss = self.criterion(reconstructed_motion_wo_style["position"], train_data["position"])
            velocity_loss = self.criterion(reconstructed_motion_wo_style["velocity"], train_data["velocity"])
        else:
            direction_loss = torch.tensor(0).to(device)
            position_loss = torch.tensor(0).to(device)
            velocity_loss = torch.tensor(0).to(device)


        self.loss_dict["direction_loss"].append(direction_loss.item())
        self.loss_dict["position_loss"].append(position_loss.item())
        self.loss_dict["velocity_loss"].append(velocity_loss.item())
        self.loss_dict["adversarial_loss"].append(adversarial_loss.item())
        self.loss_dict["style_loss"].append(style_loss.item())
        generator_loss = adversarial_loss + style_loss + 1.0 * direction_loss + 8.0 * position_loss + velocity_loss + 1.0
        self.loss_dict["generator_loss"].append(generator_loss.item())
        return generator_loss

    def direction_difference(self, pos1, pos2):
        epsilon = 1e-7
        batch_size, duration, _ = pos1.shape
        pos1 = pos1.reshape(batch_size, duration, -1, 3)
        pos2 = pos2.reshape(batch_size, duration, -1, 3)
        # weight_tmp = (np.tile([body_weight], (batch_size, duration,1)))
        # print(torch.dot(torch.abs((pos1 * pos2).sum(dim=-1))*weight_tmp).shape)
        angle = torch.acos(torch.clamp(torch.abs((pos1 * pos2).sum(dim=-1)), -1+epsilon, 1-epsilon))
        angle = angle**2
        for i,j in enumerate(body_weight):
            angle[:,:,i] *= j
        return angle.sum(dim=-1).sum(dim=-1).mean()/w_ave

    def reset_loss_dict(self):
        self.loss_dict = {
            "generator_loss": [],
            "direction_loss": [],
            "position_loss": [],
            "velocity_loss": [],
            "adversarial_loss": [],
            "style_loss": []
        }

    def train(self):
        self.style_model.eval()
        for epoch in range(self.args.n_epoch):
            for i, train_data in enumerate(self.dataloader):
                if len(train_data["position"]) == self.num_batch:

                    generator_loss = self.train_gen(train_data)
                    self.gen_optimizer.zero_grad()
                    generator_loss.backward()
                    self.gen_optimizer.step()

                    if (i + 1) % self.args.log_freq == 0:
                        self.logger.info(
                            'Train: Epoch [{}/{}], Step [{}/{}]| g_loss: {:.3f}| d_loss: {:.3f}| p_loss: {:.3f}| v_loss: {:.3f}| adv_loss: {:.3f}| sty_loss: {:.3f}'
                                .format(epoch + 1, self.args.n_epoch, i + 1, len(self.dataloader),
                                        mean(self.loss_dict["generator_loss"]),
                                        mean(self.loss_dict["direction_loss"]),
                                        mean(self.loss_dict["position_loss"]),
                                        mean(self.loss_dict["velocity_loss"]),
                                        mean(self.loss_dict["adversarial_loss"]),
                                        mean(self.loss_dict["style_loss"])
                                        ))
                        self.reset_loss_dict()
                else:
                    print("break")

            if (epoch + 1) % self.args.test_freq == 0:
                self.model.eval()
                j = np.random.randint(0, len(self.testloader), size=5)

                for i, test_data in enumerate(self.testloader):
                    for _, style_train in enumerate(self.testloader_SC):
                        style_motion = style_train
                        break

                    #style_extruction
                    style_feature = self.style_model(style_motion["direction"], style_motion["position"], style_motion["velocity"])
                    reconstructed_motion = self.model.forward_gen(test_data["direction"], test_data["position"], test_data["velocity"], style_feature=style_feature)
                    reconstruct_style = self.style_model(reconstructed_motion["direction"], reconstructed_motion["position"], reconstructed_motion["velocity"])
                    style_loss = - self.loss(style_feature[0],reconstruct_style[0])
                    _ = self.compute_generator_loss(None, None, style_loss)
                    if i in j:
                        reconstructed_motion = reconstructed_motion["direction"].squeeze(0)
                        # root_info = test_data["root"].squeeze(0).transpose(0, 1)
                        #reconstructed_motion = torch.cat((reconstructed_motion, root_info), dim=-1).transpose(0,1).detach().cpu()

                        # content = content_labels[(test_data["content"][0, 0, :] == 1).nonzero(as_tuple=True)[0]]
                        content = "no_content"
                        save_bvh_from_network_output(
                            reconstructed_motion,
                            os.path.join(self.video_dir, "{}_{}_{}".format(epoch + 1, i, content))
                        )


                self.logger.info('Test: Epoch [{}/{}]| g_loss: {:.3f}| d_loss: {:.3f}| p_loss: {:.3f}| v_loss: {:.3f}| adv_loss: {:.3f}| sty_loss: {:.3f}'
                                 .format(epoch + 1, self.args.n_epoch,
                                    mean(self.loss_dict["generator_loss"]),
                                    mean(self.loss_dict["direction_loss"]),
                                    mean(self.loss_dict["position_loss"]),
                                    mean(self.loss_dict["velocity_loss"]),
                                    mean(self.loss_dict["adversarial_loss"]),
                                    mean(self.loss_dict["style_loss"])
                                         ))
                self.reset_loss_dict()
                self.model.train()

            if (epoch + 1) % self.args.save_freq == 0:
                torch.save(self.model.state_dict(), os.path.join(self.model_dir, "model_{}.pt".format(epoch + 1)))


if __name__ == "__main__":
    main()
