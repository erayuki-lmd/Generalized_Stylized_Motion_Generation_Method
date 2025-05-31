import argparse
import json
import os
import torch
import time
import numpy as np
import os
from statistics import mean

from models.model_pm import style_Encoder
from loader.loader_pm_B import MotionDataset_S
# from utils.other_postprocess import save_bvh_from_network_output, remove_fs
from utils.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
content_labels = ["walk", "run", "jump", "punch", "kick"]
style_labels = ["angry", "childlike", "depressed", "old", "proud", "sexy", "strutting"]

exp_dir = "RESULT/models/pm"

class ArgParserTrain(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # setup hyperparameters
        self.add_argument('--n_epoch', type=int, default=60)
        self.add_argument('--episode_length', type=int, default=24)
        self.add_argument("--seed", default=88, type=int)
        self.add_argument('--test_freq', type=int, default=5)
        self.add_argument('--log_freq', type=int, default=30)
        self.add_argument('--save_freq', type=int, default=5)
        self.add_argument('--style_num', type=int, default=7)
        self.add_argument('--content_num', type=int, default=5)
        self.add_argument("--data_path", default='./data/xia.npz')
        self.add_argument("--work_dir", default="./"+exp_dir+"/")
        self.add_argument("--load_dir", default=None, type=str)
        self.add_argument("--tag", default='', type=str)
        self.add_argument("--train_classifier", default=False, action='store_true')

        # training hyperparameters
        self.add_argument("--batch_size", default=32, type=int)
        self.add_argument("--w_reg", default=128, type=int)
        self.add_argument("--dis_lr", default=5e-6, type=float)
        self.add_argument("--gen_lr", default=1e-4, type=float)
        self.add_argument("--se_lr", default=5e-5, type=float)
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
    trainer.train()

class Trainer():
    def __init__(self, args):
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)

        exp_name = "style_pretrain"
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
        self.model_dir = make_dir(os.path.join(experiment_dir, 'model'))
        self.args = args
        self.logger = create_logger(experiment_dir)

        with open(os.path.join(experiment_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)
        
        self.dataset = MotionDataset_S(args)
        self.test_dataset = MotionDataset_S(args, subset_name="test")
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=args.batch_size, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=1, shuffle=False)

        self.model = style_Encoder(args, self.dataset.dim_dict).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.se_lr)
        self.loss = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

        self.reset_loss_dict()

    def train(self):
        for epoch in range(self.args.n_epoch):
            for i, train_data in enumerate(self.dataloader):
                embed_feature = self.model(train_data["direction"], train_data["position"], train_data["velocity"])
                style_loss = self.calc_triplet_loss(embed_feature,train_data["style"])
                if style_loss != None:
                    self.optimizer.zero_grad()
                    style_loss.backward()
                    self.optimizer.step()
                    self.loss_dict["style_loss"].append(style_loss.item())

                if (i + 1) % self.args.log_freq == 0:
                    self.logger.info(
                        'Train: Epoch [{}/{}], Step [{}/{}]| s_loss: {:.3f}'
                            .format(epoch + 1, self.args.n_epoch, i + 1, len(self.dataloader),
                                    mean(self.loss_dict["style_loss"])
                                    ))
                    self.reset_loss_dict()

            if (epoch + 1) % self.args.test_freq == 0:
                self.model.eval()

                for i, test_data in enumerate(self.testloader):
                    embed_feature = self.model(test_data["direction"], test_data["position"], test_data["velocity"])
                    style_loss = self.calc_triplet_loss(embed_feature,test_data["style"])
                    if style_loss != None:
                        self.loss_dict["style_loss"].append(style_loss.item())


                        self.logger.info('Test: Epoch [{}/{}]| g_loss: {:.3f}| r_loss: {:.3f}| p_loss: {:.3f}| v_loss: {:.3f}'
                                        .format(epoch + 1, self.args.n_epoch,
                                            mean(self.loss_dict["style_loss"])
                                            ))
                        self.reset_loss_dict()
                    self.model.train()

            if (epoch + 1) % self.args.save_freq == 0:
                torch.save(self.model.state_dict(), os.path.join(self.model_dir, "model_{}.pt".format(epoch + 1)))

    def calc_triplet_loss(self,feature,style):
        epsilon = 1e-7
        style = style.mean(dim=1)
        new_style = torch.zeros(len(style))

        plus_count = 0
        minus_count = 0

        plus_loss = 0
        minus_loss = 0

        for i in range(len(style)):
            if sum(style[i]) == 0:
                new_style[i] = 0
            else:
                new_style[i] = 1 + torch.argmax(style[i])

        for i in range(len(new_style)):
            for j in range(len(new_style)):
                if i > j:
                    if new_style[i] == new_style[j]:
                        plus_loss -= self.loss(feature[i],feature[j])
                        plus_count += 1
                    else:
                        minus_loss -= self.loss(feature[i],feature[j])
                        minus_count += 1

        if plus_count+minus_count == 0:
            output_loss = None
        elif plus_count == 0:
            output_loss = - minus_loss/float(minus_count)
        elif minus_count == 0:
            output_loss = plus_loss/float(plus_count)
        else:
            output_loss = plus_loss/float(plus_count) - minus_loss/float(minus_count)

        return output_loss


    def reset_loss_dict(self):
        self.loss_dict = {
            "style_loss": []
        }


if __name__ == "__main__":
    main()
