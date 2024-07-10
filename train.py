# Copyright (c) ByteDance, Inc. and its affiliates.
# Copyright (c) Chutong Meng
#
# This source code is licensed under the CC BY-NC license found in the
# LICENSE file in the root directory of this source tree.
# Based on AudioDec (https://github.com/facebookresearch/AudioDec)
import argparse
import logging

import os

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
)
logger = logging.getLogger("SpeechTokenizer_train")  # init logger before other modules

import random

import numpy as np
import torch
import torch.nn as nn
import yaml
import pickle
import itertools
from typing import Tuple
from torch.utils.data import DataLoader
from dataloader.collater import collate_fn
from speechtokenizer.model import SpeechTokenizer
from trainer.autoencoder import Trainer
from speechtokenizer.discriminator.discriminator import MultiPeriodDiscriminator
from speechtokenizer.discriminator.discriminator import MultiScaleDiscriminator
from speechtokenizer.discriminator.discriminator import MultiScaleSTFTDiscriminator
from losses.discriminator_loss import *
from losses.generator_loss import *

class TrainMain:
    def __init__(self, args):
        # Fix seed and make backends deterministic
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
            logger.info(f"device: cpu")
        else:
            self.device = torch.device('cuda:0')  # only supports single gpu for now
            logger.info(f"device: gpu")
            torch.cuda.manual_seed_all(args.seed)
            if args.disable_cudnn == "False":
                torch.backends.cudnn.benchmark = True

        # initialize config
        with open(args.config, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.config.update(vars(args))

        # initialize model folder
        expdir = os.path.join(args.exp_root, args.tag)
        os.makedirs(expdir, exist_ok=True)
        self.config["outdir"] = expdir

        # save config
        with open(os.path.join(expdir, "config.yml"), "w") as f:
            yaml.dump(self.config, f, Dumper=yaml.Dumper)
        for key, value in self.config.items():
            logger.info(f"{key} = {value}")

        # initialize attribute
        self.resume: str = args.resume
        self.data_loader = None
        self.model = None
        self.optimizer_model = None
        self.optimizer_disc = None
        self.scheduler_g = None
        self.scheduler_d = None
        self.trainer = None

        # initialize batch_length
        self.batch_length: int = self.config['batch_length']
        self.data_path: str = self.config['data']['path']

    def initialize_data_loader(self):
        train_set = self._build_dataset("train")
        valid_set = self._build_dataset("valid")
        
        
        logger.info(f"The number of training files = {len(train_set)}.")
        logger.info(f"The number of validation files = {len(valid_set)}.")
        dataset = {"train": train_set, "dev": valid_set}
        self._set_data_loader(dataset, collate_fn)


    def define_model_optimizer_scheduler(self):
        # model arch
        self.model = {
            "ST": SpeechTokenizer(self.config).cuda(),
            "msd": MultiScaleDiscriminator().cuda(),
            "mpd": MultiPeriodDiscriminator().cuda(),
            "stft_disc": MultiScaleSTFTDiscriminator(filters=32).cuda(),
            "fc": nn.Linear(self.config["model_params"]["semantic_dimension"], self.config["num_lang_class"]).cuda()
        }
        logger.info(f"Model Arch:\n{self.model['ST']}")
        
        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            self.model["ST"] = nn.DataParallel(self.model["ST"]) 
            self.model["msd"] = nn.DataParallel(self.model["msd"])
            self.model["mpd"] = nn.DataParallel(self.model["mpd"])
            self.model["stft_disc"] = nn.DataParallel(self.model["stft_disc"])
            self.model["fc"] = nn.DataParallel(self.model["fc"])
        # opt
        optimizer_class_model = getattr(
            torch.optim,
            self.config["model_optimizer_type"]
        )
        optimizer_class_disc = getattr(
            torch.optim,
            self.config["disc_optimizer_type"]
        )
        optimizer_class_fc = getattr(
            torch.optim,
            self.config["fc_optimizer_type"]
        )
        self.optimizer = {
            "ST": optimizer_class_model(
                self.model["ST"].parameters(),
                **self.config["model_optimizer_params"]
            ),
            "disc": optimizer_class_disc(
                itertools.chain(self.model["stft_disc"].parameters(),
                        self.model["msd"].parameters(), self.model["mpd"].parameters()),
                **self.config["disc_optimizer_params"]
            ),
            "fc": optimizer_class_fc(
                self.model["fc"].parameters(),
                **self.config["fc_optimizer_params"]
            )
        }

        # scheduler_g
        scheduler_class_g = getattr(
            torch.optim.lr_scheduler,
            self.config.get("model_scheduler_type", "StepLR"),
        )
        # scheduler_d
        scheduler_class_d = getattr(
            torch.optim.lr_scheduler,
            self.config.get("disc_scheduler_type", "StepLR"),
        )
        scheduler_class_fc = getattr(
            torch.optim.lr_scheduler,
            self.config.get("fc_scheduler_type","stepLR")
        )
        self.scheduler = {
            "ST": scheduler_class_g(
                optimizer=self.optimizer["ST"],
                **self.config["model_scheduler_params"]
            ),
            "disc": scheduler_class_d(
                optimizer=self.optimizer["disc"],
                **self.config["disc_scheduler_params"]
            ),
            "fc": scheduler_class_fc(
                optimizer = self.optimizer["fc"],
                **self.config["fc_scheduler_params"]
            )
        }


    def define_trainer(self):
        self.trainer = Trainer(
            steps=0,
            epochs=0,
            data_loader=self.data_loader,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            config=self.config,
            device=self.device
        )

    def initialize_model(self):
        initial = self.config.get("initial", "")
        if os.path.exists(self.resume):  # resume from trained model
            self.trainer.load_checkpoint(self.resume)
            logger.info(f"Successfully resumed from {self.resume}.")
        elif os.path.exists(initial):  # initial new model with the pre-trained model
            self.trainer.load_checkpoint(initial, load_only_params=True)
            logger.info(f"Successfully initialize parameters from {initial}.")
        else:
            logger.info("Train from scrach")

    def run(self):
        assert self.trainer is not None
        self.trainer: Trainer
        try:
            logger.info(f"The current training step: {self.trainer.steps}")
            self.trainer.train_max_steps = self.config["train_max_steps"]
            if not self.trainer._check_train_finish():
                self.trainer.run()
        finally:
            self.trainer.save_checkpoint(
                os.path.join(self.config["outdir"], f"checkpoint-{self.trainer.steps}steps.pkl")
            )
            logger.info(f"Successfully saved checkpoint @ {self.trainer.steps}steps.")

    def _build_dataset(self, subset: str):
        data_dir = os.path.join(
            self.data_path, self.config['data']['subset'][subset]
        )

        data = []
        label_counter = 0
        for language in self.config["language_list"]:
            #File paths for raw and teacher files
            raw_path = f'raw_{language}.pickle'
            raw_file = os.path.join(data_dir,raw_path)
            teacher_path = f'teacher_{language}.pickle'
            teacher_file = os.path.join(data_dir,teacher_path)

            #LID Label for this dataset
            lid_label = torch.zeros([self.config["num_lang_class"]],dtype=torch.long)
            lid_label[label_counter] = 1

            #Load raw and teacher datasets
            with open(raw_file,"rb") as file:
                raw = pickle.load(file)
            with open(teacher_file,"rb") as file:
                teacher = pickle.load(file)
            
            # Append dataset with label
            for i in range(len(raw)):
                data.append((raw[i],teacher[i],lid_label))

        return data

    def _set_data_loader(self, dataset, collater):
        self.data_loader = {
            "train": DataLoader(
                dataset=dataset["train"],
                shuffle=True,
                collate_fn=collater,
                batch_size=self.config["batch_size"]
            ),
            "dev": DataLoader(
                dataset=dataset["dev"],
                shuffle=False,
                collate_fn=collater,
                batch_size=self.config["batch_size"]
            ),
        }


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True,
        help="the path of config yaml file."
    )
    parser.add_argument(
        "--tag", type=str, required=True,
        help="the outputs will be saved to exp_root/tag/"
    )
    parser.add_argument(
        "--exp_root", type=str, default="exp"
    )
    parser.add_argument(
        "--resume", default="", type=str, nargs="?",
        help='checkpoint file path to resume training. (default="")',
    )
    parser.add_argument("--seed", default=1337, type=int)
    parser.add_argument("--disable_cudnn", choices=("True", "False"), default="False", help="Disable CUDNN")
    args = parser.parse_args()

    train_main = TrainMain(args)
    train_main.initialize_data_loader()
    train_main.define_model_optimizer_scheduler()
    train_main.define_trainer()
    train_main.initialize_model()
    train_main.run()


if __name__ == '__main__':
    train()
