# Copyright (c) ByteDance, Inc. and its affiliates.
# Copyright (c) Chutong Meng
#
# This source code is licensed under the CC BY-NC license found in the
# LICENSE file in the root directory of this source tree.
# Based on AudioDec (https://github.com/facebookresearch/AudioDec)

import logging
import os
from collections import defaultdict

import torch
import itertools
from tensorboardX import SummaryWriter
from tqdm import tqdm
import GPUtil
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from losses.discriminator_loss import discriminator_loss
from losses.discriminator_loss import feature_loss
from losses.discriminator_loss import adversarial_g_loss

logger = logging.getLogger("Trainer")


class Trainer:
    def __init__(
            self,
            steps: int,
            epochs: int,
            data_loader: dict,
            model: dict,
            criterion_g: dict,
            optimizer: dict,
            scheduler: dict,
            config: dict,
            device=torch.device("cpu"),
    ):
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.model = model
        self.criterion_g = criterion_g
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.writer = SummaryWriter(config["outdir"])
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)
        self.train_max_steps = config.get("train_max_steps", 0)
    
    
    def print_gpu_usage(self):
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            logger.info(f"GPU ID {gpu.id}: {gpu.memoryUsed} MB / {gpu.memoryTotal} MB ({gpu.memoryUtil * 100}%)")



    def _train_step(self, batch):
        """Single step of training."""
        mode = "train"
        x,x_teacher = batch
        x = x.cuda()
        x_teacher = x_teacher.cuda()
        
        for optmizer_idx in [0,1]:
            y_, commit_loss, z_q= self.model["ST"](x)
            if(optmizer_idx==0):

                codec_loss = 0.0

                # MPD
                x_df_r, x_df_g, fmap_f_r, fmap_f_g  = self.model["mpd"](x, y_.detach())

                # MSD
                x_ds_r, x_ds_g, fmap_s_r, fmap_s_g = self.model["msd"](x, y_.detach())

                #MSTFT
                x_stft_r, fmap_stftd_r  = self.model["stft_disc"](x)
                x_stft_gen, fmap_stftd_g = self.model["stft_disc"](y_.detach())

                fmap_discriminator = fmap_f_g, fmap_f_r, fmap_s_g, fmap_s_r, fmap_stftd_g, fmap_stftd_r
                x_gen = x_df_g,x_ds_g,x_stft_gen

                codec_loss += self._distil_loss(z_q,x_teacher)
                codec_loss += self._commit_loss(commit_loss, mode=mode)
                codec_loss += self._reconstruct_loss(y_[:,:,:x.shape[2]], x, mode=mode)
                codec_loss += self._feature_loss(fmap_discriminator,mode=mode)
                codec_loss += self._adversarial_g_loss(x_gen, mode=mode)
                self._record_loss("total_codec_loss", codec_loss, mode=mode)
                self._update_Speechtokenizer(codec_loss)
            else:
                disc_loss = 0.0
                # MPD
                x_df_r, x_df_g, _,_ = self.model["mpd"](x, y_.detach())

                # MSD
                x_ds_r, x_ds_g, _,_ = self.model["msd"](x, y_.detach())

                #MSTFT
                x_stft_r,_  = self.model["stft_disc"](x)
                x_stft_gen, _ = self.model["stft_disc"](y_.detach())

                x_discriminator = x_df_r,x_df_g,x_ds_r,x_ds_g,x_stft_r,x_stft_gen
                disc_loss += self._discriminator_loss(x_discriminator, mode=mode)
                self._record_loss("total_discriminator_loss", disc_loss, mode=mode)
                self._update_Discriminator(disc_loss)

        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    @torch.no_grad()
    def _eval_step(self, batch):
        """Single step of evaluation."""
        mode = "eval"
        x,x_teacher = batch
        x = x.cuda()
        x_teacher = x_teacher.cuda()
        
        for optmizer_idx in [0,1]:
            y_, commit_loss, z_q= self.model["ST"](x)
            if(optmizer_idx==0):

                codec_loss = 0.0

                # MPD
                x_df_r, x_df_g, fmap_f_r, fmap_f_g  = self.model["mpd"](x, y_.detach())

                # MSD
                x_ds_r, x_ds_g, fmap_s_r, fmap_s_g = self.model["msd"](x, y_.detach())

                #MSTFT
                x_stft_r, fmap_stftd_r  = self.model["stft_disc"](x)
                x_stft_gen, fmap_stftd_g = self.model["stft_disc"](y_.detach())

                fmap_discriminator = fmap_f_g, fmap_f_r, fmap_s_g, fmap_s_r, fmap_stftd_g, fmap_stftd_r
                x_gen = x_df_g,x_ds_g,x_stft_gen

                codec_loss += self._distil_loss(z_q,x_teacher)
                codec_loss += self._commit_loss(commit_loss, mode=mode)
                codec_loss += self._reconstruct_loss(y_[:,:,:x.shape[2]], x, mode=mode)
                codec_loss += self._feature_loss(fmap_discriminator,mode=mode)
                codec_loss += self._adversarial_g_loss(x_gen, mode=mode)
                self._record_loss("valid_codec_loss", codec_loss, mode=mode)

            else:
                disc_loss = 0.0
                # MPD
                x_df_r, x_df_g, _,_ = self.model["mpd"](x, y_.detach())

                # MSD
                x_ds_r, x_ds_g, _,_ = self.model["msd"](x, y_.detach())

                #MSTFT
                x_stft_r,_  = self.model["stft_disc"](x)
                x_stft_gen, _ = self.model["stft_disc"](y_.detach())

                x_discriminator = x_df_r,x_df_g,x_ds_r,x_ds_g,x_stft_r,x_stft_gen
                disc_loss += self._discriminator_loss(x_discriminator, mode=mode)
                self._record_loss("valid_discriminator_loss", disc_loss, mode=mode)

    def run(self):
        """Run training."""
        self.finish_train = False
        self.tqdm = tqdm(
            initial=self.steps, total=self.train_max_steps, desc="[train]"
        )
        while True:
            self._train_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        self.tqdm.close()
        logger.info("Finished training.")

    def save_checkpoint(self, checkpoint_path: str):
        state_dict = {
            "model": {
                "Speechtokenizer": self.model["ST"].state_dict()
            },
            "optimizer": {
                "Speechtokenizer": self.optimizer["ST"].state_dict(),
            },
            "scheduler": {
                "Speechtokenizer": self.scheduler["ST"].state_dict(),
            },
            "steps": self.steps,
            "epochs": self.epochs,
        }

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(
            self,
            checkpoint_path: str,
            strict: bool = True,
            load_only_params: bool = False
    ):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.model["ST"].load_state_dict(
            state_dict["model"]["Speechtokenizer"], strict=strict
        )

        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer["ST"].load_state_dict(
                state_dict["optimizer"]["Speechtokenizer"]
            )
            self.scheduler["Speechtokenizer"].load_state_dict(
                state_dict["scheduler"]["Speechtokenizer"]
            )

    def _train_epoch(self):
        """One epoch of training."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check interval
            self._check_log_interval()
            self._check_eval_interval()
            self._check_save_interval()

            # check whether training is finished
            if self.finish_train:
                return

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        if train_steps_per_epoch > 200:
            logger.info(
                f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
                f"({self.train_steps_per_epoch} steps per epoch)."
            )

    def _eval_epoch(self):
        """One epoch of evaluation."""
        logger.info(f"(Steps: {self.steps}) Start evaluation.")
        # change mode
        for key in self.model.keys():
            self.model[key].eval()

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(
                tqdm(self.data_loader["dev"], desc="[eval]"), 1
        ):
            # eval one step
            self._eval_step(batch)

        logger.info(
            f"(Steps: {self.steps}) Finished evaluation "
            f"({eval_steps_per_epoch} steps per epoch)."
        )

        # average loss
        for key in self.total_eval_loss.keys():
            self.total_eval_loss[key] /= eval_steps_per_epoch
            logger.info(
                f"(Steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}."
            )

        # record
        self._write_to_tensorboard(self.total_eval_loss)

        # reset
        self.total_eval_loss = defaultdict(float)

        # restore mode
        for key in self.model.keys():
            self.model[key].train()

    def _reconstruct_loss(self, predict_y, natural_y, mode='train'):
        """Metric losses."""
        reconstruct_loss = 0.0

        time_reconstruct_loss = self.criterion_g["time_reconstruct_loss"](predict_y, natural_y)
        time_reconstruct_loss *= self.config["loss_params"]["lambda_time_reconstruct_loss"]
        freq_reconstruct_loss = self.criterion_g["freq_reconstruct_loss"](predict_y, natural_y)
        freq_reconstruct_loss *= self.config["loss_params"]["lambda_freq_reconstruct_loss"]
        repr_reconstruct_loss = time_reconstruct_loss+ freq_reconstruct_loss
        self._record_loss("reconstruct_loss", repr_reconstruct_loss, mode=mode)
        reconstruct_loss += repr_reconstruct_loss

        return reconstruct_loss

    def _update_Speechtokenizer(self, repr_loss):
        """Update generator."""
        self.optimizer["ST"].zero_grad()
        repr_loss.backward()
        if self.config["grad_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model["ST"].parameters(),
                self.config["grad_norm"],
            )
        self.optimizer["ST"].step()
        self.scheduler["ST"].step()

    def _update_Discriminator(self, repr_loss):
        """Update generator."""
        self.optimizer["disc"].zero_grad()
        repr_loss.backward()
        if self.config["grad_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                itertools.chain(self.model["stft_disc"].parameters(),
                        self.model["msd"].parameters(), self.model["mpd"].parameters()),
                self.config["grad_norm"],
            )
        self.optimizer["disc"].step()
        self.scheduler["disc"].step()

    def _record_loss(self, name: str, loss, mode='train'):
        """Record loss."""
        if torch.is_tensor(loss):
            loss = loss.item()

        if mode == 'train':
            self.total_train_loss[f"train/{name}"] += loss
        elif mode == 'eval':
            self.total_eval_loss[f"eval/{name}"] += loss
        else:
            raise NotImplementedError(f"Mode ({mode}) is not supported!")

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self):
        if self.steps and (self.steps % self.config["save_interval_steps"] == 0):
            self.save_checkpoint(
                os.path.join(self.config["outdir"], f"checkpoint-{self.steps}steps.pkl")
            )
            logger.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_eval_interval(self):
        if self.steps % self.config["eval_interval_steps"] == 0:
            self._eval_epoch()

    def _check_log_interval(self):
        if self.steps % self.config["log_interval_steps"] == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config["log_interval_steps"]
                logger.info(
                    f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}."
                )
            self._write_to_tensorboard(self.total_train_loss)

            # reset
            self.total_train_loss = defaultdict(float)

    def _check_train_finish(self):
        if self.steps >= self.train_max_steps:
            self.finish_train = True
        else:
            self.finish_train = False
        return self.finish_train

    def _commit_loss(self, commit_loss, label=None, mode='train'):
        if label:
            name = f"{mode}/commit_loss_{label}"
        else:
            name = f"{mode}/commit_loss"
        commit_loss = torch.sum(commit_loss)
        commit_loss *= self.config["loss_params"]["lambda_commit_loss"]
        self._record_loss(name, commit_loss, mode=mode)

        return commit_loss
    
    def _distil_loss(self,x,x_teacher, mode='train'):
        
        distil_loss = 0.0

        repr_distillation_loss = self.criterion_g["distill_loss"](x, x_teacher)
        repr_distillation_loss *= self.config["loss_params"]["lambda_repr_distillation_loss"]
        self._record_loss("distillation_loss", repr_distillation_loss, mode=mode)
        distil_loss += repr_distillation_loss

        return distil_loss
        

    def _discriminator_loss(self,x_discriminator, mode="train"):
        x_df_r,x_df_g,x_ds_r,x_ds_g,x_stft_r,x_stft_gen = x_discriminator
        loss_disc_all = 0.0

        loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
        x_df_r, x_df_g)

        loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
            x_ds_r, x_ds_g)

        loss_disc_stft, losses_disc_stft_r, losses_disc_stft_g = discriminator_loss(
            x_stft_r, x_stft_gen)

        loss_disc_all = loss_disc_s + loss_disc_f + loss_disc_stft
        loss_disc_all *= self.config["loss_params"]["lambda_disriminator_loss"]
        self._record_loss("discriminator_loss", loss_disc_all, mode=mode)

        return loss_disc_all
    
    def _adversarial_g_loss(self,x_gen, mode="train"):
        x_df_g,x_ds_g,x_stft_gen = x_gen
        loss_gen_all = 0.0

        loss_gen_f = adversarial_g_loss(x_df_g)

        loss_gen_s = adversarial_g_loss(x_ds_g)

        loss_gen_stft = adversarial_g_loss(x_stft_gen)

        loss_gen_all = loss_gen_f + loss_gen_s + loss_gen_stft
        loss_gen_all *= self.config["loss_params"]["lambda_generator_loss"]
        self._record_loss("generator_loss", loss_gen_all, mode=mode)

        return loss_gen_all

    def _feature_loss(self,fmap_discriminator, mode='train'):
        
        feat_loss = 0.0
        fmap_f_g, fmap_f_r, fmap_s_g, fmap_s_r, fmap_stftd_g, fmap_stftd_r = fmap_discriminator
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_fm_stft = feature_loss(fmap_stftd_r, fmap_stftd_g)
        feat_loss = loss_fm_f + loss_fm_s + loss_fm_stft
        feat_loss *= self.config["loss_params"]["lambda_feature_loss"]
        self._record_loss("feature_loss", feat_loss, mode=mode)

        return feat_loss

        
