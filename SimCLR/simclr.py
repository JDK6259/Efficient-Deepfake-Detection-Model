import os
import sys
import time
import neptune
import logging
import numpy as np

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint

torch.manual_seed(0)

class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

        api = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4OTRmMzMyMC0wNDA0LTRlZDAtYTg1Ni0zZTU3NDg3NGQ3YTYifQ=="
        neptune.init("dlthdus8450/simclr", api_token=api)
        temp = neptune.create_experiment(name=self.args.experiment, params=vars(self.args))
        experiment_num = str(temp).split('-')[-1][:-1]  # 모델저장경로 설정위한 Experiment Number받기
        neptune.append_tag(self.args.tag)
        self.save_path = os.path.join(self.args.model_path, experiment_num.zfill(3))
        
        self.writer = SummaryWriter(log_dir=self.save_path)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.INFO)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {not self.args.disable_cuda}.")

        best_loss = np.inf
        best_acc = 0
        best_epoch = 0
        early_stopping = 0
        start = time.time()
        for epoch_counter in range(self.args.epochs):
            train_loss = 0
            top1_acc = 0
            top5_acc = 0
            for images, _ in tqdm(train_loader):
                self.model.train()
                images = torch.cat(images, dim=0).to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)
                    train_loss+=loss

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                top1_acc+=top1[0]
                top5_acc+=top5[0]

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            top1_acc /= len(train_loader)
            top5_acc /= len(train_loader)
            train_loss /= len(train_loader)
            logging.info(f"Epoch: {epoch_counter}\tTrain Loss: {train_loss}\tTop1 accuracy: {top1_acc}")

            # self.model.eval()
            # with torch.no_grad():
            #     val_loss=0; val_acc=0
            #     for images, _ in test_loader:
            #         images = images.to(self.args.device)
            #         with autocast(enabled=self.args.fp16_precision):
            #             features = self.model(images)
            #             logits, labels = self.info_nce_loss(features)
            #             loss = self.criterion(logits, labels)
            #             val_loss+=loss

            #         top1, top5 = accuracy(logits, labels, topk=(1, 5))
            #         val_acc+=top1[0]
            #     val_acc /= len(test_loader.dataset)
            #     val_loss /= len(test_loader.dataset)

            neptune.log_metric('Train top1 acc', top1_acc)
            neptune.log_metric('Train top5 acc', top5_acc)
            neptune.log_metric('Train loss', loss)

            # neptune.log_metric('valid acc', val_acc)
            # neptune.log_metric('Valid loss', val_loss)

            # logging.info(f"Valid Loss: {val_loss}\tTop1 accuracy: {val_acc}")

            # Save models
            if train_loss < best_loss:
                early_stopping = 0
                best_epoch = epoch_counter
                best_loss = train_loss
                best_acc = top1_acc

                state_dict = self.model.module.state_dict() if self.args.multi_gpu else self.model.state_dict()

                # save model checkpoints
                checkpoint_name = 'best_model.pth.tar'
                save_checkpoint({
                    'epoch': epoch_counter,
                    'arch': self.args.arch,
                    'state_dict': state_dict,
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
                logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
            else:
                early_stopping += 1

            # Early Stopping
            if early_stopping == self.args.patience:
                break

        logging.info(f'\nBest Epoch:{best_epoch} | Loss:{best_loss:.4f} | Acc:{best_acc:.4f}')
        end = time.time()
        logging.info(f'Total Process time:{(end - start) / 60:.3f}Minute')
        neptune.stop() 

        state_dict = self.model.module.state_dict() if self.args.multi_gpu else self.model.state_dict()
        # save model checkpoints
        checkpoint_name = 'last_epoch.pth.tar'
        save_checkpoint({
            'epoch': epoch_counter,
            'arch': self.args.arch,
            'state_dict': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"-------------------------- Add model at last epoch{epoch_counter} --------------------------")
        logging.info("Training has finished.")
