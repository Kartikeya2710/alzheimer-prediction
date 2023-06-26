from agents.base import Agent
import shutil
import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from graphs.models.ResNeXt.resnext import ResNeXt
from dataset.alzheimer_loader import AlzheimerDataLoader
# from dataset.data_info import get_mean_and_std
from torchsummary import summary
from graphs.losses.cross_entropy import CrossEntropyLoss
from utils.metrics import AverageMetric
from dataset.annotations import create_annotations
from tensorboardX import SummaryWriter

class AlzheimerAgent(Agent):
    def __init__(self, config):
        super(AlzheimerAgent, self).__init__(config)

        self.config = config
        self.model_version = config.model_version
        self.model = globals()[self.config.model](self.config, self.model_version)
        self.logger.info(f"Using {self.config.model_name}...")

        create_annotations(self.config)
        self.logger.info(f"Created annotations at {self.config.annotations_dir}")
        self.data_loader = AlzheimerDataLoader(self.config)

        # mean, std = get_mean_and_std(self.data_loader)
        # self.logger.info(f"Mean and Standard Deviation of the training set: \n {mean} {std}")
        
        self.loss = CrossEntropyLoss()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience=5,
            mode='max',
            factor=0.5,
            threshold=0.02,
            verbose=True
        )

        self.is_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.is_cuda else "cpu")

        if self.is_cuda:
            torch.cuda.manual_seed_all(self.config.seed)
            self.logger.info("Operation will be on *****GPU-CUDA***** ")
        
        else:
            torch.manual_seed(self.config.seed)
            self.logger.info("Operation will be on *****CPU***** ")


        self.model = self.model.to(self.device)
        
        self.loss = self.loss.to(self.device)

        self.current_epoch = 0
        self.best_val_acc = 0
        self.current_iteration = 0

        self.load_checkpoint(self.config.checkpoint_file)
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir, comment='EfficientNet')

    def save_checkpoint(self, filename='checkpoint.pth.tar', is_best=0):
        """
        Saving the latest checkpoint of the training
        :param filename: filename which will contain the state
        :param is_best: flag is it is the best model
        :return:
        """
        state = {
            'epoch': self.current_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        # Save the state
        torch.save(state, self.config.checkpoint_dir + filename)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + filename,
                            self.config.checkpoint_dir + 'model_best.pth.tar')
            
    def load_checkpoint(self, filename):
        filename = self.config.checkpoint_dir + filename
        try:
            self.logger.info(f"Loading checkpoint '{filename}'")
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.logger.info(f"Checkpoint loaded successfully from '{self.config.checkpoint_dir}' at (epoch {checkpoint['epoch']})")

        except OSError as e:
            self.logger.info(f"No checkpoints found in '{self.config.checkpoint_dir}'. Skipping...")
            self.logger.info("**Loading model from scratch**")

    def run(self):
        """
        This function will the operator
        :return:
        """
        try:
            if self.config.mode == 'test':
                self.validate()
            else:
                self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training function, with per-epoch model saving
        """
        for epoch in range(self.current_epoch, self.config.epochs):
            self.train_one_epoch()

            if epoch % self.config.validate_every == 0:
                val_acc = self.validate()
                self.summary_writer.add_scalar("epoch/val_acc", val_acc, self.current_epoch)
                is_best = val_acc > self.best_val_acc

                self.best_val_acc = max(val_acc, self.best_val_acc)

                if epoch % self.config.save_every == 0:
                    self.save_checkpoint(is_best=is_best)

            self.current_epoch += 1

    def train_one_epoch(self):
        """
        One epoch training function
        """
        
        tqdm_batch = tqdm(iter(self.data_loader.train_loader), total=len(self.data_loader.train_loader),
                          desc="Epoch-{}-".format(self.current_epoch))
        # Set the model to be in training mode
        self.model.train()

        # Initialize your average metrics
        accuracy = AverageMetric()
        epoch_loss = AverageMetric()

        for (data, targets) in tqdm_batch:
            data = data.to(self.device)
            targets = targets.type(torch.LongTensor)
            targets = targets.to(self.device)
            # data is available as shape: (batch_size, channels, height, width)
            # output is of the form (batch_size, num_classes)
            outputs = self.model(data)
            loss = self.loss(outputs, targets)

            if np.isnan(float(loss.item())):
                raise ValueError('Loss is nan during training...')
            
            outputs = outputs.argmax(dim = 1)
            num_correct = (outputs == targets).sum()
            total_samples = targets.shape[0]
            batch_acc = num_correct/total_samples

            accuracy.update(batch_acc)
            epoch_loss.update(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.optimizer.step()
            

            tqdm_batch.set_description(f"Epoch [{self.current_epoch}/{self.config.epochs}]")
            tqdm_batch.set_postfix(loss = loss.item())

            self.summary_writer.add_scalar("epoch/loss", loss.item(), self.current_iteration)
            self.current_iteration += 1

        tqdm_batch.close()
        self.scheduler.step(accuracy.val)

        self.logger.info(
            "Training at epoch-" + str(self.current_epoch) + \
            " | " + "loss: " + str(epoch_loss.val) + \
            " | " + "batch_acc: " + str(accuracy.val)
        )
        
    def validate(self):
        num_correct = total_samples = 0

        self.model.eval()

        tqdm_batch = tqdm(
            self.data_loader.validation_loader, 
            total=len(self.data_loader.validation_loader),
            desc="Epoch-{}-".format(self.current_epoch)
        )

        with torch.no_grad():
            for (data, targets) in tqdm_batch:
                data = data.to(self.device)
                targets = targets.type(torch.LongTensor).to(self.device)
                # data is available as shape: (batch_size, channels, height, width)
                # output is of the form (batch_size, num_classes)
                outputs = self.model(data)
                outputs = outputs.argmax(dim = 1)

                num_correct += (outputs == targets).sum()
                total_samples += targets.shape[0]


        accuracy = (num_correct/total_samples)*100
        self.logger.info("Validation result at epoch-" + str(self.current_epoch) + " | " + "- Val Acc: " + str(accuracy))
        
        tqdm_batch.close()
        return accuracy