import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


class Learner:
    def __init__(self, observer, training_data, tensorboard=False):

        self.model = observer
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.tensorboard = tensorboard

        self.num_epochs = 20

        self.set_train_loader(training_data, 10)
        self.set_optimizer(0.001)
        self.set_scheduler(self.optimizer, [5, 10, 20])

        if tensorboard:
            self.set_tensorboard()

    def set_train_loader(self, data, batch_size) -> None:
        self.trainloader = utils.data.DataLoader(data, batch_size=batch_size,
                                                 drop_last=True)

    def set_optimizer(self, learning_rate) -> None:
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def set_scheduler(self, optimizer, milestones) -> None:
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)

    def set_tensorboard(self) -> None:
        self.writer = SummaryWriter()

    def forward(self, x, step):

        # Zero gradients
        self.optimizer.zero_grad()

        # Predict
        z_hat, x_hat = self.model(x)

        loss, loss1, loss2 = self.model.loss(x, x_hat, z_hat)

        # Write loss to tensorboard
        if self.tensorboard:
            self.writer.add_scalars("Loss/train", {
                'loss': loss,
                'loss1': loss1,
                'loss2': loss2,
            }, step)
            self.writer.flush()

        # Gradient step and optimize
        loss.backward()

        self.optimizer.step()

    def train(self):
        for epoch in range(self.num_epochs):

            for i, batch in enumerate(self.trainloader, 0):

                x = batch.to(self.device)

                step = i + (epoch*len(self.trainloader))

                self.forward(x, step)

            print('====> Epoch: {} done! LR: {}'.format(epoch + 1, self.optimizer.param_groups[0]["lr"]))

            # Adjust learning rate
            self.scheduler.step()
