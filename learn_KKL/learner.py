import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


class Learner:
    def __init__(self, observer, training_data, tensorboard=False,
                 method="Autoencoder", num_epochs=50, batch_size=10, lr=1e-3,
                 scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
                 scheduler_options=None):

        self.method = method

        self.model = observer
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.tensorboard = tensorboard

        self.num_epochs = num_epochs

        self.set_train_loader(training_data, batch_size=batch_size)
        self.set_optimizer(lr)
        if scheduler_options is None:
            self.scheduler_options = {
                'mode': 'min', 'factor': 0.8, 'patience': 10, 'threshold': 0.1,
                'verbose': True}
        else:
            self.scheduler_options = scheduler_options
        self.optim_scheduler = scheduler(self.optimizer, **scheduler_options)
        # self.optim_stopper = pl.callbacks.early_stopping.EarlyStopping(
        #     monitor='val_loss', min_delta=0.5, patience=100,
        #     verbose=False, mode='min')

        if tensorboard:
            self.set_tensorboard()

    def set_train_loader(self, data, batch_size) -> None:
        self.trainloader = utils.data.DataLoader(data, batch_size=batch_size,
                                                 drop_last=True)

    def set_optimizer(self, learning_rate) -> None:
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def set_tensorboard(self) -> None:
        self.writer = SummaryWriter()

    def forward(self, batch, step):

        # Zero gradients
        self.optimizer.zero_grad()

        if self.method == "Autoencoder":
            x = batch.to(self.device)
        else:
            x = batch[:, :self.model.dim_x].to(self.device)
            z = batch[:, self.model.dim_x:].to(self.device)

        if self.method == "Autoencoder":
            # Predict
            z_hat, x_hat = self.model(x)

            loss, loss1, loss2 = self.model.loss(x, x_hat, z_hat)
        elif self.method == "T":
            z_hat = self.model(x)

            mse = torch.nn.MSELoss()
            loss = mse(z, z_hat)
        elif self.method == "T_star":
            x_hat = self.model(z)
            mse = torch.nn.MSELoss()
            loss = mse(x, x_hat)

        # Write loss to tensorboard
        if self.tensorboard:
            self.writer.add_scalars("Loss/train", {
                'loss': loss,
            }, step)
            self.writer.flush()

        # Gradient step and optimize
        loss.backward()

        self.optimizer.step()

    def train(self):
        for epoch in range(self.num_epochs):

            for i, batch in enumerate(self.trainloader, 0):

                step = i + (epoch*len(self.trainloader))

                self.forward(batch, step)

            print('====> Epoch: {} done! LR: {}'.format(epoch + 1, self.optimizer.param_groups[0]["lr"]))

            # Adjust learning rate
            self.scheduler.step()
