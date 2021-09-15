import os, sys
from datetime import date

import dill as pkl
import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Set double precision by default
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)


# TODO heritate from pytorch lightning and use it directly!

class Learner(pl.LightningModule):
    def __init__(self, observer, training_data, validation_data,
                 method="Autoencoder", batch_size=10, lr=1e-3,
                 optimizer=optim.Adam, optimizer_options=None,
                 scheduler=optim.lr_scheduler.ReduceLROnPlateau,
                 scheduler_options=None):
        super(Learner, self).__init__()
        # General parameters
        self.method = method
        self.model = observer
        self.model.to(self.device)

        # Data handling
        self.training_data = training_data
        self.validation_data = validation_data

        # Optimization
        self.batch_size = batch_size
        self.optim_lr = lr
        self.optimizer = optimizer
        self.optimizer_options = optimizer_options
        self.scheduler = scheduler
        self.scheduler_options = scheduler_options

        # Folder to save results
        i = 0
        while os.path.isdir(os.path.join(
                os.getcwd(), 'runs', str(date.today()), f"exp_{i}")):
            i += 1
        self.results_folder = os.path.join(
            os.getcwd(), 'runs', str(date.today()), f"exp_{i}")
        print(f'Results saved in in {self.results_folder}')

    def configure_optimizers(self):
        # https://pytorch-lightning.readthedocs.io/en/stable/common/optimizers.html
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/2976
        if self.optimizer_options is not None:
            optimizer_options = self.optimizer_options
        else:
            optimizer_options = {}
        parameters = self.model.parameters()
        optimizer = self.optimizer(parameters, self.optim_lr,
                                   **optimizer_options)
        if self.scheduler:
            if self.scheduler_options:
                scheduler_options = self.scheduler_options
            else:
                scheduler_options = {'mode': 'min', 'factor': 0.8,
                                     'patience': 10, 'threshold': 5e-2,
                                     'verbose': True}
            scheduler = {
                'scheduler': self.scheduler(optimizer, **scheduler_options),
                'monitor': 'train_loss'}
            return [optimizer], [scheduler]
        else:
            return optimizer

    def forward(self, batch):
        # Compute x_hat and/or z_hat depending on the method
        if self.method == "Autoencoder":
            x = batch  # .to(self.device)
            z_hat, x_hat = self.model(x)
            return z_hat, x_hat
        elif self.method == "T":
            x = batch[:, :self.model.dim_x]  # .to(self.device)
            z_hat = self.model(x)
            return z_hat
        elif self.method == "T_star":
            z = batch[:, self.model.dim_x:]  # .to(self.device)
            x_hat = self.model(z)
            return x_hat
        else:
            raise KeyError(f'Unknown method {self.method}')

    def train_dataloader(self):
        train_loader = DataLoader(
            self.training_data, batch_size=self.batch_size, shuffle=True)
        return train_loader

    def training_step(self, batch, batch_idx):
        # Compute transformation and loss depending on the method
        if self.method == "Autoencoder":
            z_hat, x_hat = self.forward(batch)
            loss, loss1, loss2 = self.model.loss(batch, x_hat, z_hat)
        elif self.method == "T":
            z = batch[:, self.model.dim_x:]
            z_hat = self.forward(batch)
            # mse = torch.nn.MSELoss()
            # loss = mse(z, z_hat)
            loss = self.model.loss(z, z_hat)
        elif self.method == "T_star":
            x = batch[:, :self.model.dim_x]
            x_hat = self.forward(batch)
            # mse = torch.nn.MSELoss()
            # loss = mse(x, x_hat)
            loss = self.model.loss(x, x_hat)
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        logs = {'train_loss': loss.detach()}
        return {'loss': loss, 'log': logs}

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.validation_data, batch_size=self.batch_size, shuffle=True)
        return val_dataloader

    def validation_step(self, batch, batch_idx):
        # Compute transformation and loss depending on the method
        with torch.no_grad():
            if self.method == "Autoencoder":
                z_hat, x_hat = self.forward(batch)
                loss, loss1, loss2 = self.model.loss(batch, x_hat, z_hat)
            elif self.method == "T":
                z = batch[:, self.model.dim_x:]
                z_hat = self.forward(batch)
                # mse = torch.nn.MSELoss()
                # loss = mse(z, z_hat)
                loss = self.model.loss(z, z_hat)
            elif self.method == "T_star":
                x = batch[:, :self.model.dim_x]
                x_hat = self.forward(batch)
                # mse = torch.nn.MSELoss()
                # loss = mse(x, x_hat)
                loss = self.model.loss(x, x_hat)
            self.log('val_loss', loss, on_step=True, prog_bar=True)
            logs = {'val_loss': loss.detach()}
            return {'loss': loss, 'log': logs}

    def save_results(self, checkpoint_path=None):
        with torch.no_grad():
            if checkpoint_path:
                checkpoint_model = torch.load(checkpoint_path)
                self.load_state_dict(checkpoint_model['state_dict'])



            with open(self.results_folder + '/model.pkl', 'wb') as f:
                pkl.dump(self.model, f, protocol=4)
            print(f'Saved model in {self.results_folder}')

# class Learner:
#     def __init__(self, observer, training_data, tensorboard=False,
#                  method="Autoencoder", num_epochs=50, batch_size=10, lr=1e-3,
#                  scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
#                  scheduler_options=None):
#
#         self.method = method
#
#         self.model = observer
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)
#         self.tensorboard = tensorboard
#
#         self.num_epochs = num_epochs
#
#         self.set_train_loader(training_data, batch_size=batch_size)
#         self.set_optimizer(lr)
#         if scheduler_options is None:
#             self.scheduler_options = {
#                 'mode': 'min', 'factor': 0.5, 'patience': 10, 'threshold': 1e-2,
#                 'verbose': True}
#         else:
#             self.scheduler_options = scheduler_options
#         self.scheduler = scheduler(self.optimizer, **self.scheduler_options)
#         # self.stopper = pl.callbacks.early_stopping.EarlyStopping(
#         #     monitor='val_loss', min_delta=0.5, patience=100,
#         #     verbose=False, mode='min')
#
#         if tensorboard:
#             self.set_tensorboard()
#
#     def set_train_loader(self, data, batch_size) -> None:
#         self.trainloader = utils.data.DataLoader(data, batch_size=batch_size,
#                                                  drop_last=True)
#
#     def set_optimizer(self, learning_rate) -> None:
#         self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
#
#     def set_tensorboard(self) -> None:
#         self.writer = SummaryWriter()
#
#     def forward(self, batch, step):
#
#         # Zero gradients
#         self.optimizer.zero_grad()
#
#         if self.method == "Autoencoder":
#             x = batch.to(self.device)
#         else:
#             x = batch[:, :self.model.dim_x].to(self.device)
#             z = batch[:, self.model.dim_x:].to(self.device)
#
#         if self.method == "Autoencoder":
#             # Predict
#             z_hat, x_hat = self.model(x)
#
#             # loss, loss1, loss2 = self.model.loss(x, x_hat, z_hat)  #???!!!
#             loss1, loss2, loss3 = self.model.loss(x, x_hat, z_hat)
#             loss = loss1 + loss2 + loss3
#         elif self.method == "T":
#             z_hat = self.model(x)
#
#             mse = torch.nn.MSELoss()
#             loss = mse(z, z_hat)
#         elif self.method == "T_star":
#             x_hat = self.model(z)
#             mse = torch.nn.MSELoss()
#             loss = mse(x, x_hat)
#
#         # Write loss to tensorboard
#         if self.tensorboard:
#             self.writer.add_scalars("Loss/train", {
#                 'loss': loss,
#             }, step)
#             self.writer.flush()
#
#         # Gradient step and optimize
#         loss.backward()
#         self.train_loss = loss
#         self.optimizer.step()
#
#     def train(self):
#         for epoch in range(self.num_epochs):
#
#             for i, batch in enumerate(self.trainloader, 0):
#
#                 step = i + (epoch*len(self.trainloader))
#
#                 self.forward(batch, step)
#
#             print('====> Epoch {}: LR {}, train_loss {}'.format(
#                 epoch + 1, self.optimizer.param_groups[0]["lr"]),
#                 self.train_loss)
#
#             # Adjust learning rate
#             self.scheduler.step(self.train_loss)
