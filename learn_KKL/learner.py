import os
import sys

import dill as pkl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.optim as optim
from smt.sampling_methods import LHS
from torch.utils.data import DataLoader
from scipy.interpolate import interp2d


from .utils import RMSE, StandardScaler

# Set double precision by default
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)


class Learner(pl.LightningModule):
    """
    Learner class based on pytorch-lightning
    https://pytorch-lightning.readthedocs.io/en/stable/starter/new-project.html


    Params
    ----------
    observer:
        KKL observer model.

    system:
        Dynamical system studied. Needed for simulation of test trajectories,
        so needs a simulate function.

    training_data:
        Training data.

    validation_data :
        Validation data.

    method: str
        Method for training the model's encoder, decoder, or both jointly.
        Can be 'T', 'T_star', 'Autoencoder' respectively.

    batch_size: int
        Size of the minibatches.

    lr: float
        Learning rate used for optimization.

    optimizer:
        Optimizer from torch.optim.

    optimizer_options:
        Options for the optimizer.

    scheduler:
        Scheduler from torch.optim.lr_scheduler.

    scheduler_options:
        Options for the scheduler.




    Attributes
    ----------

    results_folder: str
        Path of the folder for saving all results.

    scaler_x:
        Scaler object for x.

    scaler_z:
        Scaler object for z.

    """

    def __init__(self, observer, system, training_data, validation_data,
                 method="Autoencoder", batch_size=10, lr=1e-3,
                 optimizer=optim.Adam, optimizer_options=None,
                 scheduler=optim.lr_scheduler.ReduceLROnPlateau,
                 scheduler_options=None):
        super(Learner, self).__init__()
        # General parameters
        self.system = system
        self.method = method
        self.model = observer
        self.model.to(self.device)

        # Data handling
        self.training_data = training_data
        self.validation_data = validation_data
        if self.method == 'Autoencoder':
            self.scaler_x = StandardScaler(self.training_data)
            self.scaler_z = None
        else:
            self.scaler_x = StandardScaler(
                self.training_data[:, :self.model.dim_x])
            self.scaler_z = StandardScaler(
                self.training_data[:, self.model.dim_x:])
        self.model.set_scalers(scaler_x=self.scaler_x, scaler_z=self.scaler_z)

        # Optimization
        self.batch_size = batch_size
        self.optim_lr = lr
        self.optimizer = optimizer
        self.optimizer_options = optimizer_options
        self.scheduler = scheduler
        self.scheduler_options = scheduler_options

        # Folder to save results
        i = 0
        params = os.path.join(os.getcwd(), 'runs', self.model.method)
        if self.model.method == 'Supervised':
            params += '/' + self.method
        while os.path.isdir(os.path.join(params, f"exp_{i}")):
            i += 1
        self.results_folder = os.path.join(params, f"exp_{i}")
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
            z_hat, x_hat = self.model(self.method, x)
            return z_hat, x_hat
        elif self.method == "T":
            x = batch[:, :self.model.dim_x]  # .to(self.device)
            z_hat = self.model(self.method, x)
            return z_hat
        elif self.method == "T_star":
            z = batch[:, self.model.dim_x:]  # .to(self.device)
            x_hat = self.model(self.method, z)
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
            loss, loss1, loss2 = self.model.loss(self.method, batch, x_hat,
                                                 z_hat)
            self.log('train_loss1', loss1, on_step=True, prog_bar=False,
                     logger=True)
            self.log('train_loss2', loss2, on_step=True, prog_bar=False,
                     logger=True)
        elif self.method == "T":
            z = batch[:, self.model.dim_x:]
            z_hat = self.forward(batch)
            # mse = torch.nn.MSELoss()
            # loss = mse(z, z_hat)
            loss = self.model.loss(self.method, z, z_hat)
        elif self.method == "T_star":
            x = batch[:, :self.model.dim_x]
            x_hat = self.forward(batch)
            # mse = torch.nn.MSELoss()
            # loss = mse(x, x_hat)
            loss = self.model.loss(self.method, x, x_hat)
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
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
                loss, loss1, loss2 = self.model.loss(self.method, batch,
                                                     x_hat, z_hat)
                self.log('val_loss1', loss1, on_step=True, prog_bar=False,
                         logger=True)
                self.log('val_loss2', loss2, on_step=True, prog_bar=False,
                         logger=True)
            elif self.method == "T":
                z = batch[:, self.model.dim_x:]
                z_hat = self.forward(batch)
                # mse = torch.nn.MSELoss()
                # loss = mse(z, z_hat)
                loss = self.model.loss(self.method, z, z_hat)
            elif self.method == "T_star":
                x = batch[:, :self.model.dim_x]
                x_hat = self.forward(batch)
                # mse = torch.nn.MSELoss()
                # loss = mse(x, x_hat)
                loss = self.model.loss(self.method, x, x_hat)
            self.log('val_loss', loss, on_step=True, prog_bar=True, logger=True)
            logs = {'val_loss': loss.detach()}
            return {'loss': loss, 'log': logs}

    def save_results(self, limits: np.array, nb_trajs=10, tsim=(0, 60),
                     dt=1e-2, checkpoint_path=None, verbose=False):
        """
        Save the model, the training and validation data. Also saving several
        metrics used for evaluating the results: heatmap of estimation error
        and estimation of several test trajectories.

        Parameters
        ----------
        limits: np.array
            Array for the limits of all axes, used for sampling the heatmap
            and the initial conditions of the test trajectories.
            Form np.array([[min_1, ..., min_,n], [max_1, ..., max_n]]).

        nb_trajs: int
            Number of test trajectories.

        tsim: tuple
            Length of simulations for the test trajectories.

        dt: int
            Sampling time of the simulations for the test trajectories.

        checkpoint_path: str
            Path to the checkpoint from which to retrieve the best obtained
            model.

        verbose: bool
            Whether to show the plots or just save them.
        """
        with torch.no_grad():
            if checkpoint_path:
                checkpoint_model = torch.load(checkpoint_path)
                self.load_state_dict(checkpoint_model['state_dict'])

            # Save training and validation data
            idx = np.random.choice(np.arange(len(self.training_data)),
                                   size=(10000,))  # subsampling for plots
            filename = 'training_data.csv'
            file = pd.DataFrame(self.training_data.cpu().numpy())
            file.to_csv(os.path.join(self.results_folder, filename),
                        header=False)
            training_data = self.training_data[idx]
            for i in range(1, training_data.shape[1]):
                name = 'training_data' + str(i) + '.pdf'
                plt.scatter(training_data[:, i - 1].cpu(),
                            training_data[:, i].cpu())
                plt.title('Training data')
                plt.xlabel(rf'x_{{i}}')
                plt.ylabel(rf'x_{{j}}')
                plt.savefig(os.path.join(self.results_folder, name),
                            bbox_inches='tight')
                if verbose:
                    plt.show()
                plt.close('all')
            filename = 'validation_data.csv'
            file = pd.DataFrame(self.validation_data.cpu().numpy())
            file.to_csv(os.path.join(self.results_folder, filename),
                        header=False)

            # Save specifications and model
            specs_file = os.path.join(self.results_folder, 'Specifications.txt')
            with open(specs_file, 'w') as f:
                print(sys.argv[0], file=f)
                for key, val in vars(self).items():
                    print(key, ': ', val, file=f)

            with open(self.results_folder + '/model.pkl', 'wb') as f:
                pkl.dump(self.model, f, protocol=4)
            print(f'Saved model in {self.results_folder}')

            # No control theoretic evaluation of the observer with only T
            if self.method == 'T':
                return 0

            # Heatmap of RMSE(x, x_hat) with T_star
            num_samples = 50000
            heatmap_data = self.model.generate_data_svl(limits, num_samples)
            x = heatmap_data[:, :self.model.dim_x]
            z = heatmap_data[:, self.model.dim_x:]
            x_hat = self.model('T_star', z)
            error = RMSE(x, x_hat, dim=1)
            print(x.shape, error.shape)
            for i in range(1, x.shape[1]):
                # https://stackoverflow.com/questions/37822925/how-to-smooth-by-interpolation-when-using-pcolormesh
                name = 'RMSE_heatmap' + str(i) + '.pdf'
                plt.scatter(x[:, i - 1], x[:, i], cmap='jet',
                            c=np.log(error.numpy()))
                cbar = plt.colorbar()
                cbar.set_label('Log estimation error')
                # print(x[:, i - 1:i + 1].shape)
                # plt.imshow(x[:, i - 1:i + 1], cmap='jet',
                #            c=np.log(error.numpy()))
                # plt.pcolormesh(x[:, i - 1], x[:, i], np.log(error),
                #                cmap='jet', shading='gouraud')
                # cbar = plt.colorbar()
                # cbar.set_label('Log estimation error')
                # cbar = plt.colorbar()
                cbar.set_label('Log estimation error')
                plt.title('Log of RMSE between ' + r'$x$' + ' and '
                          + r'$\hat{'r'x}$')
                plt.xlabel(rf'$x_{i}$')
                plt.ylabel(rf'$x_{i + 1}$')
                plt.legend()
                plt.savefig(os.path.join(self.results_folder, name),
                            bbox_inches='tight')
                if verbose:
                    plt.show()
                plt.close('all')

            # Estimation over the test trajectories with T_star
            sampling = LHS(xlimits=limits)
            trajs_init = torch.as_tensor(sampling(nb_trajs)).T
            traj_folder = os.path.join(self.results_folder, 'Test_trajectories')
            tq, simulation = self.system.simulate(trajs_init, tsim, dt)
            measurement = torch.transpose(
                self.model.h(torch.transpose(simulation, 0, 1)), 0, 1)
            # Save these test trajectories
            os.makedirs(traj_folder, exist_ok=True)
            traj_error = 0.
            for i in range(nb_trajs):
                # TODO run predictions in parallel for all test trajectories!!!
                # Need to figure out how to interpolate y in parallel for all
                # trajectories!!!
                y = torch.cat((tq.unsqueeze(1), measurement[..., i]), dim=1)
                estimation = self.model.predict(y, tsim, dt)
                traj_error += RMSE(simulation[..., i], estimation)

                current_traj_folder = os.path.join(traj_folder, f'Traj_{i}')
                os.makedirs(current_traj_folder, exist_ok=True)
                filename = f'True_traj_{i}.csv'
                file = pd.DataFrame(simulation[..., i].cpu().numpy())
                file.to_csv(os.path.join(current_traj_folder, filename),
                            header=False)
                filename = f'Estimated_traj_{i}.csv'
                file = pd.DataFrame(estimation.cpu().numpy())
                file.to_csv(os.path.join(current_traj_folder, filename),
                            header=False)
                for j in range(simulation.shape[1]):
                    name = 'Traj' + str(j) + '.pdf'
                    plt.plot(tq, simulation[:, j, i].detach().numpy(),
                             label=rf'$x_{j + 1}$')
                    plt.plot(tq, estimation[:, j].detach().numpy(),
                             label=rf'$\hat{{x}}_{j + 1}$')
                    plt.legend()
                    plt.xlabel(rf'$t$')
                    plt.ylabel(rf'$x_{j + 1}$')
                    plt.savefig(os.path.join(current_traj_folder, name),
                                bbox_inches='tight')
                    if verbose:
                        plt.show()
                    plt.close('all')
            filename = 'RMSE_traj.txt'
            with open(os.path.join(traj_folder, filename), 'w') as f:
                print(traj_error, file=f)
