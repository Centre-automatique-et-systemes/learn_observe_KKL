import os
import sys

import dill as pkl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sb
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .utils import RMSE, StandardScaler

# To avoid Type 3 fonts for submission https://tex.stackexchange.com/questions/18687/how-to-generate-pdf-without-any-type3-fonts
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

sb.set_style("whitegrid")

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
        Can be 'T', 'T_star', 'Autoencoder', 'Autoencoder_jointly' respectively.

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

    def __init__(
            self,
            observer,
            system,
            training_data,
            validation_data,
            axe,
            method="Autoencoder",
            batch_size=10,
            lr=1e-3,
            optimizer=optim.Adam,
            optimizer_options=None,
            scheduler=optim.lr_scheduler.ReduceLROnPlateau,
            scheduler_options=None,
    ):
        super(Learner, self).__init__()
        # General parameters
        self.system = system
        self.method = method
        self.model = observer
        self.model.to(self.device)
        self.axe = axe

        # Data handling
        self.training_data = training_data
        self.validation_data = validation_data
        self.num_train = len(self.training_data)
        self.num_val = len(self.validation_data)
        # Indices of x and z in training data: same in and out
        self.x_idx_in = [i for i in range(self.model.dim_x)]
        self.x_idx_out = [i for i in range(self.model.dim_x)]
        self.z_idx_in = [i for i in range(self.model.dim_x, self.model.dim_x
                                          + self.model.dim_z)]
        self.z_idx_out = [i for i in range(self.model.dim_x, self.model.dim_x
                                           + self.model.dim_z)]

        # Optimization
        self.batch_size = batch_size
        self.optim_lr = lr
        self.optimizer = optimizer
        self.optimizer_options = optimizer_options
        self.scheduler = scheduler
        self.scheduler_options = scheduler_options
        self.train_loss = torch.zeros((0, 1))
        self.val_loss = torch.zeros((0, 1))

        # If method == 'T', freeze T_star and vice versa
        if self.method == "T":
            self.model.encoder.unfreeze()
            self.model.decoder.freeze()
        elif self.method == "T_star":
            self.model.encoder.freeze()
            self.model.decoder.unfreeze()

        # Folder to save results
        i = 0
        params = os.path.join(os.getcwd(), "runs", str(self.system),
                              self.model.method)
        if "Supervised" in self.model.method:
            params += "/" + self.method
        while os.path.isdir(os.path.join(params, f"exp_{i}")):
            i += 1
        self.results_folder = os.path.join(params, f"exp_{i}")
        print(f"Results saved in in {self.results_folder}")

        # Scaling data: not for "Supervised_noise", which has different scalers
        # for encoder (x, wc) -> (z) and decoder (z, wc) -> (x)
        if self.model.method not in ["Supervised_noise"]:
            if "Autoencoder" in self.method:
                self.scaler_x = StandardScaler(self.training_data, self.device)
                self.scaler_z = None
            else:
                self.scaler_x = StandardScaler(
                    self.training_data[:, self.x_idx_in], self.device
                )
                self.scaler_z = StandardScaler(
                    self.training_data[:, self.z_idx_in], self.device
                )
            self.model.set_scalers(scaler_x=self.scaler_x,
                                   scaler_z=self.scaler_z)

    def configure_optimizers(self):
        # https://pytorch-lightning.readthedocs.io/en/stable/common/optimizers.html
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/2976
        if self.optimizer_options is not None:
            optimizer_options = self.optimizer_options
        else:
            optimizer_options = {}
        parameters = self.model.parameters()
        # If method == 'T', ignore parameters of T_star for optim and vice versa
        if self.method == "T":
            parameters_to_optim = set(parameters)
            for param in self.model.decoder.parameters():
                parameters_to_optim -= {param}
            parameters = list(parameters_to_optim)
        elif self.method == "T_star":
            parameters_to_optim = set(parameters)
            for param in self.model.encoder.parameters():
                parameters_to_optim -= {param}
            parameters = list(parameters_to_optim)

        optimizer = self.optimizer(parameters, self.optim_lr,
                                   **optimizer_options)
        if self.scheduler:
            if self.scheduler_options:
                scheduler_options = self.scheduler_options
            else:
                scheduler_options = {
                    "mode": "min",
                    "factor": 0.8,
                    "patience": 10,
                    "threshold": 5e-2,
                    "verbose": True,
                }
            scheduler = {
                "scheduler": self.scheduler(optimizer, **scheduler_options),
                "monitor": "train_loss",
            }
            return [optimizer], [scheduler]
        else:
            return optimizer

    def forward(self, batch):
        # Compute x_hat and/or z_hat depending on the method
        if "Autoencoder" in self.method:
            x = batch  # .to(self.device)
            z_hat, x_hat = self.model(self.method, x)
            return z_hat, x_hat
        elif self.method == "T":
            x = batch[:, self.x_idx_in]
            z_hat = self.model(self.method, x)
            return z_hat
        elif self.method == "T_star":
            z = batch[:, self.z_idx_in]
            x_hat = self.model(self.method, z)
            return x_hat
        else:
            raise KeyError(f"Unknown method {self.method}")

    def train_dataloader(self):
        train_loader = DataLoader(
            self.training_data, batch_size=self.batch_size, shuffle=True,
            num_workers=0
        )
        return train_loader

    def training_step(self, batch, batch_idx):
        # Compute transformation and loss depending on the method
        if self.method == "Autoencoder":
            z_hat, x_hat = self.forward(batch)
            loss, loss1, loss2 = self.model.loss(self.method, batch, x_hat,
                                                 z_hat)
            self.log("train_loss1", loss1, on_step=True, prog_bar=False,
                     logger=True)
            self.log("train_loss2", loss2, on_step=True, prog_bar=False,
                     logger=True)
        elif self.method == "Autoencoder_jointly":
            z_hat, x_hat = self.forward(batch)
            loss, loss1, loss2, loss3 = self.model.loss(self.method, batch,
                                                        x_hat, z_hat)
            self.log("train_loss1", loss1, on_step=True, prog_bar=False,
                     logger=True)
            self.log("train_loss2", loss2, on_step=True, prog_bar=False,
                     logger=True)
            self.log("train_loss3", loss3, on_step=True, prog_bar=False,
                     logger=True)
        elif self.method == "T":
            z = batch[:, self.z_idx_out]
            z_hat = self.forward(batch)
            loss = self.model.loss(self.method, z, z_hat)
        elif self.method == "T_star":
            x = batch[:, self.x_idx_out]
            x_hat = self.forward(batch)
            loss = self.model.loss(self.method, x, x_hat, self.axe)
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.train_loss = torch.cat((self.train_loss, torch.tensor([[loss]])))
        logs = {"train_loss": loss.detach()}

        return {"loss": loss, "log": logs}

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.validation_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        return val_dataloader

    def validation_step(self, batch, batch_idx):
        # Compute transformation and loss depending on the method
        with torch.no_grad():
            if self.method == "Autoencoder":
                z_hat, x_hat = self.forward(batch)
                loss, loss1, loss2 = self.model.loss(self.method, batch, x_hat,
                                                     z_hat)
                self.log("val_loss1", loss1, on_step=True, prog_bar=False,
                         logger=True)
                self.log("val_loss2", loss2, on_step=True, prog_bar=False,
                         logger=True)
            elif self.method == "Autoencoder_jointly":
                z_hat, x_hat = self.forward(batch)
                loss, loss1, loss2, loss3 = self.model.loss(self.method, batch,
                                                            x_hat, z_hat)
                self.log("val_loss1", loss1, on_step=True, prog_bar=False,
                         logger=True)
                self.log("val_loss2", loss2, on_step=True, prog_bar=False,
                         logger=True)
                self.log("val_loss3", loss3, on_step=True, prog_bar=False,
                         logger=True)
            elif self.method == "T":
                z = batch[:, self.z_idx_out]
                z_hat = self.forward(batch)
                loss = self.model.loss(self.method, z, z_hat)
            elif self.method == "T_star":
                x = batch[:, self.x_idx_out]
                x_hat = self.forward(batch)
                loss = self.model.loss(self.method, x, x_hat, self.axe)
            self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True)
            self.val_loss = torch.cat((self.val_loss, torch.tensor([[loss]])))
            logs = {"val_loss": loss.detach()}

            return {"loss": loss, "log": logs}

    def save_csv(self, data, path):
        file = pd.DataFrame(data)
        file.to_csv(path, header=False)

    def save_specifications(self):
        specs_file = os.path.join(self.results_folder, "Specifications.txt")
        with open(specs_file, "w") as f:
            print(sys.argv[0], file=f)
            for key, val in vars(self.system).items():
                print(key, ": ", val, file=f)
            for key, val in vars(self).items():
                print(key, ": ", val, file=f)
            try:
                # Add t_c to specifications
                print(f'k {self.model.k}', file=f)
                print(f't_c {self.model.t_c}', file=f)
            except AttributeError:
                print('No value of t_c in observer model.')

        if 'jointly' in self.method:
            eig0 = torch.linalg.eigvals(self.model.D_0)
            eig = torch.linalg.eigvals(self.model.D)
            plt.plot(eig0.real, eig0.imag, 'x', label='Initial')
            plt.plot(eig.real, eig.imag, 'o', label='Final')
            plt.title('Eigenvalues of optimized D')
            plt.xlabel(r'$\mathbb{R}$')
            plt.ylabel(r'$i\mathbb{R}$')
            plt.legend()
            plt.savefig(os.path.join(self.results_folder, 'OptimD_eigvals.pdf'),
                        bbox_inches='tight')
            plt.clf()
            plt.close('all')
        return specs_file

    def save_pkl(self, fileName, object):
        with open(self.results_folder + fileName, "wb") as f:
            pkl.dump(object, f, protocol=4)

    def save_pdf_training(self, data, verbose):
        for i in range(1, data.shape[1]):
            name = "training_data" + str(i) + ".pdf"
            plt.scatter(data[:, i - 1].cpu(), data[:, i].cpu())
            plt.title("Training data")
            plt.xlabel(rf"$x_{i}$")
            plt.ylabel(rf"$x_{i + 1}$")
            plt.savefig(os.path.join(self.results_folder, name),
                        bbox_inches="tight")
            if verbose:
                plt.show()

            plt.clf()
            plt.close("all")

    def save_pdf_heatmap(self, x_mesh, x_hat_star, verbose):
        with torch.no_grad():
            error = RMSE(x_mesh, x_hat_star, dim=1)
            for i in range(1, x_mesh.shape[1]):
                # https://stackoverflow.com/questions/37822925/how-to-smooth-by-interpolation-when-using-pcolormesh
                name = "RMSE_heatmap" + str(i) + ".pdf"
                plt.scatter(
                    x_mesh[:, i - 1],
                    x_mesh[:, i],
                    cmap="jet",
                    c=np.log(error.detach().numpy()),
                )
                cbar = plt.colorbar()
                cbar.set_label("Log estimation error")
                plt.title(rf"RMSE between $x$ and $\hat{{x}}$: "
                          rf"{np.mean(error.detach().numpy()):0.2g}")
                plt.xlabel(rf"$x_{i}$")
                plt.ylabel(rf"$x_{i + 1}$")
                plt.legend()
                plt.savefig(os.path.join(self.results_folder, name),
                            bbox_inches="tight")
                if verbose:
                    plt.show()
                plt.clf()
                plt.close('all')

    def save_trj(self, init_state, verbose, tsim, dt, var=0.2,
                 traj_folder=None, z_0=None):
        # Estimation over the test trajectories with T_star
        if traj_folder is None:
            traj_folder = os.path.join(self.results_folder,
                                       "Test_trajectories/Traj_{var}")
        tq, simulation = self.system.simulate(init_state, tsim, dt)

        noise = torch.normal(0, var, size=(simulation.shape))

        simulation_noise = simulation.add(noise)

        measurement = self.model.h(simulation_noise)

        # Save these test trajectories
        os.makedirs(traj_folder, exist_ok=True)
        traj_error = 0.0

        # TODO run predictions in parallel for all test trajectories!!!
        # Need to figure out how to interpolate y in parallel for all
        # trajectories!!!
        y = torch.cat((tq.unsqueeze(1), measurement), dim=-1)

        estimation = self.model.predict(y, tsim, dt, z_0=z_0).detach()
        error = RMSE(simulation, estimation)
        traj_error += error

        filename = f"RMSE.txt"
        with open(os.path.join(traj_folder, filename), "w") as f:
            print(error.cpu().numpy(), file=f)

        self.save_csv(
            simulation.cpu().numpy(),
            os.path.join(traj_folder, f"True_traj.csv"),
        )
        self.save_csv(
            estimation.cpu().numpy(),
            os.path.join(traj_folder, f"Estimated_traj.csv"),
        )

        for j in range(estimation.shape[1]):
            name = "Traj" + str(j) + ".pdf"
            if j == 0:
                plt.plot(tq, simulation_noise[:, j].detach().numpy(), '-',
                         label=r"$y$")
            plt.plot(tq, simulation[:, j].detach().numpy(), '--',
                     label=rf"$x_{j + 1}$")
            plt.plot(
                tq, estimation[:, j].detach().numpy(), '-.',
                label=rf"$\hat{{x}}_{j + 1}$"
            )
            plt.legend(loc=1)
            plt.grid(visible=True)
            plt.title(rf"Test trajectory, RMSE = {np.round(error.numpy(), 4)}")
            plt.xlabel(rf"$t$")
            plt.ylabel(rf"$x_{j + 1}$")
            plt.savefig(
                os.path.join(traj_folder, name), bbox_inches="tight"
            )
            if verbose:
                plt.show()
            plt.clf()
            plt.close("all")

        filename = "RMSE_traj.txt"
        with open(os.path.join(traj_folder, filename), "w") as f:
            print(traj_error, file=f)

    def save_random_traj(self, x_mesh, num_samples, nb_trajs, verbose, tsim,
                         dt, std=0., traj_folder=None, z_0=None):
        with torch.no_grad():
            # Estimation over the test trajectories with T_star
            if traj_folder is None:
                random_idx = np.random.choice(np.arange(num_samples),
                                              size=(nb_trajs,), replace=False)
                trajs_init = x_mesh[random_idx]
                traj_folder = os.path.join(self.results_folder,
                                           "Test_trajectories")
            else:
                nb_traj = len(next(os.walk(traj_folder))[1])
                trajs_init = torch.zeros((nb_traj, self.model.dim_x))
                for i in range(nb_traj):
                    current_traj_folder = os.path.join(traj_folder, f'Traj_{i}')
                    df = pd.read_csv(os.path.join(
                        current_traj_folder, f'True_traj_{i}.csv'), sep=',',
                        header=None)
                    trajs_init[i] = torch.from_numpy(
                        df.drop(df.columns[0], axis=1).values)[0]
            tq, simulation = self.system.simulate(trajs_init, tsim, dt)

            noise = torch.normal(0, std, size=(simulation.shape))

            simulation_noise = simulation.add(noise)

            measurement = self.model.h(simulation_noise)

            # Save these test trajectories
            os.makedirs(traj_folder, exist_ok=True)
            traj_error = 0.0
            for i in range(nb_trajs):
                # TODO run predictions in parallel for all test trajectories!!!
                # Need to figure out how to interpolate y in parallel for all
                # trajectories!!!
                # y = torch.cat((tq.unsqueeze(1), measurement[..., i]), dim=1)
                y = torch.cat((tq.unsqueeze(1), measurement[:, i]), dim=1)
                estimation = self.model.predict(y, tsim, dt, z_0=z_0).detach()
                rmse = RMSE(simulation[:, i], estimation)
                traj_error += rmse

                current_traj_folder = os.path.join(traj_folder, f"Traj_{i}")
                os.makedirs(current_traj_folder, exist_ok=True)

                self.save_csv(
                    simulation[:, i].cpu().numpy(),
                    os.path.join(current_traj_folder, f"True_traj_{i}.csv"),
                )
                self.save_csv(
                    estimation.cpu().numpy(),
                    os.path.join(current_traj_folder,
                                 f"Estimated_traj_{i}.csv"),
                )

                for j in range(estimation.shape[1]):
                    name = "Traj" + str(j) + ".pdf"
                    if j == 0:
                        plt.plot(tq, simulation_noise[:, i, j].detach().numpy(),
                                 '-', label=r"$y$")
                    plt.plot(tq, simulation[:, i, j].detach().numpy(), '--',
                             label=rf"$x_{j + 1}$")
                    plt.plot(
                        tq, estimation[:, j].detach().numpy(), '-.',
                        label=rf"$\hat{{x}}_{j + 1}$"
                    )
                    plt.legend()
                    plt.grid(visible=True)
                    plt.xlabel(rf"$t$")
                    plt.ylabel(rf"$x_{j + 1}$")
                    plt.savefig(
                        os.path.join(current_traj_folder, name),
                        bbox_inches="tight"
                    )
                    if verbose:
                        plt.show()
                    plt.clf()
                    plt.close("all")

            filename = "RMSE_traj.txt"
            with open(os.path.join(traj_folder, filename), "w") as f:
                print(traj_error / nb_trajs, file=f)

    def save_invert_heatmap(self, x_mesh, x_hat_AE, verbose, wc=None):
        with torch.no_grad():
            # Invertibility heatmap
            error = RMSE(x_mesh, x_hat_AE, dim=1)
            for i in range(1, x_mesh.shape[1]):
                if wc is not None:
                    name = f"Invertibility_heatmap_wc{wc:0.2g}_{i}.pdf"
                else:
                    name = "Invertibility_heatmap" + str(i) + ".pdf"
                plt.scatter(
                    x_mesh[:, i - 1],
                    x_mesh[:, i],
                    cmap="jet",
                    c=np.log(error.detach().numpy()),
                )
                cbar = plt.colorbar()
                cbar.set_label("Log estimation error")
                if wc is not None:
                    plt.title(
                        rf"RMSE between $x$ and $T^*(T(x))$, $\omega_c$ = {wc:0.2g}")
                else:
                    plt.title(r"RMSE between $x$ and $T^*(T(x))$")
                plt.xlabel(rf"$x_{i}$")
                plt.ylabel(rf"$x_{i + 1}$")
                plt.legend()
                plt.savefig(os.path.join(self.results_folder, name),
                            bbox_inches="tight")
                if verbose:
                    plt.show()

                plt.clf()
                plt.close("all")

    def save_plot(self, name, title, y_scale, data):
        plt.plot(data, "+-", label="loss")
        plt.title(title)
        plt.yscale(y_scale)
        plt.legend()
        plt.savefig(os.path.join(self.results_folder, name),
                    bbox_inches="tight")
        plt.clf()
        plt.close("all")

    def save_loss_grid(self, x_mesh, x_hat_AE, z_hat_T, x_hat_star, verbose):
        with torch.no_grad():
            losses = []
            if self.method == "Autoencoder":
                loss, loss1, loss2 = self.model.loss_autoencoder(
                    x_mesh, x_hat_AE, z_hat_T, dim=-1
                )
                losses.append(loss1)
                losses.append(loss2)
            elif self.method == "T_star":
                loss = self.model.loss_T_star(x_mesh, x_hat_star, dim=-1)
                losses.append(loss)
            for j in range(len(losses)):
                loss = losses[j]
                for i in range(1, x_mesh.shape[1]):
                    name = f"Loss{j + 1}_{i - 1}.pdf"
                    plt.scatter(
                        x_mesh[:, i - 1],
                        x_mesh[:, i],
                        cmap="jet",
                        c=np.log(loss.detach().numpy()),
                    )
                    cbar = plt.colorbar()
                    cbar.set_label("Log loss")
                    plt.title("Loss over grid")
                    plt.xlabel(rf"$x_{i}$")
                    plt.ylabel(rf"$x_{i + 1}$")
                    plt.legend()
                    plt.savefig(
                        os.path.join(self.results_folder, name),
                        bbox_inches="tight"
                    )
                    if verbose:
                        plt.show()
                    plt.clf()
                    plt.close("all")

    def save_results(
            self,
            limits: np.array,
            nb_trajs=10,
            tsim=(0, 60),
            dt=1e-2,
            num_samples=[50,50],
            method='uniform',
            checkpoint_path=None,
            verbose=False,
            fast=False,
    ):
        """
        Save the model, the training and validation data. Also saving several
        metrics used for evaluating the results: heatmap of estimation error
        and estimation of several test trajectories.

        Parameters
        ----------
        limits: np.array
            Array for the limits of all axes of x, used for sampling the
            heatmap and the initial conditions of the test trajectories.
            Form np.array([[min_1, max_1], ..., [min_n, max_n]]).

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

        fast: bool
            Whether to compute the loss over a grid, which is slow.
        """
        with torch.no_grad():
            if checkpoint_path:
                checkpoint_model = torch.load(checkpoint_path)
                self.load_state_dict(checkpoint_model["state_dict"])

            # Save training and validation data
            nb = int(np.min([10000, len(self.training_data)]))
            idx = np.random.choice(
                np.arange(len(self.training_data)), size=(nb,), replace=False
            )  # subsampling for plots

            specs_file = self.save_specifications()

            self.save_pkl("/learner.pkl", self)

            self.save_csv(
                self.training_data.cpu().numpy(),
                os.path.join(self.results_folder, "training_data.csv"),
            )
            self.save_csv(
                self.validation_data.cpu().numpy(),
                os.path.join(self.results_folder, "validation_data.csv"),
            )

            self.save_pdf_training(self.training_data[idx], verbose)

            # Loss plot over time
            self.save_plot(
                "Train_loss.pdf",
                "Training loss over time",
                "log",
                self.train_loss.detach(),
            )
            self.save_plot(
                "Val_loss.pdf",
                "Validation loss over time",
                "log",
                self.val_loss.detach(),
            )

            # Heatmap of RMSE(x, x_hat) with T_star
            mesh,_ = self.model.generate_data_svl(limits, num_samples,
                                                method=method)
            x_mesh = mesh[:, self.x_idx_out]
            z_mesh = mesh[:, self.z_idx_out]
            x_hat_star = self.model("T_star", z_mesh)

            z_hat_T, x_hat_AE = self.model("Autoencoder", x_mesh)
            num_samples = len(mesh)  # update num_samples from uniform grid

            print(f"Shape of mesh for evaluation: {mesh.shape}")

            self.save_pdf_heatmap(x_mesh, x_hat_star, verbose)
            self.save_random_traj(x_mesh, num_samples, nb_trajs, verbose, tsim,
                                  dt)
            # Invertibility heatmap
            self.save_invert_heatmap(x_mesh, x_hat_AE, verbose)

            # Loss heatmap over space
            if not fast:  # Computing loss over grid is slow, most of all for AE
                self.save_loss_grid(x_mesh, x_hat_AE, z_hat_T, x_hat_star,
                                    verbose)

            # No control theoretic evaluation of the observer with only T
            if self.method == "T":
                return 0
