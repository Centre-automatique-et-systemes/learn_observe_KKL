import os

import random as rd

from learn_KKL.learner import Learner

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import torch
from torch.nn import functional
import torch.optim as optim

from .utils import RMSE, StandardScaler

# Set double precision by default
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)


class LearnerNoise(Learner):
    def __init__(
        self,
        observer,
        system,
        training_data,
        validation_data,
        method="Autoencoder",
        batch_size=10,
        lr=1e-3,
        optimizer=optim.Adam,
        optimizer_options=None,
        scheduler=optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_options=None
    ):

        super(LearnerNoise, self).__init__(
            observer,
            system,
            training_data,
            validation_data,
            method,
            batch_size,
            lr,
            optimizer,
            optimizer_options,
            scheduler,
            scheduler_options)

        # Indices of the x inputs = (x, wc) and the z inputs = (z, wc)
        # Indices of the x outputs = (x) and z outputs = (z)
        self.x_idx_in = [i for i in range(self.model.dim_x)] + [-1]
        self.x_idx_out = [i for i in range(self.model.dim_x)]
        self.z_idx_in = [i for i in range(self.model.dim_x, self.model.dim_x
                                          + self.model.dim_z)] + [-1]
        self.z_idx_out = [i for i in range(self.model.dim_x, self.model.dim_x
                                           + self.model.dim_z)]

        # Correct scalers from parent class
        # Encoder: xin = (x, wc), zout = (z)
        # Decoder: zin = (z, wc), xout = (x)
        self.scaler_xin = StandardScaler(
            self.training_data[:, self.x_idx_in], self.device
        )
        self.scaler_xout = StandardScaler(
            self.training_data[:, self.x_idx_out], self.device
        )
        self.scaler_zin = StandardScaler(
            self.training_data[:, self.z_idx_in], self.device
        )
        self.scaler_zout = StandardScaler(
            self.training_data[:, self.z_idx_out], self.device
        )
        self.model.set_scalers(scaler_xin=self.scaler_xin,
                               scaler_xout=self.scaler_xout,
                               scaler_zin=self.scaler_zin,
                               scaler_zout=self.scaler_zout)
    
    def save_random_traj(self, x_mesh, w_c_arr, nb_trajs, verbose, tsim, dt,
                         std=0.01):
        with torch.no_grad():
            # Estimation over the test trajectories with T_star
            nb_trajs += w_c_arr.shape[0]

            random_idx = np.random.choice(np.arange(x_mesh.shape[0]),
                                          size=(nb_trajs,), replace=False)
            trajs_init = x_mesh[random_idx]
            traj_folder = os.path.join(self.results_folder, "Test_trajectories")
            tq, simulation = self.system.simulate(trajs_init, tsim, dt)
            measurement = self.model.h(simulation)
            noise = (
                torch.normal(0, std, size=(measurement.shape[0], 1))
                .repeat(1, nb_trajs)
                .unsqueeze(2)
            )
            measurement = measurement.add(noise)

            # Save these test trajectories
            os.makedirs(traj_folder, exist_ok=True)
            traj_error = 0.0

            for i in range(nb_trajs):
                # TODO run predictions in parallel for all test trajectories!!!
                # Need to figure out how to interpolate y in parallel for all
                # trajectories!!!
                y = torch.cat((tq.unsqueeze(1), measurement[:, i]), dim=1)

                if i < len(w_c_arr):
                    w_c = w_c_arr[i]
                else:
                    w_c = rd.uniform(0.5, 1.5)

                # estimation = self.model.predict(y, tsim, dt, w_c).detach()
                estimation, z = self.model.predict(y, tsim, dt, w_c, out_z=True)
                # TODO delete
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
                    os.path.join(current_traj_folder, f"Estimated_traj_{i}.csv"),
                )

                for j in range(estimation.shape[1]):
                    name = "Traj" + str(j) + ".pdf"
                    plt.plot(
                        tq, simulation[:, i, j].detach().numpy(), label=rf"$x_{j + 1}$"
                    )
                    plt.plot(
                        tq, estimation[:, j].detach().numpy(), label=rf"$\hat{{x}}_{j + 1}$"
                    )
                    plt.legend(loc=1)
                    plt.title(rf"Random trajectory for $\omega_c$ {w_c:0.2g}, "
                              rf"RMSE = "rf"{rmse:0.2g}")
                    plt.xlabel(rf"$t$")
                    plt.ylabel(rf"$x_{j + 1}$")
                    plt.savefig(
                        os.path.join(current_traj_folder, name), bbox_inches="tight"
                    )
                    if verbose:
                        plt.show()
                    plt.close("all")
                # TODO delete
                name = "Phase_portrait.pdf"
                plt.plot(
                    simulation[:, i, 0].detach().numpy(), simulation[:, i,
                                                          1].detach().numpy(), label=rf"True"
                )
                plt.plot(
                    estimation[:, 0].detach().numpy(), estimation[:, 1].detach().numpy(),
                    label=rf"Predicted"
                )
                plt.legend(loc=1)
                plt.title(rf"Random trajectory for $\omega_c$ {w_c:0.2g}, "
                          rf"RMSE = "rf"{rmse:0.2g}")
                plt.xlabel(rf"$x_1$")
                plt.ylabel(rf"$x_2$")
                plt.savefig(
                    os.path.join(current_traj_folder, name), bbox_inches="tight"
                )
                plt.close("all")
                for j in range(z.shape[1]-1):
                    name = "Traj_z" + str(j) + ".pdf"
                    plt.plot(
                        tq, z[:, j].detach().numpy(), label=rf"$z_{j + 1}$"
                    )
                    plt.legend(loc=1)
                    plt.title(rf"Random trajectory for $\omega_c$ {w_c:0.2g}, "
                              rf"RMSE = "rf"{rmse:0.2g}")
                    plt.xlabel(rf"$t$")
                    plt.ylabel(rf"$z_{j + 1}$")
                    plt.savefig(
                        os.path.join(current_traj_folder, name), bbox_inches="tight"
                    )
                    if verbose:
                        plt.show()
                    plt.close("all")

            filename = "RMSE_traj.txt"
            with open(os.path.join(traj_folder, filename), "w") as f:
                print(traj_error / nb_trajs, file=f)

    def save_trj(self, init_state, w_c_arr, nb_trajs, verbose, tsim, dt, var=0.2):
        with torch.no_grad():
            # Estimation over the test trajectories with T_star
            nb_trajs += w_c_arr.shape[0]
            traj_folder = os.path.join(self.results_folder, "Test_trajectories_{}".format(str(var)))
            tq, simulation = self.system.simulate(init_state, tsim, dt)

            noise = torch.normal(0, var, size=(simulation.shape))

            simulation_noise = simulation.add(noise)

            measurement = self.model.h(simulation_noise)

            # Save these test trajectories
            os.makedirs(traj_folder, exist_ok=True)
            traj_error = 0.0

            for i in range(nb_trajs):
                # TODO run predictions in parallel for all test trajectories!!!
                # Need to figure out how to interpolate y in parallel for all
                # trajectories!!!
                y = torch.cat((tq.unsqueeze(1), measurement), dim=1)

                if i < len(w_c_arr):
                    w_c = w_c_arr[i]
                else:
                    w_c = rd.uniform(0.2, 1.5)

                estimation = self.model.predict(y, tsim, dt, w_c).detach()
                error = RMSE(simulation, estimation)
                traj_error += error

                current_traj_folder = os.path.join(traj_folder, f"Traj_{i}")
                os.makedirs(current_traj_folder, exist_ok=True)

                filename = f"RMSE_{i}.txt"
                with open(os.path.join(current_traj_folder, filename), "w") as f:
                    print(error.cpu().numpy(), file=f)

                self.save_csv(
                    simulation.cpu().numpy(),
                    os.path.join(current_traj_folder, f"True_traj_{i}.csv"),
                )
                self.save_csv(
                    estimation.cpu().numpy(),
                    os.path.join(current_traj_folder, f"Estimated_traj_{i}.csv"),
                )

                for j in range(estimation.shape[1]):
                    name = "Traj" + str(j) + ".pdf"
                    if j == 0:
                        plt.plot(
                            tq, measurement[:, j].detach().numpy(), "-", label=r"$y$"
                        )
                    plt.plot(
                        tq, simulation[:, j].detach().numpy(), "--", label=rf"$x_{j + 1}$"
                    )
                    plt.plot(
                        tq,
                        estimation[:, j].detach().numpy(),
                        "-.",
                        label=rf"$\hat{{x}}_{j + 1}$",
                    )
                    plt.legend(loc=1)
                    plt.grid(visible=True)
                    plt.title(
                        rf"Test trajectory for $\omega_c = $ "
                        rf"{w_c:0.2g}, RMSE = {error:0.2g}"
                    )
                    plt.xlabel(rf"$t$")
                    plt.ylabel(rf"$x_{j + 1}$")
                    plt.savefig(
                        os.path.join(current_traj_folder, name), bbox_inches="tight"
                    )
                    if verbose:
                        plt.show()
                    plt.clf()
                    plt.close("all")

            filename = "RMSE_traj.txt"
            with open(os.path.join(traj_folder, filename), "w") as f:
                print(traj_error / nb_trajs, file=f)

    def save_rmse_wc(self, mesh, w_c_array, verbose):
        with torch.no_grad():
            errors = np.zeros((len(w_c_array)))

            for j in range(mesh.shape[-1]):
                x_mesh = mesh[:, self.x_idx_out, j]
                z_mesh = mesh[:, self.z_idx_in, j]

                # noise = torch.normal(0, 0.01, size=(z_mesh.shape[0], z_mesh.shape[1]))
                # z_mesh.add(noise)

                # compute x_hat for every w_c
                x_hat_star = self.model("T_star", z_mesh)

                errors[j] = RMSE(x_mesh, x_hat_star).detach().numpy()

            # https://stackoverflow.com/questions/37822925/how-to-smooth-by-interpolation-when-using-pcolormesh
            name = "RMSE_w_c.pdf"
            self.save_csv(
                np.concatenate(
                    (np.expand_dims(w_c_array, 1), np.expand_dims(errors, 1)), axis=1
                ),
                os.path.join(self.results_folder, "RMSE.csv"),
            )
            plt.plot(w_c_array, errors)
            plt.grid(False)
            plt.title(r"RMSE between $x$ and $\hat{x}$")
            plt.xlabel(rf"$\omega_c$")
            plt.ylabel(r"RMSE($x$, $\hat{x}$)")
            plt.savefig(os.path.join(self.results_folder, name), bbox_inches="tight")
            if verbose:
                plt.show()
                plt.close("all")

            plt.clf()
            plt.close("all")

    def save_pdf_heatmap(self, mesh, verbose):
        with torch.no_grad():
            for j in range(mesh.shape[-1]):
                x_mesh = mesh[:, self.x_idx_out, j]
                z_mesh = mesh[:, self.z_idx_in, j]
                w_c = z_mesh[0, -1]

                # compute x_hat for every w_c
                x_hat_star = self.model("T_star", z_mesh)

                error = RMSE(x_mesh, x_hat_star, dim=1)

                # TODO DELETE
                tq, simulation = self.system.simulate(
                    torch.tensor([1.5, 1.5]), (0, 60), 1e-2
                )
                measurement = self.model.h(simulation)
                y = torch.cat((tq.unsqueeze(1), measurement), dim=1)
                estimation = self.model.predict(y, (0, 60), 1e-2, w_c).detach()

                for i in range(1, x_mesh.shape[1]):
                    # https://stackoverflow.com/questions/37822925/how-to-smooth-by-interpolation-when-using-pcolormesh
                    name = "RMSE_heatmap" + str(j) + ".pdf"
                    plt.scatter(
                        x_mesh[:, i - 1],
                        x_mesh[:, i],
                        cmap="jet",
                        c=np.log(error.detach().numpy()),
                        # vmin=-11,
                        # vmax=-3,
                    )
                    cbar = plt.colorbar()
                    cbar.set_label("Log estimation error")
                    # cbar.set_label("Log estimation error")
                    plt.plot(simulation[:,0], simulation[:,1])
                    plt.plot(estimation[:, 0], estimation[:, 1], '--')

                    plt.title(
                        r"RMSE between $x$ and $\hat{x}$"
                        + rf" with $\omega_c =$ {w_c.item():0.2g}"
                    )
                    plt.grid(False)
                    plt.xlabel(rf"$x_{i}$")
                    plt.ylabel(rf"$x_{i + 1}$")
                    plt.savefig(
                        os.path.join(self.results_folder, name), bbox_inches="tight"
                    )
                    if verbose:
                        plt.show()
                        plt.close("all")

                    plt.clf()
                    plt.close("all")

    def plot_sensitiviy_wc(self, mesh, w_c_array, verbose,
                           y_lim=[None, None], save=True, path=''):
        errors = np.zeros((len(w_c_array), 3))
        if path == '':
            path = os.path.join(self.results_folder, 'xzi_mesh')
            os.makedirs(path, exist_ok=True)

        D_arr = torch.zeros(0, 3, 3)
        for j in range(len(w_c_array)):
            wc = w_c_array[j]
            self.model.D, _ = self.model.set_DF(wc)

            if save:
                x_mesh = mesh[:, self.x_idx_in, j]
                z_mesh = mesh[:, self.z_idx_in, j]

                file = pd.DataFrame(mesh[:, :, j])
                file.to_csv(os.path.join(
                    path, f'xzi_data_wc{wc:0.2g}.csv'), header=False)
                for i in self.x_idx_out[:-1] + self.z_idx_out[:-1]:
                    plt.scatter(mesh[:, i, j], mesh[:, i + 1, j])
                    plt.savefig(os.path.join(
                        path, f'xzi_data_wc{wc:0.2g}_{i}.pdf'),
                        bbox_inches='tight')
                    plt.xlabel(rf'$x_{i}$')
                    plt.xlabel(rf'$x_{i+1}$')
                    plt.clf()
                    plt.close('all')
            else:
                # Load mesh that was saved in path
                df = pd.read_csv(os.path.join(
                    path, f'xzi_data_wc{wc:0.2g}.csv'), sep=',',
                    header=None)
                mesh = torch.from_numpy(df.drop(df.columns[0], axis=1).values)
                x_mesh = mesh[:, self.x_idx_in]
                z_mesh = mesh[:, self.z_idx_in]

            errors[j] = self.model.sensitivity_norm(x_mesh, z_mesh, save=save,
                                                    path=path, version=3)
            D_arr = torch.cat((D_arr, torch.unsqueeze(self.model.D, 0)))

        # errors = functional.normalize(errors)
        self.save_csv(
            D_arr.flatten(1, -1),
            os.path.join(path, "D_arr.csv"),
        )
        self.save_csv(
            w_c_array,
            os.path.join(path, "w_c_array.csv"),
        )
        self.save_csv(
            np.concatenate((np.expand_dims(w_c_array, 1), errors), axis=1),
            os.path.join(path, "sensitivity.csv"),
        )
        name = "sensitivity_wc.pdf"
        blue = "tab:blue"
        orange = "tab:orange"
        green = "tab:green"

        fig, ax1 = plt.subplots()

        ax1.set_xlabel(r"$\omega_c$")
        # ax1.set_ylabel('exp', color=color)
        ax1.tick_params(axis='y')
        # line_1 = ax1.plot(
        #     w_c_array,
        #     errors[:, 0],
        #     label=r"$\frac{1}{N}\max_{z_i} \left| \frac{\partial \mathcal{T}^*}{\partial z} (z_i) \right|_{l^2}$",
        #     color=blue
        # )
        # line_2 = ax1.plot(w_c_array, errors[:, 1], label=r"$\left| G \right|_\infty$", color=green)

        ax2 = ax1.twinx()

        line_3 = ax1.plot(
            w_c_array,
            errors[:, 2],
            label=r"$\max_{z_i} \left| \frac{\partial \mathcal{T}^*}{\partial z} (z_i) \right| \left| G_{\epsilon} \right|_\infty$",
            color=orange,
        )
        ax2.tick_params(axis='y')
        fig.tight_layout()

        # added these three lines
        # lns = line_3 + line_1 + line_2
        lns = line_3
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=1)
        ax1.grid(False)

        plt.title("Gain tuning criterion")

        plt.savefig(os.path.join(self.results_folder, name), bbox_inches="tight")

        if verbose:
            plt.show()
            plt.close("all")

        plt.clf()
        plt.close("all")

    def plot_traj_error(
        self, init_state, w_c_arr, nb_trajs, verbose, tsim, dt, var=0.0
    ):
        with torch.no_grad():
            # Estimation over the test trajectories with T_star

            nb_trajs += w_c_arr.shape[0]
            traj_folder = os.path.join(self.results_folder, "Test_trajectories_error_{}".format(str(var)))
            tq, simulation = self.system.simulate(init_state, tsim, dt)

            noise = torch.normal(0, var, size=(simulation.shape[0], 2))

            measurement_noise = simulation.add(noise)

            measurement = self.model.h(measurement_noise)
            # measurement = measurement.add(noise)

            # Save these test trajectories
            os.makedirs(traj_folder, exist_ok=True)
            traj_error = 0.0

            plot_style = ["-", "--", "-."]

            for i in range(nb_trajs):
                # TODO run predictions in parallel for all test trajectories!!!
                # Need to figure out how to interpolate y in parallel for all
                # trajectories!!!
                y = torch.cat((tq.unsqueeze(1), measurement), dim=1)

                if i < len(w_c_arr):
                    w_c = w_c_arr[i]
                else:
                    w_c = rd.uniform(0.2, 1.5)

                estimation = self.model.predict(y, tsim, dt, w_c).detach()
                error = RMSE(simulation, estimation)
                traj_error += error

                current_traj_folder = os.path.join(traj_folder, f"Traj_{i}")
                os.makedirs(current_traj_folder, exist_ok=True)

                filename = f"RMSE_{i}.txt"
                with open(os.path.join(current_traj_folder, filename), "w") as f:
                    print(error.cpu().numpy(), file=f)

                self.save_csv(
                    simulation.cpu().numpy(),
                    os.path.join(current_traj_folder, f"True_traj_{i}.csv"),
                )
                self.save_csv(
                    estimation.cpu().numpy(),
                    os.path.join(current_traj_folder, f"Estimated_traj_{i}.csv"),
                )

                self.save_csv(
                    abs(estimation.cpu().numpy() - simulation.cpu().numpy()),
                    os.path.join(current_traj_folder, f"Error_traj_{i}.csv"),
                )

                # for i in range(simulation.shape[1]):
                name = "Traj" + str(1) + ".pdf"
                plt.plot(
                    tq,
                    estimation[:, 1].cpu().numpy() - simulation[:, 1].cpu().numpy(),
                    plot_style[i],
                    linewidth=0.8,
                    markersize=1,
                    label=rf"$\omega_c = {float(w_c_arr[i]):0.2g}$",
                )

            plt.legend(loc=1)
            plt.grid(visible=True)
            plt.title(rf"Test trajectory error")
            plt.xlabel(rf"$t$")
            plt.ylabel(rf"$\hat{{x}}_{1 + 1}-x_{1+1}$")
            plt.savefig(os.path.join(current_traj_folder, name), bbox_inches="tight")
            if verbose:
                plt.show()
            plt.clf()
            plt.close("all")

            filename = "RMSE_traj.txt"
            with open(os.path.join(traj_folder, filename), "w") as f:
                print(traj_error / nb_trajs, file=f)

    def plot_traj_sens(self, init_state, w_c_array, t_sim, dt, verbose):

        # Estimation over the test trajectories with T_star
        traj_folder = os.path.join(self.results_folder, "Test_trajectories_noise")
        tq, simulation = self.system.simulate(init_state, t_sim, dt)
        noise = torch.normal(0, 0.01, size=(simulation.shape[0], simulation.shape[1]))
        simulation = simulation.add(noise)
        measurement = self.model.h(simulation)

        # Save these test trajectories
        os.makedirs(traj_folder, exist_ok=True)

        traj_error = 0.0

        for i in range(w_c_array.shape[0]):
            # TODO run predictions in parallel for all test trajectories!!!
            # Need to figure out how to interpolate y in parallel for all
            # trajectories!!!
            self.model.D = self.model.set_D(w_c_array[i])
            y = torch.cat((tq.unsqueeze(1), measurement), dim=1)

            estimation, z_hat = self.model.predict(
                y, t_sim, dt, w_c_array[i], out_z=True
            )
            traj_error += RMSE(simulation, estimation)

            sensitivity = self.model.sensitivity(z_hat)

            current_traj_folder = os.path.join(traj_folder, f"Traj_{i}")
            os.makedirs(current_traj_folder, exist_ok=True)

            self.save_csv(
                simulation.cpu().numpy(),
                os.path.join(current_traj_folder, f"True_traj_{i}.csv"),
            )
            self.save_csv(
                estimation.cpu().numpy(),
                os.path.join(current_traj_folder, f"Estimated_traj_{i}.csv"),
            )

            for j in range(estimation.shape[1]):
                name = "Traj" + str(j) + ".pdf"
                plt.plot(tq, simulation[:, j].detach().numpy(), label=rf"$x_{j + 1}$")
                plt.plot(
                    tq, estimation[:, j].detach().numpy(), label=rf"$\hat{{x}}_{j + 1}$"
                )
                plt.plot(tq, sensitivity[:, j].detach().numpy())
                plt.legend(loc=1)
                plt.title(rf"Trajectory for $\omega_c$ {w_c_array[i]}")
                plt.xlabel(rf"$t$")
                plt.ylabel(rf"$x_{j + 1}$")
                plt.savefig(
                    os.path.join(current_traj_folder, name), bbox_inches="tight"
                )
                if verbose:
                    plt.show()
                plt.clf()
                plt.close("all")

    def plot_rmse_error(self, init_state, w_c_arr, verbose, tsim, dt, std=0.0):
        with torch.no_grad():
            # Estimation over the test trajectories with T_star
            nb_trajs = w_c_arr.shape[0]
            traj_folder = os.path.join(self.results_folder, "Test_trajectories_RMSE_{}".format(str(std)))
            tq, simulation = self.system.simulate(init_state, tsim, dt)

            noise = torch.normal(0, std, size=(simulation.shape[0], 2))

            measurement_noise = simulation.add(noise)

            measurement = self.model.h(measurement_noise)

            # Save these test trajectories
            os.makedirs(traj_folder, exist_ok=True)
            traj_error = 0.0

            plot_style = ["-", "--", "-."]

            for i in range(nb_trajs):
                # TODO run predictions in parallel for all test trajectories!!!
                # Need to figure out how to interpolate y in parallel for all
                # trajectories!!!
                y = torch.cat((tq.unsqueeze(1), measurement), dim=1)

                if i < len(w_c_arr):
                    w_c = w_c_arr[i]
                else:
                    print('error')

                estimation = self.model.predict(y, tsim, dt, w_c).detach()
                error = RMSE(simulation, estimation)
                traj_error += error

                current_traj_folder = os.path.join(traj_folder, f"Traj_{i}")
                os.makedirs(current_traj_folder, exist_ok=True)

                filename = f"RMSE_{i}.txt"
                with open(os.path.join(current_traj_folder, filename), "w") as f:
                    print(error.cpu().numpy(), file=f)

                self.save_csv(
                    simulation.cpu().numpy(),
                    os.path.join(current_traj_folder, f"True_traj_{i}.csv"),
                )
                self.save_csv(
                    estimation.cpu().numpy(),
                    os.path.join(current_traj_folder, f"Estimated_traj_{i}.csv"),
                )

                os.makedirs(current_traj_folder, exist_ok=True)

                filename = f"RMSE_{i}.txt"
                with open(os.path.join(current_traj_folder, filename), "w") as f:
                    print(error.cpu().numpy(), file=f)

                # for i in range(simulation.shape[1]):
                name = "Traj" + str(1) + ".pdf"
                plt.plot(
                    tq,
                    RMSE(estimation, simulation, 1).cpu().numpy(),
                    plot_style[i],
                    linewidth=0.8,
                    markersize=1,
                    label=rf"$\omega_c = {float(w_c_arr[i]):0.2g}$",
                )

            plt.legend(loc=1)
            plt.grid(visible=True)
            plt.title(rf"Test trajectory RMSE")
            plt.xlabel(rf"$t$")
            plt.ylabel(rf"$\hat{{x}}-x$")
            plt.savefig(os.path.join(current_traj_folder, name), bbox_inches="tight")
            if verbose:
                plt.show()
            plt.clf()
            plt.close("all")

            filename = "RMSE_traj.txt"
            with open(os.path.join(traj_folder, filename), "w") as f:
                print(traj_error, file=f)
    

    def save_results(
        self, checkpoint_path=None,
    ):
        """
        Save the model, the training and validation data. Also saving several
        metrics used for evaluating the results: heatmap of estimation error
        and estimation of several test trajectories.

        Parameters
        ----------
        checkpoint_path: str
            Path to the checkpoint from which to retrieve the best obtained
            model.
        """
        with torch.no_grad():
            if checkpoint_path:
                checkpoint_model = torch.load(checkpoint_path)
                self.load_state_dict(checkpoint_model["state_dict"])

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

            # No control theoretic evaluation of the observer with only T
            if self.method == "T":
                return 0

            # Add t_c to specifications
            with open(specs_file, "a") as f:
                print(f"k {self.model.k}", file=f)
                print(f"t_c {self.model.t_c}", file=f)

