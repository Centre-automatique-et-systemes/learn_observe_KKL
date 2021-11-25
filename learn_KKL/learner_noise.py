import os

import random as rd

from learn_KKL.learner import Learner

import matplotlib.pyplot as plt
import numpy as np
import torch

from .utils import RMSE

# Set double precision by default
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)


class LearnerNoise(Learner):

    def save_random_traj(self, x_mesh, w_c_arr, nb_trajs, verbose, tsim, dt):
        # Estimation over the test trajectories with T_star
        nb_trajs += w_c_arr.shape[0]

        random_idx = np.random.choice(np.arange(x_mesh.shape[0]),
                                      size=(nb_trajs,))
        trajs_init = x_mesh[random_idx]
        traj_folder = os.path.join(self.results_folder, 'Test_trajectories')
        tq, simulation = self.system.simulate(trajs_init, tsim, dt)
        measurement = self.model.h(simulation)
        noise = torch.normal(0, 0.01, size=(measurement.shape[0], 1)).repeat(1, nb_trajs).unsqueeze(2)
        measurement = measurement.add(noise)

        # Save these test trajectories
        os.makedirs(traj_folder, exist_ok=True)
        traj_error = 0.

        for i in range(nb_trajs):
            # TODO run predictions in parallel for all test trajectories!!!
            # Need to figure out how to interpolate y in parallel for all
            # trajectories!!!
            y = torch.cat((tq.unsqueeze(1), measurement[:, i]), dim=1)

            if i < len(w_c_arr):
                w_c = w_c_arr[i]
            else:
                w_c = rd.uniform(0.5, 1.5)

            estimation = self.model.predict(y, tsim, dt, w_c).detach()
            traj_error += RMSE(simulation[:, i], estimation)

            current_traj_folder = os.path.join(traj_folder, f'Traj_{i}')
            os.makedirs(current_traj_folder, exist_ok=True)

            self.save_csv(simulation[:, i].cpu().numpy(), os.path.join(current_traj_folder, f'True_traj_{i}.csv'))
            self.save_csv(estimation.cpu().numpy(), os.path.join(current_traj_folder, f'Estimated_traj_{i}.csv'))

            for j in range(estimation.shape[1]):
                name = 'Traj' + str(j) + '.pdf'
                plt.plot(tq, simulation[:, i, j].detach().numpy(),
                         label=rf'$x_{j + 1}$')
                plt.plot(tq, estimation[:, j].detach().numpy(),
                         label=rf'$\hat{{x}}_{j + 1}$')
                plt.legend()
                plt.title(rf'Random trajectory for $w_c$ {w_c}')
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

    def save_trj(self, init_state, w_c_arr, nb_trajs, verbose, tsim, dt):
        # Estimation over the test trajectories with T_star
        nb_trajs += w_c_arr.shape[0]
        traj_folder = os.path.join(self.results_folder, 'Test_trajectories')
        tq, simulation = self.system.simulate(init_state, tsim, dt)
        noise = torch.normal(0, 0.01, size=(simulation.shape[0], simulation.shape[1]))
        simulation = simulation.add(noise)
        measurement = self.model.h(simulation)

        # Save these test trajectories
        os.makedirs(traj_folder, exist_ok=True)
        traj_error = 0.

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
            traj_error += RMSE(simulation, estimation)

            current_traj_folder = os.path.join(traj_folder, f'Traj_{i}')
            os.makedirs(current_traj_folder, exist_ok=True)

            self.save_csv(simulation.cpu().numpy(), os.path.join(current_traj_folder, f'True_traj_{i}.csv'))
            self.save_csv(estimation.cpu().numpy(), os.path.join(current_traj_folder, f'Estimated_traj_{i}.csv'))

            for j in range(estimation.shape[1]):
                name = 'Traj' + str(j) + '.pdf'
                plt.plot(tq, simulation[:, j].detach().numpy(),
                         label=rf'$x_{j + 1}$')
                plt.plot(tq, estimation[:, j].detach().numpy(),
                         label=rf'$\hat{{x}}_{j + 1}$')
                plt.legend()
                plt.title(rf'Random trajectory for $w_c$ {w_c}')
                plt.xlabel(rf'$t$')
                plt.ylabel(rf'$x_{j + 1}$')
                plt.savefig(os.path.join(current_traj_folder, name),
                            bbox_inches='tight')
                if verbose:
                    plt.show()
                plt.clf()
                plt.close('all')

        filename = 'RMSE_traj.txt'
        with open(os.path.join(traj_folder, filename), 'w') as f:
            print(traj_error, file=f)

    def save_rmse_wc(self, mesh, w_c_array, verbose):
        errors = np.zeros((len(w_c_array)))

        for j in range(mesh.shape[-1]):
            x_mesh = mesh[:, :self.model.dim_x, j]
            z_mesh = mesh[:, self.model.dim_x:, j]

            noise = torch.normal(0, 0.01, size=(z_mesh.shape[0], z_mesh.shape[1]))
            z_mesh.add(noise)

            # compute x_hat for every w_c
            x_hat_star = self.model('T_star', z_mesh)

            errors[j] = RMSE(x_mesh, x_hat_star).detach().numpy()

        # https://stackoverflow.com/questions/37822925/how-to-smooth-by-interpolation-when-using-pcolormesh
        name = 'RMSE_w_c.pdf'
        plt.plot(w_c_array, errors[:,-1])
        plt.title(r'RMSE between $x$ and $\hat{x}$')
        plt.xlabel(rf'$w_c$')
        plt.ylabel(r'RMSE($x$, $\hat{x}$)')
        plt.savefig(os.path.join(self.results_folder, name),
                    bbox_inches='tight')
        if verbose:
            plt.show()
            plt.close('all')

        plt.clf()
        plt.close('all')

    def save_pdf_heatmap(self, mesh, verbose):
        for j in range(mesh.shape[-1]):
            x_mesh = mesh[:, :self.model.dim_x, j]
            z_mesh = mesh[:, self.model.dim_x:, j]
            w_c = z_mesh[0, -1]

            # compute x_hat for every w_c
            x_hat_star = self.model('T_star', z_mesh)

            error = RMSE(x_mesh, x_hat_star, dim=1)

            for i in range(1, x_mesh.shape[1]):
                # https://stackoverflow.com/questions/37822925/how-to-smooth-by-interpolation-when-using-pcolormesh
                name = 'RMSE_heatmap' + str(j) + '.pdf'
                plt.scatter(x_mesh[:, i - 1], x_mesh[:, i], cmap='jet',
                            c=np.log(error.detach().numpy()))
                cbar = plt.colorbar()
                cbar.set_label('Log estimation error')
                cbar.set_label('Log estimation error')
                plt.title(r'RMSE between $x$ and $\hat{x}$' + rf' with $w_c =$ {round(w_c.item(), 2)}')
                plt.xlabel(rf'$x_{i}$')
                plt.ylabel(rf'$x_{i + 1}$')
                plt.legend()
                plt.savefig(os.path.join(self.results_folder, name),
                            bbox_inches='tight')
                if verbose:
                    plt.show()
                    plt.close('all')

                plt.clf()
                plt.close('all')

    def plot_sensitiviy_wc(self, mesh, w_c_array, verbose):
        errors = np.zeros((len(w_c_array), 3))

        for j in range(len(w_c_array)):
            z_mesh = mesh[:, self.model.dim_x:, j]
            self.model.D = self.model.place_poles(w_c_array[j])
            errors[j] = self.model.sensitivity_norm(z_mesh)
            print('||dTdz||/N, h-infinity, product')
            print(errors[j])

        self.save_csv(errors.numpy(), os.path.join(self.results_folder, 'sensitivity.csv'))
        name = 'sensitivity_wc.pdf'
        plt.plot(w_c_array, errors)
        plt.title('Sensitivity')
        plt.xlabel(r'$w_c$')
        plt.ylabel(r'$||\frac{\partial{T^*}}{\partial{z} }(z) D^{-1} F||$')
        plt.savefig(os.path.join(self.results_folder, name),
                    bbox_inches='tight')

        if verbose:
            plt.show()
            plt.close('all')

        plt.clf()
        plt.close('all')

    def plot_traj_sens(self, init_state, w_c_array, t_sim, dt, verbose):

        # Estimation over the test trajectories with T_star
        traj_folder = os.path.join(self.results_folder, 'Test_trajectories_noise')
        tq, simulation = self.system.simulate(init_state, t_sim, dt)
        noise = torch.normal(0, 0.01, size=(simulation.shape[0], simulation.shape[1]))
        simulation = simulation.add(noise)
        measurement = self.model.h(simulation)

        # Save these test trajectories
        os.makedirs(traj_folder, exist_ok=True)
        traj_error = 0.

        for i in range(w_c_array.shape[0]):
            # TODO run predictions in parallel for all test trajectories!!!
            # Need to figure out how to interpolate y in parallel for all
            # trajectories!!!
            self.model.D = self.model.place_poles(w_c_array[i])
            y = torch.cat((tq.unsqueeze(1), measurement), dim=1)

            estimation, z_hat = self.model.predict(y, t_sim, dt, w_c_array[i], out_z=True)
            traj_error += RMSE(simulation, estimation)

            sensitivity = self.model.sensitivity(z_hat)

            current_traj_folder = os.path.join(traj_folder, f'Traj_{i}')
            os.makedirs(current_traj_folder, exist_ok=True)

            self.save_csv(simulation.cpu().numpy(), os.path.join(current_traj_folder, f'True_traj_{i}.csv'))
            self.save_csv(estimation.cpu().numpy(), os.path.join(current_traj_folder, f'Estimated_traj_{i}.csv'))

            for j in range(estimation.shape[1]):
                name = 'Traj' + str(j) + '.pdf'
                plt.plot(tq, simulation[:, j].detach().numpy(),
                         label=rf'$x_{j + 1}$')
                plt.plot(tq, estimation[:, j].detach().numpy(),
                         label=rf'$\hat{{x}}_{j + 1}$')
                plt.plot(tq, sensitivity[:, j].detach().numpy())
                plt.legend()
                plt.title(rf'Trajectory for $w_c$ {w_c_array[i]}')
                plt.xlabel(rf'$t$')
                plt.ylabel(rf'$x_{j + 1}$')
                plt.savefig(os.path.join(current_traj_folder, name),
                            bbox_inches='tight')
                if verbose:
                    plt.show()
                plt.clf()
                plt.close('all')

        filename = 'RMSE_traj.txt'
        with open(os.path.join(traj_folder, filename), 'w') as f:
            print(traj_error, file=f)

    def save_results(self, limits: np.array, wc_arr_train, nb_trajs=10, t_sim=(0, 60),
                     dt=1e-2, num_samples=80000, checkpoint_path=None,
                     verbose=False, fast=False):
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
                self.load_state_dict(checkpoint_model['state_dict'])

            # Save training and validation data
            idx = np.random.choice(np.arange(len(self.training_data)),
                                   size=(10000,))  # subsampling for plots

            specs_file = self.save_specifications()

            self.save_pkl('/model.pkl', self.model)
            self.save_pkl('/learner.pkl', self)

            self.save_csv(self.training_data.cpu().numpy(), os.path.join(self.results_folder, 'training_data.csv'))
            self.save_csv(self.validation_data.cpu().numpy(), os.path.join(self.results_folder, 'validation_data.csv'))

            # self.save_pdf_training(self.training_data[idx], verbose)

            # No control theoretic evaluation of the observer with only T
            if self.method == 'T':
                return 0

            # self.save_trj(torch.tensor([1., 1.]), wc_arr_train, nb_trajs, verbose, t_sim, dt)

            # create array of w_c from [0.1, ..., 1]
            # w_c_array = torch.arange(0.2, 0.9, 0.2)

            mesh = self.model.generate_data_svl(limits, wc_arr_train, 30000,
                                                method='LHS', stack=False)

            # self.save_rmse_wc(mesh, wc_arr_train, verbose)
            self.plot_sensitiviy_wc(mesh, wc_arr_train, verbose)

            # generate data
            mesh_1 = self.model.generate_data_svl(limits, wc_arr_train, num_samples,
                                                method='uniform', stack=False)

            self.save_pdf_heatmap(mesh_1, verbose)

            # Loss plot over time and loss heatmap over space
            self.save_plot('Train_loss.pdf', 'Training loss over time', 'log', self.train_loss.detach())
            self.save_plot('Val_loss.pdf', 'Validation loss over time', 'log', self.val_loss.detach())

            # Add t_c to specifications
            with open(specs_file, 'a') as f:
                print(f'k {self.model.k}', file=f)
                print(f't_c {self.model.t_c}', file=f)
