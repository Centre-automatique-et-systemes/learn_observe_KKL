# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from seaborn import utils
import torch
from scipy import linalg
from torch import nn
from torchdiffeq import odeint

from learn_KKL.luenberger_observer import LuenbergerObserver

from .utils import RMSE, generate_mesh, compute_h_infinity, MLPn

# Set double precision by default
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)


class LuenbergerObserverNoise(LuenbergerObserver):
    def __init__(
            self,
            dim_x: int,
            dim_y: int,
            method: str = "Supervised_noise",
            dim_z: int = None,
            wc_array=np.array([1.0]),
            num_hl: int = 5,
            size_hl: int = 50,
            activation=nn.ReLU(),
            recon_lambda: float = 1.0,
            D="block_diag",
            solver_options=None,
    ):

        LuenbergerObserver.__init__(
            self,
            dim_x,
            dim_y,
            method,
            dim_z,
            wc_array[0],
            num_hl,
            size_hl,
            activation,
            recon_lambda,
            D,
            solver_options,
        )

        self.wc_array = wc_array
        self.encoder = MLPn(num_hl=self.num_hl, n_in=self.dim_x + 1,
                            n_hl=self.size_hl, n_out=self.dim_z,
                            activation=self.activation)
        self.decoder = MLPn(num_hl=self.num_hl, n_in=self.dim_z + 1,
                            n_hl=self.size_hl, n_out=self.dim_x,
                            activation=self.activation)

    def set_scalers(self, scaler_xin, scaler_xout, scaler_zin, scaler_zout):
        """
        Set the scaler objects for input and output data. Then, the internal
        NN will normalize every input and denormalize every output. These
        scalers are different for the encoder and the decoder due to wc being
        an extra input to each NN.

        Parameters
        ----------
        scaler_X :
            Input scaler.

        scaler_out :
            Output scaler.
        """
        self.scaler_xin = scaler_xin
        self.scaler_xout = scaler_xout
        self.scaler_zin = scaler_zin
        self.scaler_zout = scaler_zout
        self.encoder.set_scalers(scaler_X=self.scaler_xin,
                                 scaler_Y=self.scaler_zout)
        self.decoder.set_scalers(scaler_X=self.scaler_zin,
                                 scaler_Y=self.scaler_xout)

    def __repr__(self):
        return "\n".join(
            [
                "Luenberger Observer Noise object",
                "dim_x " + str(self.dim_x),
                "dim_y " + str(self.dim_y),
                "dim_z " + str(self.dim_z),
                "wc_array " + str(self.wc_array),
                "nb_wc " + str(len(self.wc_array)),
                "D " + str(self.D),
                "F " + str(self.F),
                "encoder " + str(self.encoder),
                "decoder " + str(self.decoder),
                "method " + self.method,
                "recon_lambda " + str(self.recon_lambda),
                "solver_options " + str(self.solver_options)
            ]
        )

    def generate_data_mesh(
            self,
            limits: tuple,
            num_samples: int,
            k: int = 10,
            dt: float = 1e-2,
            method: str = "LHS",
            z_0=None,
            **kwargs
    ):
        return LuenbergerObserver.generate_data_svl(
            self, limits, num_samples, k, dt, method, z_0, **kwargs)

    def generate_data_svl(self, limits: np.array, w_c: np.array,
                          num_datapoints: int, k: int = 10, dt: float = 1e-2,
                          stack: bool = True, method: str = "LHS", z_0=None,
                          **kwargs):

        num_samples = int(np.ceil(num_datapoints / len(w_c)))

        df = torch.zeros(
            size=(num_samples, self.dim_x + self.dim_z + 1, len(w_c)))

        for idx, w_c_i in np.ndenumerate(w_c):
            self.D, self.F = self.set_DF(w_c_i)

            data = self.generate_data_mesh(limits, num_samples, k, dt,
                                           method, z_0=z_0, w_c=w_c_i)

            wc_i_tensor = torch.tensor(w_c_i).repeat(num_samples).unsqueeze(1)
            data = torch.cat((data, wc_i_tensor), 1)

            df[..., idx] = data.unsqueeze(-1)

        if stack:
            return torch.cat(torch.unbind(df, dim=-1), dim=0)
        else:
            return df

    def generate_trajectory_data(
            self,
            limits: tuple,
            w_c: np.array,
            num_samples: int,
            tsim: tuple,
            k: int = 10,
            dt: float = 1e-2,
            method: str = "LHS",
            stack: bool = True,
            z_0: bool = None
    ):
        """
        Generate data points by simulating the system forward in time from
        some initial conditions, which are sampled with LHS or uniform,
        then z(0) is obtained with backward/forward sampling.
        Parameters
        ----------
        limits: tuple
            Limits in which to draw the initial conditions x(0).
        w_c: np.array
            Array of w_c values for which to simulate.
        num_samples: int
            Number of initial conditions.
        k: int
           Parameter for time t_c = k/min(lambda) before which to cut.
        dt: float
            Simulation step.
        method: string
            Method for sampling the initial conditions.
        stack: bool
            Whether to stack the data, see output.

        Returns
        ----------
        data: torch.tensor
            Pairs of (x, z) data points, in shape (tsim, num_samples, dx+dz)
            if stack is False, shape (tsim * num_samples, dx+dz) if True.
        """
        # Get initial conditions for x,z from backward forward sampling
        # num_datapoints = num_samples * len(w_c)
        # y_0 = self.generate_data_svl(
        #     limits=limits, w_c=w_c, num_datapoints=num_datapoints, method=method,
        #     k=k, dt=dt
        # )
        # df = torch.zeros(
        #     size=(num_samples, self.dim_x + self.dim_z + 1, len(w_c)))

        tq = torch.arange(tsim[0], tsim[1], dt)
        df = torch.zeros(
            size=(len(tq), num_samples, self.dim_x + self.dim_z + 1, len(w_c)))

        for idx, w_c_i in np.ndenumerate(w_c):
            self.D, self.F = self.set_DF(w_c_i)

            # Get initial conditions for this wv with B/F sampling
            y_0 = self.generate_data_mesh(limits, num_samples, k, dt, method,
                                          z_0=z_0, w_c=w_c_i)
            # Simulate x(t), z(t) to obtain trajectories for tsim
            _, data = self.simulate_system(y_0, tsim, dt)

            wc_i_tensor = torch.tensor(w_c_i).repeat(
                (data.shape[0], data.shape[1])).unsqueeze(-1)
            data = torch.cat((data, wc_i_tensor), -1)

            df[..., idx] = data.unsqueeze(-1)

        # Fix issue with grad tensor in pipeline
        if stack:
            return torch.cat(torch.unbind(df, dim=1), dim=0)
        else:
            return df

    def generate_data_forward(self, init: torch.tensor, w_c: np.array,
                              tsim: tuple, num_datapoints: int, k: int = 10,
                              dt: float = 1e-2, stack: bool = True):
        """
        Generate data points by simulating the system forward in time from
        some initial conditions, then cutting the beginning of the trajectory.
        Parameters
        ----------
        init: torch.tensor
            Initial conditions (x0, z0) from which to simulate.
        w_c: np.array
            Array of w_c values for which to simulate.
        tsim: tuple
            Simulation time.
        num_datapoints: int
            Number of samples to take along trajectory * len(w_c) (convention).
        k: int
           Parameter for time t_c = k/min(lambda) before which to cut.
        dt: float
            Simulation step.
        Returns
        ----------
        df: torch.tensor
            Pairs of (x, z, wc) data points.
        """
        num_samples = int(np.ceil(num_datapoints / len(w_c)))

        df = torch.zeros(
            size=(num_samples, len(init), self.dim_x + self.dim_z + 1,
                  len(w_c)))

        for idx, w_c_i in np.ndenumerate(w_c):
            self.D, self.F = self.set_DF(w_c_i)

            tq, data = self.simulate_system(init, tsim, dt)
            self.k = k
            self.t_c = self.k / min(
                abs(linalg.eig(self.D.detach().numpy())[0].real))
            data = data[(tq >= self.t_c)]  # cut trajectory before t_c
            random_idx = np.random.choice(np.arange(len(data)),
                                          size=(num_samples,), replace=False)
            data = torch.squeeze(data[random_idx])

            wc_i_tensor = torch.tensor(w_c_i).repeat(
                (data.shape[0], data.shape[1])).unsqueeze(-1)
            data = torch.cat((data, wc_i_tensor), 1)

            df[..., idx] = data.unsqueeze(-1)

        if stack:
            return torch.cat(torch.unbind(df, dim=-1), dim=0)
        else:
            return df

    def sensitivity_norm(self, x, z, save=True, path='', version=9):
        print('Python version of our gain-tuning criterion: the estimation of '
              'the H-infinity norm is not very smooth and we have replaced '
              'the H-2 norm by a second H-infinity norm, hence, the Matlab '
              'script criterion.m should be used instead to compute the final '
              'criterion as it was in the paper.')
        if save:
            # TODO more efficient computation for dNN/dx(x)! Symbolic?JAX?
            # Compute dTdx over grid
            dTdh = torch.autograd.functional.jacobian(
                self.encoder, x, create_graph=False, strict=False,
                vectorize=False
            )
            dTdx = torch.transpose(
                torch.transpose(
                    torch.diagonal(dTdh, dim1=0, dim2=2), 1, 2), 0, 1
            )
            dTdx = dTdx[:, :, : self.dim_x]
            idx_max = torch.argmax(torch.linalg.matrix_norm(dTdx, ord=2))
            Tmax = dTdx[idx_max]

            # Compute dTstar_dz over grid
            dTstar_dh = torch.autograd.functional.jacobian(
                self.decoder, z, create_graph=False, strict=False,
                vectorize=False
            )
            dTstar_dz = torch.transpose(
                torch.transpose(
                    torch.diagonal(dTstar_dh, dim1=0, dim2=2), 1, 2), 0, 1
            )
            dTstar_dz = dTstar_dz[:, :, : self.dim_z]
            idxstar_max = torch.argmax(
                torch.linalg.matrix_norm(dTstar_dz, ord=2))
            Tstar_max = dTstar_dz[idxstar_max]

            # Save this data
            wc = z[0, -1].item()
            file = pd.DataFrame(Tmax)
            file.to_csv(os.path.join(path, f'Tmax_wc{wc:0.2g}.csv'),
                        header=False)
            file = pd.DataFrame(Tstar_max)
            file.to_csv(os.path.join(path, f'Tstar_max_wc{wc:0.2g}.csv'),
                        header=False)
            file = pd.DataFrame(dTdx.flatten(1, -1))
            file.to_csv(os.path.join(path, f'dTdx_wc{wc:0.2g}.csv'),
                        header=False)
            file = pd.DataFrame(dTstar_dz.flatten(1, -1))
            file.to_csv(os.path.join(path, f'dTstar_dz_wc{wc:0.2g}.csv'),
                        header=False)
        else:
            # Load intermediary data
            wc = z[0, -1].item()
            df = pd.read_csv(os.path.join(path, f'Tmax_wc{wc:0.2g}.csv'),
                             sep=',', header=None)
            Tmax = torch.from_numpy(df.drop(df.columns[0], axis=1).values)
            df = pd.read_csv(os.path.join(path, f'dTdx_wc{wc:0.2g}.csv'),
                             sep=',', header=None)
            dTdx = torch.from_numpy(
                df.drop(df.columns[0], axis=1).values).reshape(
                (-1, Tmax.shape[0], Tmax.shape[1]))
            df = pd.read_csv(os.path.join(path, f'Tstar_max_wc{wc:0.2g}.csv'),
                             sep=',', header=None)
            Tstar_max = torch.from_numpy(df.drop(df.columns[0], axis=1).values)
            df = pd.read_csv(os.path.join(path, f'dTstar_dz_wc{wc:0.2g}.csv'),
                             sep=',', header=None)
            dTstar_dz = torch.from_numpy(
                df.drop(df.columns[0], axis=1).values).reshape(
                (-1, Tstar_max.shape[0], Tstar_max.shape[1]))

        if version == 1:
            C = np.eye(self.dim_z)
            sv = torch.tensor(
                compute_h_infinity(self.D.numpy(), self.F.numpy(), C, 1e-10))
            product = torch.linalg.matrix_norm(Tstar_max, ord=2) * sv
            return torch.cat(
                (torch.linalg.matrix_norm(Tstar_max, ord=2).unsqueeze(0),
                 sv.unsqueeze(0), product.unsqueeze(0)), dim=0
            )
        elif version == 2:
            C = np.eye(self.dim_z)
            sv = torch.tensor(compute_h_infinity(
                self.D.numpy(), self.F.numpy(),
                np.dot(Tstar_max.detach().numpy(),
                       C), 1e-3))
            product = sv
            return torch.cat(
                (torch.linalg.matrix_norm(Tstar_max, ord=2).unsqueeze(0),
                 sv.unsqueeze(0), product.unsqueeze(0)), dim=0
            )
        elif version == 3:
            C = np.eye(self.dim_x)
            sv = torch.tensor(compute_h_infinity(
                np.dot(np.dot(Tstar_max.detach().numpy(),
                              self.D.numpy()), Tmax.numpy()),
                np.dot(Tstar_max.detach().numpy(), self.F.numpy()),
                C, 1e-3))
            return torch.cat(
                (torch.linalg.matrix_norm(Tmax, ord=2).unsqueeze(0),
                 torch.linalg.matrix_norm(Tstar_max, ord=2).unsqueeze(0),
                 sv.unsqueeze(0)), dim=0
            )
        elif version == 9:
            C = np.eye(self.dim_z)
            l2_norm = torch.linalg.norm(
                torch.linalg.matrix_norm(dTstar_dz, dim=(1, 2), ord=2))
            sv1 = torch.tensor(
                compute_h_infinity(self.D.numpy(), self.F.numpy(), C, 1e-10))
            sv2 = torch.tensor(  # TODO implement H2 norm instead!
                compute_h_infinity(self.D.numpy(), np.eye(self.dim_z), C,
                                   1e-10))
            product = l2_norm * (sv1 + sv2)
            return torch.cat(
                (l2_norm.unsqueeze(0), sv1.unsqueeze(0),
                 product.unsqueeze(0)), dim=0
            )
        else:
            raise NotImplementedError(f'Gain-tuning criterion version '
                                      f'{version} is not implemented.')

    def predict(
            self,
            measurement: torch.tensor,
            t_sim: tuple,
            dt: int,
            w_c: float,
            out_z: bool = False,
            z_0=None
    ) -> torch.tensor:
        """
        Forward function for autoencoder. Used for training the model.
        Computation follows as:
        z = encoder(x)
        x_hat = decoder(z)

        Parameters
        ----------
        measurement: torch.tensor
            Measurement y of the state vector of the system driving the observer.

        tsim: tuple
            Tuple of (Start, End) time of simulation.

        dt: float 
            Step width of tsim.

        Returns
        ----------
        x_hat: torch.tensor
            Estimation of the observer model.
        """
        self.D, _ = self.set_DF(w_c)

        _, sol = self.simulate(measurement, t_sim, dt, z_0)

        w_c_tensor = torch.tensor(w_c).repeat(sol.shape[0]).unsqueeze(1)

        z_hat = torch.cat((sol[:, :, 0], w_c_tensor), 1)
        x_hat = self.decoder(z_hat)

        if out_z:
            return x_hat, z_hat
        else:
            return x_hat
