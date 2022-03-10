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
            ]
        )

    def generate_data_mesh(
        self,
        limits: tuple,
        num_samples: int,
        k: int = 10,
        dt: float = 1e-2,
        method: str = "LHS",
    ):
        """
        Generate a grid of data points by simulating the system backward and
        forward in time.

        Parameters
        ----------
        limits: np.array
            Array for the limits of all axes, used for sampling with LHS.
            Form np.array([[min_1, max_1], ..., [min_n, max_n]]).

        num_samples: int
            Number of samples in compact set.

        k: int
            Parameter for t_c = k/min(lambda)

        dt: float
            Simulation step.

        Returns
        ----------
        data: torch.tensor
            Pairs of (x, z) data points.
        """
        mesh = generate_mesh(limits=limits, num_samples=num_samples,
                             method=method)
        self.k = k
        self.t_c = self.k / min(abs(linalg.eig(self.D)[0].real))

        y_0 = torch.zeros((num_samples, self.dim_x + self.dim_z))  # TODO
        y_1 = y_0.clone()

        # Simulate only x system backward in time
        tsim = (0, -self.t_c)
        y_0[:, : self.dim_x] = mesh
        _, data_bw = self.simulate_system(y_0, tsim, -dt, only_x=True)

        # Simulate both x and z forward in time starting from the last point
        # from previous simulation
        tsim = (-self.t_c, 0)
        y_1[:, : self.dim_x] = data_bw[-1, :, : self.dim_x]
        _, data_fw = self.simulate_system(y_1, tsim, dt)

        # Data contains (x_i, z_i) pairs in shape [dim_x + dim_z,
        # number_simulations]
        data = data_fw[-1]
        return data

    def generate_data_svl(self, limits: np.array, w_c: np.array,
                          num_datapoints: int, k: int = 10, dt: float = 1e-2,
                          stack: bool = True, method: str = "LHS"):

        num_samples = int(np.ceil(num_datapoints / len(w_c)))

        df = torch.zeros(size=(num_samples, self.dim_x + self.dim_z + 1, len(w_c)))

        for idx, w_c_i in np.ndenumerate(w_c):
            self.D, self.F = self.set_DF(w_c_i)

            data = self.generate_data_mesh(limits, num_samples, k, dt, method)

            wc_i_tensor = torch.tensor(w_c_i).repeat(num_samples).unsqueeze(1)
            data = torch.cat((data, wc_i_tensor), 1)

            df[..., idx] = data.unsqueeze(-1)

        if stack:
            return torch.cat(torch.unbind(df, dim=-1), dim=0)
        else:
            return df

    def sensitivity_norm(self, x, z, save=True, path='', version=1):
        print('Python version of our gain-tuning criterion: the estimation of '
              'the H-infinity norm is not very smooth, hence, the Matlab '
              'script criterion.m was used instead to generate the final plots '
              'in the paper.')
        if save:
            # Compute dTdx over grid
            dTdh = torch.autograd.functional.jacobian(
                self.encoder, x, create_graph=False, strict=False, vectorize=False
            )
            dTdx = torch.transpose(
                torch.transpose(torch.diagonal(dTdh, dim1=0, dim2=2), 1, 2), 0, 1
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
                torch.transpose(torch.diagonal(dTstar_dh, dim1=0, dim2=2), 1, 2), 0,
                1
            )
            dTstar_dz = dTstar_dz[:, :, : self.dim_z]
            idxstar_max = torch.argmax(torch.linalg.matrix_norm(dTstar_dz, ord=2))
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
            # df = pd.read_csv(os.path.join(path, f'Tmax_wc{wc:0.2g}.csv'),
            #                  sep=',', header=None)
            df = pd.read_csv(os.path.join(path, f'Tmax_wc'
                                                f'{round(float(wc), 2)}.csv'),
                             sep=',', header=None)
            Tmax = torch.from_numpy(df.drop(df.columns[0], axis=1).values)
            # df = pd.read_csv(os.path.join(path, f'dTdx_wc{wc:0.2g}.csv'),
            #                  sep=',', header=None)
            df = pd.read_csv(os.path.join(path, f'dTdx_wc{round(float(wc), 2)}.csv'),
                             sep=',', header=None)
            dTdx = torch.from_numpy(
                df.drop(df.columns[0], axis=1).values).reshape(
                (-1, Tmax.shape[0], Tmax.shape[1]))
            # df = pd.read_csv(os.path.join(path, f'Tstar_max_wc{wc:0.2g}.csv'),
            #                  sep=',', header=None)
            df = pd.read_csv(os.path.join(path, f'Tstar_max_wc{round(float(wc), 2)}.csv'),
                             sep=',', header=None)
            Tstar_max = torch.from_numpy(df.drop(df.columns[0], axis=1).values)
            # df = pd.read_csv(os.path.join(path, f'dTstar_dz_wc{wc:0.2g}.csv'),
            #                  sep=',', header=None)
            df = pd.read_csv(os.path.join(path, f'dTstar_dz_wc{round(float(wc), 2)}.csv'),
                             sep=',', header=None)
            dTstar_dz = torch.from_numpy(
                df.drop(df.columns[0], axis=1).values).reshape(
                (-1, Tmax.shape[0], Tmax.shape[1]))

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
                self.D.numpy(), self.F.numpy(), np.dot(Tstar_max.detach().numpy(),
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

    def predict(
        self,
        measurement: torch.tensor,
        t_sim: tuple,
        dt: int,
        w_c: float,
        out_z: bool = False,
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

        _, sol = self.simulate(measurement, t_sim, dt)

        w_c_tensor = torch.tensor(w_c).repeat(sol.shape[0]).unsqueeze(1)

        z_hat = torch.cat((sol[:, :, 0], w_c_tensor), 1)
        x_hat = self.decoder(z_hat)

        if out_z:
            return x_hat, z_hat
        else:
            return x_hat
