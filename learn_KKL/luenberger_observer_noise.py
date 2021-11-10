# -*- coding: utf-8 -*-

import numpy as np
import torch
from scipy import linalg
from torch import nn

from learn_KKL.luenberger_observer import LuenbergerObserver

from .utils import generate_mesh

# Set double precision by default
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)


class LuenbergerObserverNoise(LuenbergerObserver):

    def __init__(self, dim_x: int, dim_y: int, method: str = "Autoencoder",
                 dim_z: int = None, wc: float = 1., num_hl: int = 5,
                 size_hl: int = 50, activation=nn.ReLU(),
                 recon_lambda: float = 1., D='bessel'):

        LuenbergerObserver.__init__(self, dim_x, dim_y, method,
                                    dim_z, wc, num_hl,
                                    size_hl, activation,
                                    recon_lambda, D)

        self.encoder_layers = self.create_layers(
            self.num_hl, self.size_hl, self.activation, self.dim_x, self.dim_z + 1)
        self.decoder_layers = self.create_layers(
            self.num_hl, self.size_hl, self.activation, self.dim_z + 1, self.dim_x)

    def __repr__(self):
        return '\n'.join([
            'Luenberger Observer Noise object',
            'dim_x ' + str(self.dim_x),
            'dim_y ' + str(self.dim_y),
            'dim_z ' + str(self.dim_z),
            'wc ' + str(self.wc),
            'D ' + str(self.D),
            'F ' + str(self.F),
            'encoder ' + str(self.encoder_layers),
            'decoder ' + str(self.decoder_layers),
            'method ' + self.method,
            'recon_lambda ' + str(self.recon_lambda),
        ])

    def generate_data_mesh(self, limits: tuple, num_samples: int, k: int = 10,
                           dt: float = 1e-2, method: str = 'LHS'):
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
        num_samples = mesh.shape[0]  # in case regular grid: changed
        self.k = k
        self.t_c = self.k / min(abs(linalg.eig(self.D)[0].real))

        y_0 = torch.zeros((num_samples, self.dim_x + self.dim_z))  # TODO
        y_1 = y_0.clone()

        # Simulate only x system backward in time
        tsim = (0, -self.t_c)
        y_0[:, :self.dim_x] = mesh
        _, data_bw = self.simulate_system(y_0, tsim, -dt, only_x=True)

        # Simulate both x and z forward in time starting from the last point
        # from previous simulation
        tsim = (-self.t_c, 0)
        y_1[:, :self.dim_x] = data_bw[-1, :, :self.dim_x]
        _, data_fw = self.simulate_system(y_1, tsim, dt)

        # Data contains (x_i, z_i) pairs in shape [dim_x + dim_z,
        # number_simulations]
        data = data_fw[-1]
        return data

    def generate_data_svl(self, limits: np.array, w_c: np.array, num_datapoints: int, k: int = 10,
                          dt: float = 1e-2, method: str = 'LHS'):

        num_samples = int(np.ceil(num_datapoints / len(w_c)))

        df = torch.zeros(size=(num_samples*len(w_c), self.dim_x+self.dim_z + 1))

        for idx, w_c_i in np.ndenumerate(w_c):
            self.D = self.bessel_D(w_c_i)

            data = self.generate_data_mesh(limits, num_samples, k, dt, method)

            wc_i_tensor = torch.tensor(w_c_i).repeat(num_samples).unsqueeze(1)
            data = torch.cat((data, wc_i_tensor), 1)

            df[idx[0] * num_samples: (idx[0]+1) * num_samples, ] = data

        return df

    def predict(self, measurement: torch.tensor, tsim: tuple,
                dt: int, w_c: float) -> torch.tensor:
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
        self.D = self.bessel_D(w_c)

        _, sol = self.simulate(measurement, tsim, dt)

        w_c_tensor = torch.tensor(w_c).repeat(sol.shape[0]).unsqueeze(1)

        z_hat = torch.cat((sol[:, :, 0], w_c_tensor), 1)
        x_hat = self.decoder(z_hat)

        return x_hat
