# -*- coding: utf-8 -*-

import numpy as np
import torch
from scipy import linalg
from scipy import signal
from torch import nn
from torchdiffeq import odeint
from torchinterp1d import Interp1d

from learn_KKL.luenberger_observer import LuenbergerObserver

from .utils import MSE, generate_mesh

# Set double precision by default
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)


class LuenbergerObserverNoise(LuenbergerObserver):

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

    def generate_data_mesh(self, limits: np.array, w_c: np.array, num_datapoints: int, k: int = 10,
                           dt: float = 1e-2, method: str = 'LHS'):

        num_samples = int(np.ceil(num_datapoints / len(w_c)))

        df = torch.zeros(size=(num_samples*len(w_c), self.dim_x+self.dim_z + 1))

        # TODO slow method, give D matrix for faster simulation
        for idx, w_c_i in np.ndenumerate(w_c):
            self.D = self.bessel_D(w_c_i)

            data = self.generate_data_svl(limits, num_samples, k, dt, method)

            wc_i_tensor = torch.tensor(w_c_i).repeat(num_samples).unsqueeze(1)
            data = torch.cat((data, wc_i_tensor), 1)

            df[idx[0] * num_samples: (idx[0]+1) * num_samples, ] = data

        return df