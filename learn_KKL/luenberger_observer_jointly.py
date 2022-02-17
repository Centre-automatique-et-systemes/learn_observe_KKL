# -*- coding: utf-8 -*-

import torch
from torch import nn

from learn_KKL.luenberger_observer import LuenbergerObserver

from .utils import MSE, generate_mesh

# Set double precision by default
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)


class LuenbergerObserverJointly(LuenbergerObserver):
    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        method: str = "Autoencoder",
        dim_z: int = None,
        wc: float = 1.0,
        num_hl: int = 5,
        size_hl: int = 50,
        activation=nn.ReLU(),
        recon_lambda: float = 1.0,
        D="bessel",
    ):

        LuenbergerObserver.__init__(
            self,
            dim_x,
            dim_y,
            method,
            dim_z,
            wc,
            num_hl,
            size_hl,
            activation,
            recon_lambda,
            D,
        )

        if D == "bessel":
            init_D = self.place_poles(wc=self.wc)
        else:
            self.wc = 0.0
            init_D = torch.as_tensor(D)

        self.D = torch.nn.parameter.Parameter(data=init_D, requires_grad=True)

    def __repr__(self):
        return "\n".join(
            [
                "Luenberger Observer optimize D jointly object",
                "dim_x " + str(self.dim_x),
                "dim_y " + str(self.dim_y),
                "dim_z " + str(self.dim_z),
                "wc " + str(self.wc),
                "D " + str(self.D),
                "F " + str(self.F),
                "encoder " + str(self.encoder_layers),
                "decoder " + str(self.decoder_layers),
                "method " + self.method,
                "recon_lambda " + str(self.recon_lambda),
            ]
        )

    def loss_autoencoder(
             self, x: torch.tensor, x_hat: torch.tensor,
             z_hat: torch.tensor, dim=None) -> torch.tensor:
         """
         Loss function for training the observer model with the autoencoder
         method. See reference for detailed information.

         Parameters
         ----------
         x: torch.tensor
             State vector of the system driving the observer.

         x_hat: torch.tensor
             Estimation of the observer model.

         z_hat: torch.tensor
             Estimation of the state vector of the observer.

         dim: int
             Dimension along which to take the loss (if None, mean over all
             dimensions).

         Returns
         ----------
         loss: torch.tensor
             Reconstruction loss plus PDE loss.

         loss_1: torch.tensor
             Reconstruction loss MSE(x, x_hat).

         loss_2: torch.tensor
             PDE loss MSE(dTdx*f(x), D*z+F*h(x)).
         """
         # Reconstruction loss MSE(x,x_hat)
         loss_1 = self.recon_lambda * MSE(x, x_hat, dim=dim)

         D = self.D.to(self.device)
         F = self.F.to(self.device)

         loss_2 = torch.norm(torch.matmul(torch.inverse(D), F))

         return loss_1 + loss_2, loss_1, loss_2

