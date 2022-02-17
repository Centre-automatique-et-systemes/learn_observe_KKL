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
            D="block_diag",
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

    @property
    def D(self):
        # Whenever self.D is called, D_scaled * ||D0|| is returned
        # Only D_scaled is optimized jointly
        return self.D_scaled * self.norm_D_0

    @D.setter
    def D(self, D):
        # Require grad for D so that gets optimized jointly with NN weights
        # Keep its initial value in variables
        # Also scale it: D = D / ||D0||, KKL ODE with D * ||D0||
        # This is called in LuenbergerObserver.__init__ when self.D is set
        self.D_0 = D.detach().clone()
        self.norm_D_0 = torch.linalg.norm(self.D_0.detach().clone())
        D_init = self.D_0.detach().clone() / self.norm_D_0
        self.D_scaled = torch.nn.parameter.Parameter(data=D_init,
                                                     requires_grad=True)

    def __repr__(self):
        return "\n".join(
            [
                "Luenberger Observer optimize D jointly object",
                "dim_x " + str(self.dim_x),
                "dim_y " + str(self.dim_y),
                "dim_z " + str(self.dim_z),
                "wc " + str(self.wc),
                "D_0 " + str(self.D_0),
                "D " + str(self.D),
                "F " + str(self.F),
                "encoder " + str(self.encoder),
                "decoder " + str(self.decoder),
                "method " + self.method,
                "recon_lambda " + str(self.recon_lambda),
            ]
        )

    def loss_autoencoder_sensitivity(
             self, x: torch.tensor, x_hat: torch.tensor,
             z_hat: torch.tensor, dim=None) -> torch.tensor:
         """
         Loss function for training the observer model with the autoencoder
         method, adding an extra term to penalize noise sensitivity. See
         reference for detailed information.

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

         loss_3: torch.tensor
             Noise sensitvity loss norm(dTdz*D^{-1}*F).
         """
         loss, loss_1, loss_2 = self.loss_autoencoder(
             x, x_hat, z_hat, dim)

         # Compute gradients of T_star with respect to inputs
         dTdh = torch.autograd.functional.jacobian(
             self.decoder, z_hat, create_graph=False, strict=False, vectorize=False
         )
         dTdz = torch.transpose(
             torch.transpose(torch.diagonal(dTdh, dim1=0, dim2=2), 1, 2), 0, 1
         )
         dTdz = dTdz[:, :, : self.dim_z]  # TODO correct this loss
         # D = self.D.to(self.device)
         # F = self.F.to(self.device)
         # loss_3 = torch.linalg.norm(dTdz,
         #                            torch.matmul(torch.matmul(torch.inverse(D), F)))
         loss_3 = torch.linalg.norm(
             dTdz, torch.matmul(torch.matmul(torch.inverse(self.D), self.F)))

         return loss + loss_3, loss_1, loss_2

