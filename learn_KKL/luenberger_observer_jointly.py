# -*- coding: utf-8 -*-

import torch
from torch import nn
from functorch import vmap, jacrev

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
            method: str = "Autoencoder_jointly",
            dim_z: int = None,
            wc: float = 1.0,
            num_hl: int = 5,
            size_hl: int = 50,
            activation=nn.ReLU(),
            recon_lambda: float = 1.0,
            sensitivity_lambda=0,
            D="block_diag",
            solver_options=None,
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
            solver_options,
        )
        self.sensitivity_lambda = sensitivity_lambda
        if self.sensitivity_lambda > 0:
            raise Exception('Adding a sensitivity term to the AE loss is '
                            'experimental and should be corrected first!')

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
                "method_setD " + str(self.method_setD),
                "wc " + str(self.wc),
                "D_0 " + str(self.D_0),
                "D " + str(self.D),
                "F " + str(self.F),
                "encoder " + str(self.encoder),
                "decoder " + str(self.decoder),
                "method " + self.method,
                "recon_lambda " + str(self.recon_lambda),
                "sensitivity_lambda " + str(self.sensitivity_lambda),
            ]
        )

    def loss_autoencoder_jointly(
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

         if self.sensitivity_lambda > 0:
             # Compute gradients of T_star with respect to inputs
             dTdz = vmap(jacrev(self.decoder))(z_hat)
             dTdz = dTdz[:, :, : self.dim_z]  # TODO correct this loss!!!
             loss_3 = self.sensitivity_lambda * torch.linalg.norm(
                 torch.matmul(dTdz, torch.matmul(torch.inverse(self.D), self.F)))

         else:
             loss_3 = torch.zeros_like(loss)

         return loss + loss_3, loss_1, loss_2, loss_3


    def loss(self, method="Autoencoder_jointly", *input):
        if method == "T":
            return self.loss_T(*input)
        elif method == "T_star":
            return self.loss_T_star(*input)
        elif method == "Autoencoder":
            return self.loss_autoencoder(*input)
        elif method == "Autoencoder_jointly":
            return self.loss_autoencoder_jointly(*input)

