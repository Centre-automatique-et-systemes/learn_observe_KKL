# -*- coding: utf-8 -*-

"""
This module implements the Luenberger Observer for linear and
non-linear system. The Luenberger Observer class implements the
autoencoder model which needs to be trained on a given dynamical
sytem and has an easy to use interface to minimize the parameter
tuning on your side.

The general approach to use this model is as follows. Firstly,
the Luenberger Observer must be initiated with the dimension of
the state variable x and the dimension of the measurement of the
dynamical system. In order to use the observer system the matrices
D and F must be initiated appropriately to the given system dynamics.
By default the matrix D is set by the eigenvalues of a third order
bessel filter with a cutoff frequency of one and matrix F is set to
one.

Then we need to define the functions of the dynamical system. If you
implemented the sytem class from this package, you can use the
set_dynamics(system) method. Alternatively, you can set the functions
in usual procedural style.

Next, we need to train the autoencoder model from a dataset of system states
(x_i). The  learner class in this package gives a simple interface to learn
the model based on pytorch-lightning. However, you might want to overwrite
some learning parameters with your own. With the trained model we are now
able to estimate the state vectors given the measurement y of our system.

This module can also be used to learn the forward and inverse transformations
(encoder and decoder) separately, using supervised learning instead of an
autoencoder. In that case, the user generate_data_svl to generate both the
system states dataset (x_i) and the corresponding observer states dataset (
z_i). Then, the forward transformation T and its inverse T_star are trained
separately using two learner objects. To train the autoencoder,
use method='Autoencoder' when creating the observer, the learner and when
making predictions. To train with supervised learning,
use method='Supervised' when creating the observer, then either 'T' and
'T_star' for each of the learners and when making predictions.

This module expects torch tensor for all attributes with shape greater than
one. All other numerical types are either integer or float.
Exceptions for different method inputs are documented in the method.
All tensors representing states of dynamical systems for simulation are
expected to have shape (number of time steps, number of different simulations,
dimension of the state).

Examples
--------
The following example constructs a Luenberger Observer for a non-linear
system, and plots the results of the estimation.

.. code-block:: Python

    import matplotlib.pyplot as plt
    from learn_KKL.luenberger_observer import LuenbergerObserver
    from learn_KKL.system import RevDuffing
    from learn_KKL.learner import Learner

    # Get reversed duffing as example system
    system = RevDuffing()
    data = system.generate_mesh([-2.,2.], 4000)

    # Initiate the Luenberger Observer
    observer = LuenbergerObserver(2, 1)
    observer.set_dynamics(system)

    # Train the model
    learner = Learner(observer,data)
    learner.train()

    tsim = (0, 50)
    dt = 1e-2
    tq, simulation = system.simulate(torch.tensor([[1.],[1.]]), tsim, dt)
    measurement = simulation[:, 0]

    # Predict from measurement
    estimation = observer.predict(measurement, tsim, dt)

    # Plot truth ground data and estimation
    plt.plot(tq, simulation)
    plt.plot(tq, estimation)

    plt.show()


Luenberger Observer library.
http://github.com/Centre-automatique-et-systemes/lena

This is licensed under an MIT license. See the readme.MD file
for more information.

Copyright 2021 Lukas Bahr.
"""

import numpy as np
import torch
from scipy import linalg
from scipy import signal
from torch import nn
from torchdiffeq import odeint
from torchinterp1d import Interp1d

from .utils import MSE, generate_mesh

# Set double precision by default
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)


class LuenbergerObserver(nn.Module):
    """ Implements a Luenberger observer. You are responsible for setting the
    state variables and functions to reasonable values; the defaults  will
    not give you a functional observer.

    The jupyter notebooks in the repository give you a
    basic idea of use, albeit without much description.

    In brief, you will first construct this object, specifying the size of
    the state vector with dim_x and the size of the measurement vector that
    you will be using with dim_y. The state vector dim_z for the Luenberger Observer
    will then automatically be set.

    After construction the observer will have default matrices D and F created for you,
    but you must specify the values for each. Matrix D, with shape dim_z X dim_z, is set by
    the eigenvalue roots of a third order bessel filter with the cutoff frequency of one.
    Matrix F is set to ones with dimension dim_z X 1. It’s usually easiest to just
    overwrite them rather than assign to each element yourself. This will be
    clearer in the example below. All are of type torch.tensor.


    Examples
    --------
    .. code-block:: Python

        from learn_KKL.luenberger_observer import LuenbergerObserver

        # Initiate the Luenberger Observer
        observer = LuenbergerObserver(2, 1)

        # Set observer dynamic functions
        observer.set_dynamics(system)
        # OR
        observer.f = f
        observer.g = g
        observer.h = h
        observer.u = u
        observer.u_1 = u_1

        # Set D matrix
        observer.D = bessel_D(w_c)
        # OR
        observer.D = torch.tensor()

        # Set F matrix
        observer.F = torch.tensor()

        # Predict from measurement
        estimation = observer.predict(measurement, tsim, dt)


    Params
    ----------
    dim_x : int
        Number of state variables of the dynamical system driving the Luenberger Observer.
        For example, if you are tracking the position and velocity of an object, dim_x would be 2.

    dim_y : int
        Number of of measurement inputs. For example, if the sensor provides you with position in y, dim_y would be 1.

    dim_z : int
        Number of state variables of Luenberger Observer given dim_x and dim_y.
        Default: dim_z = dim_y * (dim_x + 1)

    method : str
        Method for the observer object. Either "Autoencoder" or "Supervised".

    wc : float
        Cut-off frequency of the Bessel filter used to define D. By default,
        D is chosen such that it has the same poles as a Bessel filter of
        cut-off frequency 2 * np.pi * wc.



    Attributes
    ----------

    F : torch.tensor
        dim_z X 1 matrix F driving the dynamical system of the Luenberger observer.

    D : torch.tensor
        Real dim_z X dim_z matrix D representing the state matrix of a Luenberger observer.

    device : torch.device
        Default set to gpu if available. Used for training the model on either cpu or gpu.

    recon_lambda : float
        Default set to 1.01. Multiplicator on reconstruction loss.

    scaler_x :
        Scaler for x in the NN model.

    scaler_z :
        Scaler for z in the NN model.


    Methods
    ----------
    f(x: torch.tensor) : torch.tensor
        Dynamic function of the system.

    g(x: torch.tensor) : torch.tensor
        State vector of input function on dynamics.

    h(x: torch.tensor) : torch.tensor
        Gives the measurement of the system.

    u(x: torch.tensor) : torch.tensor
        Input on the system used for training the model.

    u_1(x: torch.tensor) : torch.tensor
        New input on the system which is not equal to the input used for the trained model.

    set_dynamics(system) : void
        Set the dynamic functions from a given system.

    bessel_D(wc: float = 1.) : toch.tensor
        Get D matrix from the eigenvalues of a third order
        bessel filter given the cutoff frequency.

    simulate(y: torch.tensor, tsim: tuple, dt: float) : [torch.tensor, torch.tensor]
        Simulate Luenberger Observer with dynamics and D, F matrices in time.

    create_layers(num_hl: int, size_hl: int, activation: torch.nn, dim_in: int, dim_out: int) : nn.ModuleList()
        Helper function that creates a list containing the network architecture.

    encoder(x: torch.tensor) -> torch.tensor:
        Compute latent space for a given input.

    decoder(z: torch.tensor) -> torch.tensor:
        Estimates x_hat vector given the simulated data from the Luenberger Observer.

    loss(x: torch.tensor, x_hat: torch.tensor, z_hat: torch.tensor) :
        [torch.tensor, torch.tensor, torch.tensor]
        Compute loss for the model optimizer.

    forward(x: torch.tensor) : [torch.tensor, torch.tensor]
        Pipeline for training the model.

    predict(self,  measurement: torch.tensor, tsim: tuple, dt: int) : torch.tensor:
        Predict x_hat for a given measurement of a state vector.


    References
    ----------

    .. [1] Pauline Bernard, "Observer Design for Nonlinear Systems"
       Springer, 2019

    .. [2] Louise da C. Ramos et al., "Numerical design of Luenberger observers for nonlinear systems"
       59th IEEE CDC, 2020

    """

    def __init__(self, dim_x: int, dim_y: int, method: str = "Autoencoder",
                 dim_z: int = None, wc: float = 1., num_hl: int = 5,
                 size_hl: int = 50, activation=nn.ReLU(),
                 recon_lambda: float = 1., D='bessel'):
        super(LuenbergerObserver, self).__init__()

        self.method = method

        self.dim_x = dim_x
        self.dim_y = dim_y

        if dim_z is None:
            self.dim_z = dim_y * (dim_x + 1)
        else:
            self.dim_z = dim_z

        if self.dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if self.dim_y < 1:
            raise ValueError('dim_y must be 1 or greater')

        # Set observer matrices D and F
        self.wc = wc
        self.F = torch.ones((self.dim_z, 1))
        if D == 'bessel':
            self.D = self.bessel_D(wc=self.wc)
        else:
            self.wc = 0.
            self.D = torch.as_tensor(D)

        # Model params
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.recon_lambda = recon_lambda

        # Define encoder and decoder architecture
        self.num_hl = num_hl
        self.size_hl = size_hl
        self.activation = activation
        self.encoder_layers = self.create_layers(
            self.num_hl, self.size_hl, self.activation, self.dim_x, self.dim_z)
        self.decoder_layers = self.create_layers(
            self.num_hl, self.size_hl, self.activation, self.dim_z, self.dim_x)
        self.scaler_x = None
        self.scaler_z = None

    def f(self, x: torch.tensor):
        return 0

    def h(self, x: torch.tensor):
        return 0

    def g(self, x: torch.tensor):
        return 0

    def u(self, x: torch.tensor):
        return 0

    def u_1(self, x: torch.tensor):
        return 0

    def __repr__(self):
        return '\n'.join([
            'Luenberger Observer object',
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

    def __call__(self, method='Autoencoder', *input):
        if method == "T":
            return self.forward_T(*input)
        elif method == "T_star":
            return self.forward_T_star(*input)
        elif method == "Autoencoder":
            return self.forward_autoencoder(*input)

    def set_dynamics(self, system):
        """
        Set the dynamic functions for the Luenberger Observer, from a
        given system object.

        Parameters
        ----------
        system : object
            System object, containing callable f, g, h, u, u_1
        """

        if not system.f or not system.g or not system.h or not system.u:
            print("One or more functions are missing")

        self.f = system.f
        self.g = system.g
        self.h = system.h
        self.u = system.u
        self.u_1 = system.u_1

    def set_scalers(self, scaler_x, scaler_z):
        """
        Set the scaler objects for input and output data. Then, the internal
        NN will normalize every input and denormalize every output.

        Parameters
        ----------
        scaler_X :
            Input scaler.

        scaler_out :
            Output scaler.
        """

        self.scaler_x = scaler_x
        self.scaler_z = scaler_z

    def bessel_D(self, wc: float = 1.) -> torch.tensor:
        """
        Returns a matrix from the eigenvalues of a third order
        bessel filter with a given cutoff frequency.

        Parameters
        ----------
        wc : float
            Multiplicator for the cutoff frequency of the bessel filter,
            for which Wn=2*pi*wc.
        """
        b, a = signal.bessel(N=self.dim_z, Wn=wc * 2 * np.pi, analog=True)
        whole_D = signal.place_poles(
            A=np.zeros((self.dim_z, self.dim_z)),
            B=-np.eye(self.dim_z),
            poles=np.roots(a))
        return torch.Tensor(whole_D.gain_matrix)

    def phi(self, z: torch.tensor) -> torch.tensor:
        """
        @Staticmethod.
        Returns matrix for the forward function driving the Luenberger Observer
        simulation. See reference for a detailed explanation.

        Parameters
        ----------
        z : torch.tensor
            State vector of the observer.

        Returns
        ----------
        phi: torch.tensor
            Computed matrix from the observer state vector.
        """
        # Compute jacobian for T^*(z)
        dTdy = torch.autograd.functional.jacobian(
            self.encoder, self.decoder(z.T), create_graph=False, strict=False,
            vectorize=True)  # TODO vectorize is experimental but faster!

        # Shape jacobian
        dTdx = torch.zeros((self.dim_z, self.dim_x))
        for j in range(dTdy.shape[1]):
            dTdx[j, :] = dTdy[0, j, 0, :]

        # System measurement g(T^*(z))
        msm = self.g(self.decoder(z.T).T)

        return torch.matmul(dTdx, msm)

    def simulate(self, y: torch.tensor, tsim: tuple, dt: float) -> torch.tensor:
        """
        Runs and outputs the results from Luenberger observer system.

        Parameters
        ----------
        y: torch.tensor
            Measurment expected to be in form (t, y).

        tsim: tuple
            Tuple of (Start, End) time of simulation.

        dt: float
            Step width of tsim.

        Returns
        ----------
        tq: torch.tensor
            Tensor of timesteps of tsim.

        sol: torch.tensor
            Solution of the simulation.
        """
        # Output timestemps of solver
        tq = torch.arange(tsim[0], tsim[1], dt)

        # 1D interpolation of y
        measurement = self.interpolate_func(y)

        # Zero initial value
        z_0 = torch.zeros((self.dim_z, 1))

        def dydt(t, z: torch.tensor):
            if self.u_1 == self.u:
                z_dot = torch.matmul(self.D, z) + self.F * measurement(t)
            else:
                z_dot = torch.matmul(self.D, z) + self.F * measurement(t) \
                        + torch.mul(self.phi(z), self.u_1(t) - self.u(t))
            return z_dot

        # Solve
        z = odeint(dydt, z_0, tq)

        return tq, z

    def simulate_system(self, y_0: torch.tensor, tsim: tuple,
                        dt, only_x: bool = False) -> torch.tensor:
        """
        Simulate Luenberger observer driven by a dynamical system.

        Parameters
        ----------
        y_0: torch.tensor
            Initial value for simulation.

        tsim: tuple
            Tuple of (Start, End) time of simulation.

        dt: float
            Step width of tsim.

        only_x: bool
            Whether to simulate both x and z or only x.

        Returns
        ----------
        tq: torch.tensor
            Tensor of timesteps of tsim.

        sol: torch.tensor
            Solution of the simulation.
        """

        def dydt(t, y):  # TODO only simulate x backward, z forward (interpol y)
            x = y[..., :self.dim_x]  # TODO change notation y
            z = y[..., self.dim_x:]
            x_dot = self.f(x) + self.g(x) * self.u(t)
            if only_x:
                z_dot = torch.zeros_like(z)
            else:
                z_dot = torch.matmul(z, self.D.t()) + torch.matmul(
                    self.h(x), self.F.t())
            return torch.cat((x_dot, z_dot), dim=-1)

        # Output timestemps of solver
        tq = torch.arange(tsim[0], tsim[1], dt)

        # Solve
        sol = odeint(dydt, y_0, tq)

        return tq, sol

    def generate_data_svl(self, limits: tuple, num_samples: int, k: int = 10,
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
        # limits_xz = np.array([[-1., 1.] * (self.dim_x + self.dim_z)])
        # print(limits_xz)
        # sampling = LHS(limits=limits_xz)
        # y_0 = torch.as_tensor(sampling(num_samples))
        # print(y_0.shape)
        y_1 = y_0.clone()

        # Simulate only x system backward in time
        tsim = (0, -self.t_c)
        y_0[:, :self.dim_x] = mesh
        tq_bw, data_bw = self.simulate_system(y_0, tsim, -dt, only_x=True)

        # Simulate both x and z forward in time starting from the last point
        # from previous simulation
        tsim = (-self.t_c, 0)
        y_1[:, :self.dim_x] = data_bw[-1, :, :self.dim_x]
        tq, data_fw = self.simulate_system(y_1, tsim, dt)

        # Data contains (x_i, z_i) pairs in shape [dim_x + dim_z,
        # number_simulations]
        data = data_fw[-1]
        return data

    @staticmethod
    def interpolate_func(x: torch.tensor) -> callable:
        """
        @Staticmethod.
        Takes a vector of times and values, returns a callable function which
        interpolates the given vector (along each output dimension independently).

        Author
        ----------
        Mona Buisson-Fenet

        Parameters
        ----------
        x: torch.tensor
            Vector of (t_i, x(t_i)) to interpolate.

        Returns
        ----------
        interpolate: callable[[List[float]], np.ndarray]
            Callable interoplation function.
        """
        # Don't save backpropagation
        with torch.no_grad():

            # Create list of interp1d functions
            points, values = x[:, 0].contiguous().view(-1, 1).t(), \
                             x[:, 1:].contiguous().view(-1, 1).t()
            interp_function = Interp1d()

            def interp(t, *args, **kwargs):
                if len(t.shape) == 0:
                    t = t.view(1, 1)
                else:
                    t = t.contiguous().view(-1, 1).t()
                if len(x) == 1:
                    # If only one value of x available, assume constant
                    interpolate_x = x[0, 1:].repeat(len(t[0]), 1)
                else:
                    interpolate_x = interp_function(points.expand(
                        values.shape[0], -1), values, t).t()
                return interpolate_x

        return interp

    def create_layers(self, num_hl: int, size_hl: int, activation: torch.nn,
                      dim_in: int, dim_out: int) -> nn.ModuleList():
        """
        Creates the NN model based on number of hidden layers, sizes etc.

        Parameters
        ----------
        num_hl: int
            Number of the hidden layers.

        size_hl: tuple
            Size of the hidden layers.

        activation: torch.nn
            Activation function for layers.
            Example: torch.nn.ReLu()

        dim_in: int
            Dimension of the input vector.

        dim_out: int
            Dimension of the output vector.

        Returns
        ----------
        module_list: tensor.nn.ModuleList()
            List of nn layers.
        """

        # Create ModuleList and add first layer with input dimension
        layers = nn.ModuleList()
        layers.append(nn.Linear(dim_in, size_hl))

        # Add layers of number hidden layers with size hidden layers
        # and activation
        for i in range(num_hl):
            layers.append(activation)
            layers.append(nn.Linear(size_hl, size_hl))

        # Append last layer with output dimension
        layers.append(nn.Linear(size_hl, dim_out))

        return layers

    def encoder(self, x: torch.tensor) -> torch.tensor:
        """
        Encodes input data and returns latent space data.

        Parameters
        ----------
        x: torch.tensor
            State vector of the driving system.

        Returns
        ----------
        z: torch.tensor
            Computation of observer state vector.
        """
        # Normalize input, iterate over encoder layers, denormalize output
        if self.scaler_x is not None:
            x = self.scaler_x.transform(x)
        for layer in self.encoder_layers:
            x = layer(x)
        if self.scaler_z is not None:
            x = self.scaler_z.inverse_transform(x)

        return x

    def decoder(self, z: torch.tensor) -> torch.tensor:
        """
        Estimates x_hat from a given obser state vector.

        Parameters
        ----------
        z: torch.tensor
            State vector of the observer.

        Returns
        ----------
        x_hat: torch.tensor
            Computation of state vector estimation driving the
            observer.
        """
        # Normalize input, iterate over decoder layers, denormalize output
        if self.scaler_z is not None:
            z = self.scaler_z.transform(z)
        for layer in self.decoder_layers:
            z = layer(z)
        if self.scaler_x is not None:
            z = self.scaler_x.inverse_transform(z)

        return z

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
        # Init mean squared error
        batch_size = x.shape[0]

        # Reconstruction loss MSE(x,x_hat)
        loss_1 = self.recon_lambda * MSE(x, x_hat, dim=dim)

        # Compute gradients of T_u with respect to inputs
        dTdh = torch.autograd.functional.jacobian(
            self.encoder, x, create_graph=False, strict=False, vectorize=False)
        # dTdx reshape (batch_size, self.dim_z, self.dim_x)
        dTdx = torch.transpose(torch.transpose(
            torch.diagonal(dTdh, dim1=0, dim2=2), 1, 2), 0, 1)
        # lhs = dTdx * f(x) of shape (batch_size, self.dim_z)
        lhs = torch.einsum('ijk,ik->ij', dTdx, self.f(x))

        D = self.D.to(self.device)
        F = self.F.to(self.device)
        rhs = torch.matmul(z_hat, D.t()) + torch.matmul(self.h(x), F.t())

        # PDE loss MSE(lhs, rhs)
        loss_2 = MSE(lhs, rhs, dim=dim)

        return loss_1 + loss_2, loss_1, loss_2

    def loss_T(self, z: torch.tensor, z_hat: torch.tensor, dim=None) -> \
            torch.tensor:
        """
        Loss function for training only the forward transformation T.

        Parameters
        ----------
        z: torch.tensor
            State vector of the observer.

        z_hat: torch.tensor
            Estimation of the state vector of the observer.

        dim: int
            Dimension along which to take the loss (if None, mean over all
            dimensions).

        Returns
        ----------
        loss: torch.tensor
            Reconstruction loss MSE(z, z_hat).
        """
        loss = MSE(z, z_hat, dim=dim)
        return loss

    def loss_T_star(self, x: torch.tensor, x_hat: torch.tensor, dim=None) -> \
            torch.tensor:
        """
        Loss function for training only the forward transformation T.

        Parameters
        ----------
        x: torch.tensor
            State vector of the system.

        x_hat: torch.tensor
            Estimation of the state vector of the system.

        dim: int
            Dimension along which to take the loss (if None, mean over all
            dimensions).

        Returns
        ----------
        loss: torch.tensor
            Reconstruction loss MSE(x, x_hat).
        """
        loss = MSE(x, x_hat, dim=dim)
        return loss

    def loss(self, method='Autoencoder', *input):
        if method == "T":
            return self.loss_T(*input)
        elif method == "T_star":
            return self.loss_T_star(*input)
        elif method == "Autoencoder":
            return self.loss_autoencoder(*input)

    def forward_autoencoder(self, x: torch.tensor) -> torch.tensor:
        """
        Forward function for autoencoder. Used for training the model.
        Computation follows as:
        z = encoder(x)
        x_hat = decoder(z)

        Parameters
        ----------
        x: torch.tensor
            State vector of the system driving the observer.

        Returns
        ----------
        z: torch.tensor
            State vector of the observer.

        x_hat: torch.tensor
            Estimation of the observer model.
        """
        # Enocde input space
        z = self.encoder(x)

        # Decode latent space
        x_hat = self.decoder(z)

        return z, x_hat

    def forward_T(self, x: torch.tensor) -> torch.tensor:
        """
        Forward function for T. Used for training the model.
        Computation follows as:
        z = encoder(x)

        Parameters
        ----------
        x: torch.tensor
            State vector of the system driving the observer.

        Returns
        ----------
        z: torch.tensor
            State vector of the observer.
        """
        # Enocde input space
        z = self.encoder(x)

        return z

    def forward_T_star(self, z: torch.tensor) -> torch.tensor:
        """
        Forward function for T_star. Used for training the model.
        Computation follows as:
        x_hat = decoder(z)

        Parameters
        ----------
        z: torch.tensor
            State vector of the the observer.

        Returns
        ----------
        x_hat: torch.tensor
            State vector of the observer.
        """
        # Decode latent space
        x_hat = self.decoder(z)

        return x_hat

    def predict(self, measurement: torch.tensor, tsim: tuple,
                dt: int) -> torch.tensor:
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
        tq, sol = self.simulate(measurement, tsim, dt)

        x_hat = self.decoder(sol[:, :, 0])

        return x_hat
