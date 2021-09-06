# -*- coding: utf-8 -*-

"""
This module implements the Luenberger Observer for linear and
non-linear system. The Lueneberger Observer class implements the
autoencoder model which needs to be trained on a given dynamical
sytem and has an easy to use interface to minimize the parameter
tuning on your side.

The general approach to use this model is as follows. Firstly,
the Lueneberger Observer must be initiated with the dimension of
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

Next, we need to train the autoencoder model. The learner class in this
package gives a simple interface to learn the model. However, you might
want to overwrite some learning parameters with your own.
With the trained model we are now able to estimate the state vectors
given the measurement y of our system.

This module expects torch tensor for all attributes with shape greater
one. All other numerical types are either integer or float.
Exceptions for different method inputs are documented in the method.


Examples
--------
The following example constructs a Lueneberger Observer for a non-linear
system, and plots the results of the estimation.

.. code-block:: Python

    import matplotlib.pyplot as plt
    from lena.lueneberger_observer import LuenebergerObserver
    from lena.system import RevDuffing
    from lena.learner import Learner

    # Get reversed duffing as example system
    system = RevDuffing()
    data = system.generate_mesh([-2.,2.], 4000)

    # Initiate the Lueneberger Observer
    observer = LuenebergerObserver(2, 1)
    observer.set_dynamics(system)

    # Train the model
    learner = Learner(observer,data)
    learner.train()

    tsim = (0, 50)
    dt = 1e-2
    tq, simulation = system.simulate(torch.tensor([[1.],[1.]]), tsim, dt)
    measurement = simulation[:, 0]

    # Predict from measurment
    estimation = observer.predict(measurement, tsim, dt)

    # Plot truth ground data and estimation
    plt.plot(tq, simulation)
    plt.plot(tq, estimation)

    plt.show()


Lueneberger Observer library.
http://github.com/Centre-automatique-et-systemes/lena

This is licensed under an MIT license. See the readme.MD file
for more information.

Copyright 2021 Lukas Bahr.
"""


from torchinterp1d import Interp1d
import torch
from torch import nn
from scipy import signal
from torchdiffeq import odeint
from scipy import linalg
import math
import numpy as np


class LuenebergerObserver(nn.Module):
    """ Implements a Lueneberger observer. You are responsible for setting the
    state variables and functions to reasonable values; the defaults  will
    not give you a functional observer.

    The jupyter notebooks in the repository give you a
    basic idea of use, albeit without much description.

    In brief, you will first construct this object, specifying the size of
    the state vector with dim_x and the size of the measurement vector that
    you will be using with dim_y. The state vector dim_z for the Lueneberger Observer
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

        from lena.lueneberger_observer import LuenebergerObserver

        # Initiate the Lueneberger Observer
        observer = LuenebergerObserver(2, 1)

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
        Number of state variables of the dynamical system driving the Lueneberger Observer.
        For example, if you are tracking the position and velocity of an object, dim_x would be 2.

    dim_y : int
        Number of of measurement inputs. For example, if the sensor provides you with position in y, dim_y would be 1.


    Attributes
    ----------
    dim_z : int
        Number of state variables of Lueneberger Observer given dim_x and dim_y.
        dim_z = dim_y * (dim_x + 1)

    F : torch.tensor
        dim_z X 1 matrix F driving the dynamical system of the Luenberger observer.

    D : torch.tensor
        Real dim_z X dim_z matrix D representing the state matrix of a Luenberger observer.

    device : torch.device
        Default set to gpu if available. Used for training the model on either cpu or gpu.

    recon_lambda : float
        Default set to 1.01. Multiplacator on reconstruction loss.


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
        Simulate Lueneberger Observer with dynamics and D, F matrices in time.

    create_layers(num_hl: int, size_hl: int, activation: torch.nn, dim_in: int, dim_out: int) : nn.ModuleList()
        Helper function that creates a list containing the network architecture.

    encoder(x: torch.tensor) -> torch.tensor:
        Compute latent space for a given input.

    decoder(z: torch.tensor) -> torch.tensor:
        Estimates x_hat vector given the simulated data from the Lueneberger Observer.

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

    def __init__(self, dim_x: int, dim_y: int):
        super(LuenebergerObserver, self).__init__()

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_y * (dim_x + 1)

        if self.dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if self.dim_y < 1:
            raise ValueError('dim_y must be 1 or greater')

        # Set observer matrices D and F
        self.F = torch.ones((self.dim_z, 1))
        self.D = self.bessel_D()

        # Model params
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.recon_lambda = 1.01

        # Define encoder and decoder architecture
        num_hl = 5
        size_hl = 40
        act = nn.ReLU()

        self.encoder_layers = self.create_layers(num_hl, size_hl, act, self.dim_x, self.dim_z)
        self.decoder_layers = self.create_layers(num_hl, size_hl, act, self.dim_z, self.dim_x)

    def f(self, x: torch.tensor): return 0
    def h(self, x: torch.tensor): return 0
    def g(self, x: torch.tensor): return 0
    def u(self, x: torch.tensor): return 0
    def u_1(self, x: torch.tensor): return 0

    def __repr__(self):
        return '\n'.join([
            'Lueneberger Observer object',
            'dim_x ' + str(self.dim_x),
            'dim_y ' + str(self.dim_y),
            'dim_z ' + str(self.dim_z),
            'D ' + str(self.D),
            'F ' + str(self.F),
            'encoder' + str(self.encoder_layers),
            'decoder' + str(self.decoder_layers),
        ])

    def set_dynamics(self, system):
        """
        Set the dynamic functions for the Lueneberger Observer, from a
        given system object.

        Parameters
        ----------
        system : object
            System object, containing callable f, g, h, u, u_1
        """

        if system.f or system.g or system.h or system.h:
            print("One or more functions are missing")

        self.f = system.f
        self.g = system.g
        self.h = system.h
        self.u = system.u
        self.u_1 = system.u_1

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
        b, a = signal.bessel(N=3, Wn=wc*2 * np.pi, analog=True)
        whole_D = signal.place_poles(
            A=np.zeros((self.dim_z, self.dim_z)),
            B=-np.eye(self.dim_z),
            poles=np.roots(a))
        return torch.Tensor(whole_D.gain_matrix)

    @staticmethod
    def phi(self, z: torch.tensor) -> torch.tensor:
        """
        @Staticmethod.
        Returns matrix for the forward function driving the Lueneberger Observer
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
        # TODO check jacobian
        dTdy = torch.autograd.functional.jacobian(
            self.encoder, self.decoder(z.T), create_graph=False, strict=False, vectorize=False)

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
        y_0: torch.tensor
            Initial value for observer simulation.

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
        def dydt(t, z: torch.tensor):
            if self.u_1 == self.u:
                z_dot = torch.matmul(self.D, z)+self.F*self.measurement(t)
            else:
                z_dot = torch.matmul(self.D, z)+self.F*self.measurement(t) + \
                    torch.mul(self.phi(z), self.u_new(t)-self.u_0(t))
            return z_dot

        # Output timestemps of solver
        tq = torch.arange(tsim[0], tsim[1], dt)

        # 1D interpolation of y
        self.measurement = self.interpolate_func(y)

        # Zero initial value
        z_0 = torch.zeros((self.dim_z, 1))

        # Solve
        z = odeint(dydt, z_0, tq)

        return tq, z

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
            points, values = x[:, 0].contiguous(), x[:, 1:].contiguous()
            interp_list = [Interp1d() for i in range(values.shape[1])]

            # Interpolation function
            def interp(t):
                if len(t.shape) == 0:
                    t = t.reshape(1, ).contiguous()
                else:
                    t = t.contiguous()
                if len(x) == 1:
                    # If only one value of x available, assume constant
                    interpolate_x = x[0, 1:].repeat(len(t), 1)
                else:
                    res = [interp_list[i](points, values[:, i], t) for i
                           in range(values.shape[1])]
                    interpolate_x = torch.squeeze(torch.stack(res), dim=1).t()
                return interpolate_x

        return interp

    def create_layers(self, num_hl: int, size_hl: int, activation: torch.nn, dim_in: int, dim_out: int) -> nn.ModuleList():
        """
        Runs and outputs the results from Luenberger observer system.

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
        # Iterate over encoder layers and compute next input
        for layer in self.encoder_layers:
            x = layer(x)

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
        # Iterate over decoder layers and compute next input
        for layer in self.decoder_layers:
            z = layer(z)

        return z

    def loss(
            self, x: torch.tensor, x_hat: torch.tensor, z_hat: torch.tensor) -> torch.tensor:
        """
        Loss function for training the observer model.
        See reference for detailed information.

        Parameters
        ----------
        x: torch.tensor
            State vector of the system driving the observer.

        x_hat: torch.tensor
            Estimation of the observer model.

        z_hat: torch.tensor
            Estimation of the state vector of the observer.

        Returns
        ----------
        loss: torch.tensor
            Reconstruction loss plus PDE loss.

        loss_1: torch.tensor
            Reconstruction loss MSE(x,x_hat).

        loss_2: torch.tensor
            PDE loss MSE(dTdx*f(x),D*z+F*h(x)).
        """
        # Init mean squared error
        batch_size = x.shape[0]

        mse = nn.MSELoss()

        # Reconstruction loss MSE(x,x_hat)
        loss_1 = self.recon_lambda * mse(x, x_hat)

        # Compute gradients of T_u with respect to inputs
        dTdx = torch.autograd.functional.jacobian(
            self.encoder, x, create_graph=False, strict=False, vectorize=False)
        dTdx = dTdx[dTdx != 0].reshape((batch_size, self.dim_z, self.dim_x))

        # lhs = dTdx * f(x)
        lhs = torch.zeros((self.dim_z, batch_size)).to(self.device)
        for i in range(batch_size):
            lhs[:, i] = torch.matmul(dTdx[i], self.f(x.T).T[i]).T

        # rhs = D * z + F * h(x)
        D = self.D.to(self.device)
        F = self.F.to(self.device)
        h_x = self.h(x.T).to(self.device)
        rhs = torch.matmul(D, z_hat.T) + torch.matmul(F, h_x)

        # PDE loss MSE(lhs, rhs)
        loss_2 = mse(lhs, rhs)

        return loss_1 + loss_2, loss_1, loss_2

    def forward(self, x: torch.tensor) -> torch.tensor:
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

    def predict(self,  measurement: torch.tensor, tsim: tuple, dt: int) -> torch.tensor:
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

        x_hat = self.decoder(sol)

        return x_hat
