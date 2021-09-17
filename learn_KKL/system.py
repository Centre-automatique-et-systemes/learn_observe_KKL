# -*- coding: utf-8 -*-

"""
This module implements a system class for which linear or non-linear
systems can be inherited. By inheritancing the system class
the dynamical systems recieve a box of tools for generating data
and setting the control input to different functions.

This module also contains two non-linear example systems. The Reversed Duffing
class implements the classic Duffing oscillator in a reversed form.
Van der Pol implements the system with default parameters.


Examples
--------
The following example initates the Reversed Duffing oscillator and plots the simulation
in time as well as the phase potrait.

.. code-block:: Python

    import matplotlib.pyplot as plt
    from learn_KKL.system import RevDuffing

    # Get reversed duffing as example system
    system = RevDuffing()

    # Simulation params
    x_0 = torch.tensor([[1.],[2.]])
    tsim = (0, 50)
    dt = 2e-1

    # Simulation with null input
    system.set_controller('null_controller')
    tq, sol_null = system.simulate(x_0, tsim, dt)

    # Plot phase potrait
    plt.plot(soll_null[:,0], sol_null[:,1])
    plt.show()

    # Simulation with chirp input
    system.set_controller('lin_chirp_controller')
    tq, sol_chirp = system.simulate(x_0, tsim, dt)

    # Plot phase potrait
    plt.plot(soll_chirp[:,0], sol_chirp[:,1])
    plt.show()

    # Plot truth ground data and estimation
    plt.plot(tq, sol_null)
    plt.plot(tq, sol_chirp)
    plt.show()


For more examples see the test subdirectory, or refer to the
book cited below. In it I both teach Kalman filtering from basic
principles, and teach the use of this library in great detail.

Luenberger Observer library.
http://github.com/Centre-automatique-et-systemes/lena

This is licensed under an MIT license. See the readme.MD file
for more information.

Copyright 2021 Lukas Bahr.
"""

import numpy as np
import torch
from torchdiffeq import odeint
from math import pi
from smt.sampling_methods import LHS
from scipy import signal

# Set double precision by default
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)


class System():
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

    Attributes
    ----------
    u : callable
        Default set to null controller.

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

    set_controller(controller: string) : void
        Set the controll input for a system.

    generate_mesh(limits: tuple, num_samples: int, method: string) : torch.tensor
        Returns 2D mesh for params either from LHS or uniform sampling.

    simulate(x_0: torch.tensor, tsim: tuple, dt: float) : [torch.tensor, torch.tensor]
        Simulate system in time.

    lin_chirp_controller(self, t: float, t_0: float = 0.0, a: float = 0.001, b: float = 9.99e-05) : torch.tensor
        Returns computed vector of a linear chirp function for time t.

    sin_controller(self, t: float, t_0: float = 0.0, gamma: float = 0.4, omega: float = 1.2) : torch.tensor
        Returns computed vector of a sinus function for time t.

    chirp_controller(self, t: float, t_0: float = 0, f_0: float = 6.0, f_1: 
        float = 1.0, t_1: float = 10.0, gamma: float = 1.0) : torch.tensor
        Returns computed vector of a chirp function for time t.

    null_controller(t: float) : torch.tensor
        Returns zero vector for time t.
    """

    def __init__(self):
        self.u = self.null_controller

    def f(self, x: torch.tensor): return 0
    def h(self, x: torch.tensor): return 0
    def g(self, x: torch.tensor): return 0
    def u(self, x: torch.tensor): return 0
    def u_1(self, x: torch.tensor): return 0

    def set_controller(self, controller) -> None:
        if controller == 'null_controller':
            self.u = self.null_controller
        elif controller == 'sin_controller':
            self.u = self.sin_controller
        elif controller == 'lin_chirp_controller':
            self.u = self.lin_chirp_controller

    def generate_mesh(self, limits: np.array, num_samples: int, method: str = 'lhs') -> torch.tensor:
        """
        Generates 2D mesh either from a uniform distribution or uses latin hypercube
        sampling.

        Parameters
        ----------
        limits: np.array
            Array for the limits of all axes, used for sampling.
            Form np.array([[min_1, max_1], ..., [min_n, max_n]]).

        num_sample: int
            Number of samples in this space.

        method: str
            Use 'lhs' or 'uniform'.

        Returns
        ----------
        mesh: torch.tensor
            Mesh in shape (num_samples, 2).
        """

        # Sample either a uniformly grid or use latin hypercube sampling
        if method == 'uniform':
            axes = np.linspace(limits[:, 0], limits[:, 1], num_samples)
            mesh = \
                np.array(np.meshgrid(axes, axes)).T.reshape(-1, axes.shape[1])
        elif method == 'lhs':
            sampling = LHS(xlimits=limits)
            mesh = sampling(num_samples)

        return torch.as_tensor(mesh)

    def simulate(self, x_0: torch.tensor, tsim: tuple, dt) -> torch.tensor:
        """
        Simulates the system for given params.

        Parameters
        ----------
        x_0: torch.tensor
            Initial value for simulation.

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
        def dxdt(t, x):
            x_dot = self.f(x) + self.g(x) * self.u(t)
            return x_dot

        # Output timestemps of solver
        tq = torch.arange(tsim[0], tsim[1], dt)

        # Solve
        sol = odeint(dxdt, x_0, tq)

        return tq, sol

    def lin_chirp_controller(self, t: float, t_0: float = 0.0, a: float = 0.001, b: float = 9.99e-05) -> torch.tensor:
        """
        Linear chirp controller function.

        Parameters
        ----------
        t: float
            Point for function computation.

        t_0: float
            Initial time value for control input.

        a: float 
            See https://en.wikipedia.org/wiki/Chirp.

        b: float 
            See https://en.wikipedia.org/wiki/Chirp.

        Returns
        ----------
        u: torch.tensor
            Linear Chirp output.
        """
        if t <= t_0:
            u = 0.0
        else:
            u = torch.sin(2 * pi * t * (a + b * t))
        return u

    def sin_controller(self, t: float, t_0: float = 0.0, gamma: float = 0.4, omega: float = 1.2) -> torch.tensor:
        """
        Harmonic oscillator controller function.

        Parameters
        ----------
        t: float
            Point for function computation.

        t_0: float
            Initial time value for control input.

        gamma: float 
            See https://en.wikipedia.org/wiki/Harmonic_oscillator.

        omega: float 
            See https://en.wikipedia.org/wiki/Harmonic_oscillator.

        Returns
        ----------
        u: torch.tensor
            Linear Chirp output.
        """
        if t <= t_0:
            u = 0.0
        else:
            u = gamma * torch.cos(omega * t)
        return u

    def chirp_controller(self, t: float, t_0: float = 0, f_0: float = 6.0, f_1: float = 1.0, t_1: float = 10.0, gamma: float = 1.0) -> torch.tensor:
        """
        Linear chirp controller function.

        Parameters
        ----------
        t: float
            Point for function computation.

        t_0: float
            Initial time value for control input.

        a: float 
            See https://en.wikipedia.org/wiki/Chirp.

        b: float 
            See https://en.wikipedia.org/wiki/Chirp.

        Returns
        ----------
        u: torch.tensor
            Chirp output.
        """
        t = t.numpy()
        nb_cycles = int(np.floor(np.min(t) / t_1))
        t = t - nb_cycles * t_1
        if t <= t_0:
            u = 0.0
        else:
            u = signal.chirp(t, f0=f_0, f1=f_1, t1=t_1, method='linear')
        return torch.tensor(gamma * u)

    def null_controller(self, t: float) -> torch.tensor:
        """
        Null controller function.

        Parameters
        ----------
        t: float
            Point for function computation.

        Returns
        ----------
        u: torch.tensor
            Zero out.
        """
        return torch.tensor(0.0)


class RevDuffing(System):
    """ See https://en.wikipedia.org/wiki/Duffing_equation for detailed 
    reference for this system. 
    """

    def __init__(self):
        super().__init__()
        self.dim_x = 2
        self.dim_y = 1

        self.u = self.null_controller
        self.u_1 = self.null_controller

    def f(self, x):
        x_0 = torch.reshape(torch.pow(x[1, :], 3), (1, -1))
        x_1 = torch.reshape(-x[0, :], (1, -1))
        return torch.cat((x_0, x_1), 0)

    def h(self, x):
        # return torch.reshape(x[0, :], (1, -1))
        return torch.unsqueeze(x[0, :], dim=0)

    def g(self, x):
        return torch.zeros(x.shape[0], x.shape[1])


class VanDerPol(System):
    """ See https://en.wikipedia.org/wiki/Van_der_Pol_oscillator for detailed 
    reference for this system. 
    """

    def __init__(self):
        super().__init__()
        self.dim_x = 2
        self.dim_y = 1

        self.eps = 1

        self.u = self.null_controller
        self.u_1 = self.null_controller

    def f(self, x):
        x_0 = torch.reshape(x[1, :], (1, -1))
        x_1 = torch.reshape(self.eps*(1-torch.pow(x[0, :], 2))*x[1, :]-x[0, :], (1, -1))
        return torch.cat((x_0, x_1))

    def h(self, x):
        return torch.reshape(x[0, :], (1, -1))

    def g(self, x):
        zeros = torch.reshape(torch.zeros_like(x[1, :]), (1, -1))
        ones = torch.reshape(torch.ones_like(x[0, :]), (1, -1))
        return torch.cat((zeros, ones))