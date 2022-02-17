# -*- coding: utf-8 -*-

"""
This module implements a system class for which linear or non-linear
systems can be inherited. By inheritancing the system class
the dynamical systems recieve a box of tools for generating data
and setting the control input to different functions.

This module also contains two non-linear example systems. The Reversed Duffing
class implements the classic Duffing oscillator in a reversed form.
Van der Pol implements the system with default parameters.

All tensors representing states of dynamical systems for simulation are
expected to have shape (number of time steps, number of different simulations,
dimension of the state).


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

from math import pi

import numpy as np
import torch
from scipy import signal
from torchdiffeq import odeint

# Set double precision by default
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)


class System:
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

    def set_controller(self, controller) -> None:
        if controller == "null_controller":
            self.u = self.null_controller
        elif controller == "sin_controller":
            self.u = self.sin_controller
        elif controller == "lin_chirp_controller":
            self.u = self.lin_chirp_controller

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

        return tq, torch.squeeze(sol)

    def lin_chirp_controller(
        self, t: float, t_0: float = 0.0, a: float = 0.001, b: float = 9.99e-05
    ) -> torch.tensor:
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

    def sin_controller(
        self, t: float, t_0: float = 0.0, gamma: float = 0.4, omega: float = 1.2
    ) -> torch.tensor:
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

    def chirp_controller(
        self,
        t: float,
        t_0: float = 0,
        f_0: float = 6.0,
        f_1: float = 1.0,
        t_1: float = 10.0,
        gamma: float = 1.0,
    ) -> torch.tensor:
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
            u = signal.chirp(t, f0=f_0, f1=f_1, t1=t_1, method="linear")
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

    def __repr__(self):
        raise NotImplementedError


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
        xdot = torch.zeros_like(x)
        xdot[..., 0] = torch.pow(x[..., 1], 3)
        xdot[..., 1] = -x[..., 0]
        return xdot

    def h(self, x):
        return torch.unsqueeze(x[..., 0], dim=-1)

    def g(self, x):
        return torch.zeros_like(x)

    def __repr__(self):
        return "RevDuffing"


class VanDerPol(System):
    """ See https://en.wikipedia.org/wiki/Van_der_Pol_oscillator for detailed 
    reference for this system. 
    """

    def __init__(self, eps: float = 1.0):
        super().__init__()
        self.dim_x = 2
        self.dim_y = 1

        self.eps = eps

        self.u = self.null_controller
        self.u_1 = self.null_controller

    def f(self, x):
        xdot = torch.zeros_like(x)
        xdot[..., 0] = x[..., 1]
        xdot[..., 1] = self.eps * (1 - torch.pow(x[..., 0], 2)) * x[..., 1] - x[..., 0]
        return xdot

    def h(self, x):
        return torch.unsqueeze(x[..., 0], dim=-1)

    def g(self, x):
        xdot = torch.zeros_like(x)
        xdot[..., 1] = torch.ones_like(x[..., 1])
        return xdot

    def __repr__(self):
        return "VanDerPol"


class HO_unknown_freq(System):
    """
    harmonic oscillator with unknown frequency.
    extended state-space where x1 is the angle, x2 is the angular velocity,
    and x3 is the constant but unknown frequency.

    see example 8.1.1 in "observer design for nonlinear systems" by pauline
    bernard for more information.
    """

    def __init__(self,):
        super().__init__()
        self.dim_x = 3
        self.dim_y = 1

        self.u = self.null_controller
        self.u_1 = self.null_controller

    def f(self, x):
        xdot = torch.zeros_like(x)
        xdot[..., 0] = x[..., 1]
        xdot[..., 1] = -x[..., 2] * x[..., 0]
        return xdot

    def h(self, x):
        return torch.unsqueeze(x[..., 0], dim=-1)

    def g(self, x):
        xdot = torch.zeros_like(x)
        xdot[..., 1] = torch.ones_like(x[..., 1])
        return xdot

    def __repr__(self):
        return "HO_unknown_freq"


class QuanserQubeServo2(System):
    """ See https://www.quanser.com/products/qube-servo-2/ QUBE SERVO 2 and for a detailed 
    reference for this system 
    https://github.com/BlueRiverTech/quanser-openai-driver/blob/main/gym_brt/quanser/qube_simulator.py. 
    """

    def __init__(self):
        super().__init__()
        self.dim_x = 4
        self.dim_y = 1

        # Motor
        self.Rm = 8.4  # Resistance
        self.kt = 0.042  # Current-torque (N-m/A)
        self.km = 0.042  # Back-emf constant (V-s/rad)

        # Rotary Arm
        self.mr = 0.095  # Mass (kg)
        self.Lr = 0.085  # Total length (m)
        self.Jr = self.mr * self.Lr ** 2 / 12  # Moment of inertia about pivot (kg-m^2)
        self.Dr = 0.00027  # Equivalent viscous damping coefficient (N-m-s/rad)

        # Pendulum Link
        self.mp = 0.024  # Mass (kg)
        self.Lp = 0.129  # Total length (m)
        self.Jp = self.mp * self.Lp ** 2 / 12  # Moment of inertia about pivot (kg-m^2)
        self.Dp = 0.00005  # Equivalent viscous damping coefficient (N-m-s/rad)

        self.gravity = 9.81  # Gravity constant

        self.u = self.null_controller
        self.u_1 = self.null_controller

    def f(self, x, action=0):
        theta = x[..., 0]
        alpha = x[..., 1]
        theta_dot = x[..., 2]
        alpha_dot = x[..., 3]

        Vm = action
        tau = -(self.km * (Vm - self.km * theta_dot)) / self.Rm

        xdot = torch.zeros_like(x)
        xdot[..., 0] = theta_dot 
        xdot[..., 1] = alpha_dot
        xdot[..., 2] = (
            -self.Lp
            * self.Lr
            * self.mp
            * (
                -8.0 * self.Dp * alpha_dot
                + self.Lp ** 2 * self.mp * theta_dot ** 2 * np.sin(2.0 * alpha)
                + 4.0 * self.Lp * self.gravity * self.mp * np.sin(alpha)
            )
            * np.cos(alpha)
            + (4.0 * self.Jp + self.Lp ** 2 * self.mp)
            * (
                4.0 * self.Dr * theta_dot
                + self.Lp ** 2 * alpha_dot * self.mp * theta_dot * np.sin(2.0 * alpha)
                + 2.0 * self.Lp * self.Lr * alpha_dot ** 2 * self.mp * np.sin(alpha)
                - 4.0 * tau
            )
        ) / (
            4.0 * self.Lp ** 2 * self.Lr ** 2 * self.mp ** 2 * np.cos(alpha) ** 2
            - (4.0 * self.Jp + self.Lp ** 2 * self.mp)
            * (
                4.0 * self.Jr
                + self.Lp ** 2 * self.mp * np.sin(alpha) ** 2
                + 4.0 * self.Lr ** 2 * self.mp
            )
        )

        xdot[..., 3] = (
            2.0
            * self.Lp
            * self.Lr
            * self.mp
            * (
                4.0 * self.Dr * theta_dot
                + self.Lp ** 2 * alpha_dot * self.mp * theta_dot * np.sin(2.0 * alpha)
                + 2.0 * self.Lp * self.Lr * alpha_dot ** 2 * self.mp * np.sin(alpha)
                - 4.0 * tau
            )
            * np.cos(alpha)
            - 0.5
            * (
                4.0 * self.Jr
                + self.Lp ** 2 * self.mp * np.sin(alpha) ** 2
                + 4.0 * self.Lr ** 2 * self.mp
            )
            * (
                -8.0 * self.Dp * alpha_dot
                + self.Lp ** 2 * self.mp * theta_dot ** 2 * np.sin(2.0 * alpha)
                + 4.0 * self.Lp * self.gravity * self.mp * np.sin(alpha)
            )
        ) / (
            4.0 * self.Lp ** 2 * self.Lr ** 2 * self.mp ** 2 * np.cos(alpha) ** 2
            - (4.0 * self.Jp + self.Lp ** 2 * self.mp)
            * (
                4.0 * self.Jr
                + self.Lp ** 2 * self.mp * np.sin(alpha) ** 2
                + 4.0 * self.Lr ** 2 * self.mp
            )
        )

        return xdot

    def h(self, x):
        x_dot = torch.zeros((x.shape[0], self.dim_y))
        x_dot[..., 0] = x[...,1]
        # x_dot[..., 1] = x[...,1]
        return x_dot

    def g(self, x):
        xdot = torch.zeros_like(x)
        xdot[..., 1] = torch.ones_like(x[..., 1])
        return xdot

    def __repr__(self):
        return "QuanserQubeServo2"

class SaturatedVanDerPol(System):
    """ See https://en.wikipedia.org/wiki/Van_der_Pol_oscillator for detailed
    reference for this system.
    """

    def __init__(self, eps: float = 1.0):
        super().__init__()
        self.dim_x = 2
        self.dim_y = 1

        self.eps = eps

        self.u = self.null_controller
        self.u_1 = self.null_controller

    def f(self, x):
        xdot = torch.zeros_like(x)
        a = torch.max(torch.abs(x), dim=-1).values
        idx = torch.gt(a, 3)
        g = torch.ones_like(a)
        g[idx] = 1 - torch.exp(-.1 / (a[idx] - 3))
        xdot[..., 0] = x[..., 1]
        xdot[..., 1] = self.eps * (1 - torch.pow(x[..., 0], 2)) * x[..., 1] - x[..., 0]
        return xdot * torch.unsqueeze(g, dim=-1)

    def h(self, x):
        return torch.unsqueeze(x[..., 0], dim=-1)

    def g(self, x):
        xdot = torch.zeros_like(x)
        xdot[..., 1] = torch.ones_like(x[..., 1])
        return xdot

    def __repr__(self):
        return "SaturatedVanDerPol"