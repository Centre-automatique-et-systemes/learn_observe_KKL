# -*- coding: utf-8 -*-

"""
This module implements a system class for which linear or non-linear
systems can be inherited. By inheriting the system class
the dynamical systems receive a box of tools for generating data
and setting the control input to different functions.

This module also contains two non-linear example systems. The Reversed Duffing
class implements the classic Duffing oscillator in a reversed form.
Van der Pol implements the system with default parameters.

All tensors representing states of dynamical systems for simulation are
expected to have shape (number of time steps, number of different simulations,
dimension of the state).


Examples
--------
The following example imitates the Reversed Duffing oscillator and plots the
simulation in time as well as the phase portrait.

.. code-block:: Python

    import matplotlib.pyplot as plt
    from learn_KKL.system import RevDuffing
    import torch

    # Get reversed duffing as example system
    system = RevDuffing()

    # Simulation params
    x_0 = torch.transpose(torch.tensor([[1.],[2.]]),0,1)
    tsim = (0, 50)
    dt = 2e-1

    # Simulation with null input
    system.set_controller('null_controller')
    tq, sol_null = system.simulate(x_0, tsim, dt)

    # Plot phase potrait
    plt.plot(sol_null[:,0], sol_null[:,1])
    plt.show()

    # Simulation with chirp input
    system.set_controller('lin_chirp_controller')
    tq, sol_chirp = system.simulate(x_0, tsim, dt)

    # Plot phase potrait
    plt.plot(sol_chirp[:,0], sol_chirp[:,1])
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

import copy
import numpy as np
import torch
from scipy import signal
from torchdiffeq import odeint
from functorch import vmap, jacrev, hessian

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
        self.needs_remap = False

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

    def remap(self, x, wc=False):
        raise NotImplementedError

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

        if self.needs_remap:
            return tq, self.remap(torch.squeeze(sol))
        else:
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
        return "Reversed_Duffing_Oscillator"


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
    """ See https://www.quanser.com/products/qube-servo-2/ QUBE SERVO 2 and
    for a detailed reference for this system.
    Documentation on the simulator:
    https://github.com/BlueRiverTech/quanser-openai-driver/blob/main/gym_brt/quanser/qube_simulator.py
    https://github.com/BlueRiverTech/quanser-openai-driver/blob/main/tests/notebooks/Compare%20Qube%20Hardware%20to%20ODEint.ipynb

    State: (theta, alpha, theta_dot, alpha_dot)
    Measurement: (theta, alpha)
    """

    def __init__(self, r: float = 50., d: float = 100.):
        super().__init__()
        self.dim_x = 4
        self.dim_y = 2
        self.needs_remap = True
        # after each simulation, remap trajectory to [-pi,pi] and [0,
        # 2pi]: belongs to systems that need to remap simulated trajectories

        # Saturation to avoid simulation exploding in backward time
        self.r = r  # for saturation = 1 if norm(x) <= r
        self.d = d  # for saturation = 0 if norm(x) >= r+d
        self.coef = self.set_coef()  # saturation = polynomial(norm(x) - r)

        # Motor
        # self.Rm = 8.4  # Resistance
        self.kt = 0.042  # Current-torque (N-m/A)
        # self.km = 0.042  # Back-emf constant (V-s/rad)

        # Rotary Arm
        self.mr = 0.095  # Mass (kg)
        self.Lr = 0.085  # Total length (m)
        self.Jr = self.mr * self.Lr ** 2 / 12  # Moment of inertia about pivot (kg-m^2)
        # self.Dr = 0.00027  # Equivalent viscous damping coefficient (N-m-s/rad)

        # Pendulum Link
        self.mp = 0.024  # Mass (kg)
        self.Lp = 0.129  # Total length (m)
        self.Jp = self.mp * self.Lp ** 2 / 12  # Moment of inertia about pivot (kg-m^2)
        # self.Dp = 0.00005  # Equivalent viscous damping coefficient (N-m-s/rad)

        # After identification on hardware data:
        self.Rm = 14
        self.km = 0.01
        self.Dr = 0.0005
        self.Dp = -3e-5

        self.gravity = 9.81  # Gravity constant

        self.u = self.null_controller
        self.u_1 = self.null_controller

    def p(self, x):
        # Saturation = polynomial(x) that goes from 1 to 0 over [0, d] with
        # deriv = 0 at both ends to ensure Lipschitz: 4 conditions, 4 unknowns
        return self.coef[0] * x ** 3 + self.coef[1] * x ** 2 + self.coef[
            2] * x + self.coef[3]

    def set_coef(self):
        A = torch.tensor(
            [[0, 0, 0, 1.],
             [0, 0, 1., 0],
             [self.d ** 3, self.d ** 2, self.d, 1],
             [3 * self.d ** 2, 2 * self.d, 1, 0]])
        B = torch.tensor([[1.], [0], [0], [0]])
        return torch.linalg.solve(A, B)

    def f(self, x, action=0.):
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
                + self.Lp ** 2 * self.mp * theta_dot ** 2 * torch.sin(2.0 * alpha)
                + 4.0 * self.Lp * self.gravity * self.mp * torch.sin(alpha)
            )
            * torch.cos(alpha)
            + (4.0 * self.Jp + self.Lp ** 2 * self.mp)
            * (
                4.0 * self.Dr * theta_dot
                + self.Lp ** 2 * alpha_dot * self.mp * theta_dot * torch.sin(2.0 * alpha)
                + 2.0 * self.Lp * self.Lr * alpha_dot ** 2 * self.mp * torch.sin(alpha)
                - 4.0 * tau
            )
        ) / (
            4.0 * self.Lp ** 2 * self.Lr ** 2 * self.mp ** 2 * torch.cos(alpha) ** 2
            - (4.0 * self.Jp + self.Lp ** 2 * self.mp)
            * (
                4.0 * self.Jr
                + self.Lp ** 2 * self.mp * torch.sin(alpha) ** 2
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
                + self.Lp ** 2 * alpha_dot * self.mp * theta_dot * torch.sin(2.0 * alpha)
                + 2.0 * self.Lp * self.Lr * alpha_dot ** 2 * self.mp * torch.sin(alpha)
                - 4.0 * tau
            )
            * torch.cos(alpha)
            - 0.5
            * (
                4.0 * self.Jr
                + self.Lp ** 2 * self.mp * torch.sin(alpha) ** 2
                + 4.0 * self.Lr ** 2 * self.mp
            )
            * (
                -8.0 * self.Dp * alpha_dot
                + self.Lp ** 2 * self.mp * theta_dot ** 2 * torch.sin(2.0 * alpha)
                + 4.0 * self.Lp * self.gravity * self.mp * torch.sin(alpha)
            )
        ) / (
            4.0 * self.Lp ** 2 * self.Lr ** 2 * self.mp ** 2 * torch.cos(alpha) ** 2
            - (4.0 * self.Jp + self.Lp ** 2 * self.mp)
            * (
                4.0 * self.Jr
                + self.Lp ** 2 * self.mp * torch.sin(alpha) ** 2
                + 4.0 * self.Lr ** 2 * self.mp
            )
        )

        # Saturation
        xnorm = torch.linalg.norm(x, 2, dim=-1)
        sat = torch.where(xnorm <= self.r, 1.,
                          torch.where(xnorm < self.r + self.d,
                                      self.p(xnorm - self.r), 0.))
        # xbig = torch.argwhere(xnorm > self.r)
        # if torch.numel(xbig) > 0:
        #     print(torch.numel(xbig))

        return xdot * torch.unsqueeze(sat, dim=-1)

    def h(self, x):
        return x[..., :2]

    def g(self, x):
        xdot = torch.zeros_like(x)
        xdot[..., 1] = torch.ones_like(x[..., 1])
        return xdot

    # For flexibility and coherence: use remap function after every simulation
    # But be prepared to change its behavior!
    def remap(self, traj, wc=False):
        # Map theta to [-pi,pi] and alpha to [0, 2pi]
        if not wc:
            traj[..., 0] = ((traj[..., 0] + np.pi) % (2 * np.pi)) - np.pi
            traj[..., 1] = traj[..., 1] % (2 * np.pi)
        else:
            traj[..., 0, :] = ((traj[..., 0, :] + np.pi) % (2 * np.pi)) - np.pi
            traj[..., 1, :] = traj[..., 1, :] % (2 * np.pi)
        return traj

    # For adapting hardware data to the conventions of the simulation model
    def remap_hardware(self, traj, add_pi_alpha=False, wc=False):
        # Reorder as (theta, alpha, thetadot, alphadot)
        # Convention for alpha: 0 is upwards (depends on dataset!)
        # Remap as simulation data
        traj_copy = copy.deepcopy(traj)
        if not wc:
            traj[..., 0], traj[..., 1] = traj_copy[..., 1], traj_copy[..., 0]
            traj[..., 2], traj[..., 3] = traj_copy[..., 3], traj_copy[..., 2]
            if add_pi_alpha:
                traj[..., 1] += np.pi
        else:
            traj[..., 0, :], traj[..., 1, :] = \
                traj_copy[..., 1, :], traj_copy[..., 0, :]
            traj[..., 2, :], traj[..., 3, :] = \
                traj_copy[..., 3, :], traj_copy[..., 2, :]
            if add_pi_alpha:
                traj[..., 1, :] += np.pi
        return self.remap(traj, wc=wc)

    def __repr__(self):
        return "QuanserQubeServo2_meas12"

    # Useful functions for EKF
    def __call__(self, t, x, u, t0, init_control, process_noise_var, kwargs,
                 impose_init_control=False):
        return self.f(x)

    def call_deriv(self, t, x, u, t0, init_control, process_noise_var,
                   kwargs, impose_init_control=False):
        return self.predict_deriv(x, self.f)

    # Jacobian of function that predicts xt+dt (or other) from xt and the
    # function u. Useful for EKF!
    def predict_deriv(self, x, f):
        # Compute Jacobian of f with respect to input x
        dfdh = vmap(jacrev(f))(x)
        dfdx = torch.squeeze(dfdh)
        return dfdx

    def predict_double_deriv(self, x, f):
        # Compute Jacobian of f with respect to input x
        dfdh = vmap(hessian(f))(x)
        dfdx = torch.squeeze(dfdh)
        return dfdx


class QuanserQubeServo2_meas2(QuanserQubeServo2):
    """
    Same as QuanserQubeServo2 except we measure only alpha.
    """
    def __init__(self):
        super().__init__()
        self.dim_y = 1

    def h(self, x):
        return torch.unsqueeze(x[..., 1], dim=-1)

    def __repr__(self):
        return "QuanserQubeServo2_meas2"


class QuanserQubeServo2_meas1(QuanserQubeServo2):
    """
    Same as QuanserQubeServo2 except we measure only theta.
    """
    def __init__(self):
        super().__init__()
        self.dim_y = 1

    def h(self, x):
        return torch.unsqueeze(x[..., 0], dim=-1)

    def __repr__(self):
        return "QuanserQubeServo2_meas1"



class OldSaturatedVanDerPol(System):
    """ See https://en.wikipedia.org/wiki/Van_der_Pol_oscillator for detailed
    reference for this system.
    """

    def __init__(self, eps: float = 1.0, limit: float = 3.):
        super().__init__()
        self.dim_x = 2
        self.dim_y = 1

        self.eps = eps
        self.limit = limit  # for saturation

        self.u = self.null_controller
        self.u_1 = self.null_controller

    def f(self, x):
        xdot = torch.zeros_like(x)
        a = torch.max(torch.abs(x), dim=-1).values
        idx = torch.gt(a, self.limit)
        g = torch.ones_like(a)
        g[idx] = 1 - torch.exp(-.1 / (a[idx] - self.limit))
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
        return "OldSaturatedVanDerPol"

class SaturatedVanDerPol(System):
    """ See https://en.wikipedia.org/wiki/Van_der_Pol_oscillator for detailed
    reference for this system.
    """

    def __init__(self, eps: float = 1.0, r: float = 3., d: float = 7.):
        super().__init__()
        self.dim_x = 2
        self.dim_y = 1

        self.eps = eps
        self.r = r  # for saturation = 1 if norm(x) <= r
        self.d = d  # for saturation = 0 if norm(x) >= r+d
        self.coef = self.set_coef()  # saturation = polynomial(norm(x) - r)

        self.u = self.null_controller
        self.u_1 = self.null_controller

    def p(self, x):
        # Saturation = polynomial(x) that goes from 1 to 0 over [0, d] with
        # deriv = 0 at both ends to ensure Lipschitz: 4 conditions, 4 unknowns
        return self.coef[0] * x ** 3 + self.coef[1] * x ** 2 + self.coef[
            2] * x + self.coef[3]

    def set_coef(self):
        A = torch.tensor(
            [[0, 0, 0, 1.],
             [0, 0, 1., 0],
             [self.d ** 3, self.d ** 2, self.d, 1],
             [3 * self.d ** 2, 2 * self.d, 1, 0]])
        B = torch.tensor([[1.], [0], [0], [0]])
        return torch.linalg.solve(A, B)

    def f(self, x):
        xdot = torch.zeros_like(x)
        xnorm = torch.linalg.norm(x, 2, dim=-1)
        g = torch.where(xnorm <= self.r, 1.,
                        torch.where(xnorm < self.r + self.d,
                                    self.p(xnorm - self.r), 0.))
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