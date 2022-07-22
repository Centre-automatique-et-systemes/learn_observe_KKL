import torch
import numpy as np
from scipy.interpolate import interp1d
from torchinterp1d import Interp1d
from typing import Callable
from torchdiffeq import odeint

# Useful functions for filtering: EKF...
# Mostly code from other repos added here for transfers

# Reshape any vector of (length,) to (length, 1) (possibly several points but
# of dimension 1)
def reshape_dim1(x):
    if torch.is_tensor(x):
        if len(x.shape) == 0:
            x = x.view(1, 1)
        elif len(x.shape) == 1:
            x = x.view(x.shape[0], 1)
    else:
        if np.isscalar(x) or np.array(x).ndim == 0:
            x = np.reshape(x, (1, 1))
        else:
            x = np.array(x)
        if len(x.shape) == 1:
            x = np.reshape(x, (x.shape[0], 1))
    return x


# Same as reshape_dim1 but for difftraj when the first 2 dimensions stay
def reshape_dim1_difftraj(x):
    if torch.is_tensor(x):
        if len(x.shape) == 0:
            x = x.veiw(1, 1, 1)
        elif len(x.shape) == 1:
            x = x.view(1, x.shape[0], 1)
        elif len(x.shape) == 2:
            x = x.view(x.shape[0], x.shape[1], 1)
    else:
        if np.isscalar(x) or np.array(x).ndim == 0:
            x = np.reshape(x, (1, 1, 1))
        else:
            x = np.array(x)
        if len(x.shape) == 1:
            x = np.reshape(x, (1, x.shape[0], 1))
        elif len(x.shape) == 2:
            x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    return x


# Reshape any vector of (length,) to (1, length) (single point of certain
# dimension)
def reshape_pt1(x):
    if torch.is_tensor(x):
        if len(x.shape) == 0:
            x = x.view(1, 1)
        elif len(x.shape) == 1:
            x = x.view(1, x.shape[0])
    else:
        if np.isscalar(x) or np.array(x).ndim == 0:
            x = np.reshape(x, (1, 1))
        else:
            x = np.array(x)
        if len(x.shape) == 1:
            x = np.reshape(x, (1, x.shape[0]))
    return x


# Same as reshape_pt1 but for difftraj when the first and last dimensions stay
def reshape_pt1_difftraj(x):
    if torch.is_tensor(x):
        if len(x.shape) == 0:
            x = x.view(1, 1, 1)
        elif len(x.shape) == 1:
            x = x.view(1, 1, x.shape[0])
        elif len(x.shape) == 2:
            x = x.view(x.shape[0], 1, x.shape[1])
    else:
        if np.isscalar(x) or np.array(x).ndim == 0:
            x = np.reshape(x, (1, 1, 1))
        else:
            x = np.array(x)
        if len(x.shape) == 1:
            x = np.reshape(x, (1, 1, x.shape[0]))
        elif len(x.shape) == 2:
            x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
    return x


# Reshape any point of type (1, length) to (length,)
def reshape_pt1_tonormal(x):
    if torch.is_tensor(x):
        if len(x.shape) == 0:
            x = x.view(1, )
        elif len(x.shape) == 1:
            x = x.view(x.shape[0], )
        elif x.shape[0] == 1:
            x = torch.squeeze(x, 0)
    else:
        if np.isscalar(x) or np.array(x).ndim == 0:
            x = np.reshape(x, (1,))
        elif len(x.shape) == 1:
            x = np.reshape(x, (x.shape[0],))
        elif x.shape[0] == 1:
            x = np.reshape(x, (x.shape[1],))
    return x


# Reshape any vector of type (length, 1) to (length,)
def reshape_dim1_tonormal(x):
    if torch.is_tensor(x):
        if len(x.shape) == 0:
            x = x.view(1, )
        elif len(x.shape) == 1:
            x = x.view(x.shape[0], )
        elif x.shape[1] == 1:
            x = x.view(x.shape[0], )
    else:
        if np.isscalar(x) or np.array(x).ndim == 0:
            x = np.reshape(x, (1,))
        elif len(x.shape) == 1:
            x = np.reshape(x, (x.shape[0],))
        elif x.shape[1] == 1:
            x = np.reshape(x, (x.shape[0],))
    return x

# Vector x = (t_i, x(t_i)) of time steps t_i at which x is known is
# interpolated at given time t, interpolating along each output dimension
# independently if there are more than one. Returns a function interp(t,
# any other args) which interpolates x at times t
def interpolate_func(x, t0, init_value, method='linear', impose_init=False) -> \
        Callable:
    """
    Takes a vector of times and values, returns a callable function which
    interpolates the given vector (along each output dimension independently).

    :param x: vector of (t_i, x(t_i)) to interpolate
    :type x: torch.tensor
    param t0: initial time at which to impose initial condition
    :type t0: torch.tensor
    param init_value: initial condition to impose
    :type init_value: torch.tensor
    param impose_init: whether to impose initial condition
    :type impose_init: bool

    :returns: function interp(t, other args) which interpolates x at t
    :rtype:  Callable[[List[float]], np.ndarray]
    """
    x = reshape_pt1(x)
    if torch.is_tensor(x):  # not building computational graph!
        with torch.no_grad():
            if method != 'linear':
                raise NotImplementedError(
                    'Only linear interpolator available in pytorch!')
            points, values = reshape_dim1(x[:, 0].contiguous()).t(), \
                             reshape_dim1(x[:, 1:].contiguous()).t()
            interp_function = Interp1d()

            def interp(t, *args, **kwargs):
                if len(t.shape) == 0:
                    t = t.view(1, 1)
                else:
                    t = reshape_dim1(t.contiguous()).t()
                if len(x) == 1:
                    # If only one value of x available, assume constant
                    interpolate_x = x[0, 1:].repeat(len(t[0]), 1)
                else:
                    interpolate_x = interp_function(points.expand(
                        values.shape[0], -1), values, t).t()
                if t[0, 0] == t0 and impose_init:
                    # Impose initial value
                    interpolate_x[0] = reshape_pt1(init_value)
                return interpolate_x

    else:
        points, values = x[:, 0], x[:, 1:].T
        interp_function = interp1d(x=points, y=values, kind=method,
                                   fill_value="extrapolate")

        def interp(t, *args, **kwargs):
            if np.isscalar(t):
                t = np.array([t])
            else:
                t = reshape_dim1_tonormal(t)
            if len(x) == 1:
                # If only one value of x available, assume constant
                interpolate_x = np.tile(reshape_pt1(x[0, 1:]), (len(t), 1))
            else:
                interpolate_x = interp_function(t).T
            if t[0] == t0 and impose_init:
                # Impose initial value
                interpolate_x[0] = reshape_pt1(init_value)
            return interpolate_x

    return interp

# Input x, u, version and parameters, output x over t_eval with torchdiffeq ODE
# solver (or manually if discrete)
def dynamics_traj_observer(x0, u, y, t0, dt, init_control, discrete=False,
                           version=None, method='dopri5', t_eval=[0.1],
                           GP=None, stay_GPU=False, lightning=False,
                           impose_init_control=False, **kwargs):
    # Go to GPU at the beginning of simulation
    if torch.cuda.is_available() and not lightning:
        x0 = x0.cuda()
    device = x0.device
    if kwargs['kwargs'].get('solver_options'):
        solver_options = kwargs['kwargs'].get('solver_options').copy()
        rtol = solver_options['rtol']
        atol = solver_options['atol']
        solver_options.pop('rtol')
        solver_options.pop('atol')
    else:
        solver_options = {}
        rtol = 1e-3
        atol = 1e-6
    x0 = reshape_pt1(x0)
    if not torch.is_tensor(t_eval):
        t_eval = torch.tensor(t_eval, device=device)
    if torch.cuda.is_available() and not lightning:
        t_eval = t_eval.cuda()
    if discrete:
        if torch.is_tensor(t0):
            t = torch.clone(t0)
        else:
            t = torch.tensor([t0], device=device)
        if len(t_eval) == 1:
            # Solve until reach final time in t_eval
            x = reshape_pt1(x0).clone()
            while t < t_eval[-1]:
                xnext = reshape_pt1(
                    version(t, x, u, y, t0, init_control, GP,
                            impose_init_control=impose_init_control, **kwargs))
                x = xnext
                t += dt
            xtraj = reshape_pt1(x)
        else:
            # Solve one time step at a time until end or length of t_eval
            # xtraj = torch.empty((len(t_eval), x0.shape[1]), device=device)
            xtraj = torch.empty(tuple([len(t_eval)] + list(x0.shape[1:])),
                                device=device)
            xtraj[0] = reshape_pt1(x0)
            i = 0
            while (i < len(t_eval) - 1) and (t < t_eval[-1]):
                i += 1
                xnext = reshape_pt1(
                    version(t, xtraj[i - 1], u, y, t0, init_control, GP,
                            impose_init_control=impose_init_control, **kwargs))
                xtraj[i] = xnext
                t += dt
    else:
        def f(tl, xl):
            return version(tl, xl, u, y, t0, init_control, GP,
                           impose_init_control=impose_init_control, **kwargs)

        if len(t_eval) == 1:
            # t0 always needed for odeint, then deleted
            t_eval = torch.cat((torch.tensor([t0], device=device), t_eval))
            xtraj = odeint(f, reshape_pt1_tonormal(x0), t_eval,
                           method=method, rtol=rtol, atol=atol,
                           options=solver_options)[1:, :]
        else:
            xtraj = odeint(f, reshape_pt1_tonormal(x0), t_eval,
                           method=method, rtol=rtol, atol=atol,
                           options=solver_options)
    # Go back to CPU at end of simulation
    if not stay_GPU:
        return reshape_pt1(xtraj.cpu())
    else:
        return reshape_pt1(xtraj)

# Continuous-time EKF that takes distribution over current state, measurement
# and prior continuous-time dynamics with their Jacobian, outputs distribution
# over whole next state, using linear measurement function. xhat contains
# mean and flattened covariance of distribution over state. Expects ODE = a
# continuous-time dynamics model, and assumes state x = (mean, covar)
# https://en.wikipedia.org/wiki/Extended_Kalman_filter#Continuous-time_extended_Kalman_filter
class EKF_ODE:
    def __init__(self, device, kwargs):
        self.device = device
        self.n = kwargs.get('prior_kwargs').get('n')
        self.C = reshape_pt1(
            kwargs.get('prior_kwargs').get('observation_matrix'))

    def __call__(self, t, xhat, u, y, t0, init_control, ODE, kwargs,
                 impose_init_control=False):
        y = reshape_pt1(y(t, kwargs))
        return self.call_withyu(t, xhat, u, y, t0, init_control, ODE, kwargs,
                                impose_init_control)

    def call_withyu(self, t, xhat, u, y, t0, init_control, ODE, kwargs,
                 impose_init_control=False):
        device = xhat.device
        x = reshape_pt1(xhat)
        xhat = reshape_pt1(x[:, :self.n])
        covarhat = x[:, self.n:].view(self.n, self.n)
        # Prediction step: compute xhatdot without correction term and
        # Jacobian of estimated dynamics at (xhat, u)
        if ODE:
            if 'NODE' in ODE.__class__.__name__:
                mean = ODE.defunc.dyn_NODE(
                    t=t, x=xhat, u=u, t0=t0, init_control=init_control,
                    process_noise_var=0., kwargs=kwargs,
                    impose_init_control=impose_init_control)
                partial_ODE = lambda xt: ODE.defunc.dyn_NODE(
                    t=t, x=xt, u=u, t0=t0, init_control=init_control,
                    process_noise_var=0., kwargs=kwargs,
                    impose_init_control=impose_init_control)
                mean_deriv = ODE.predict_deriv(xhat, partial_ODE)
            else:
                mean = ODE(t=t, x=xhat, u=u, t0=t0, init_control=init_control,
                           process_noise_var=0., kwargs=kwargs,
                           impose_init_control=impose_init_control)
                mean_deriv = ODE.call_deriv(
                    t=t, x=xhat, u=u, t0=t0, init_control=init_control,
                    process_noise_var=0., kwargs=kwargs,
                    impose_init_control=impose_init_control)
        else:
            mean = torch.zeros_like(xhat)
            mean_deriv = torch.zeros((self.n, self.n), device=device)
        # Update step: compute correction term for xhatdot, K, and covarhatdot
        K = torch.matmul(torch.matmul(covarhat, self.C.t()), torch.inverse(
            kwargs.get('prior_kwargs').get('EKF_meas_covar')))
        S = torch.matmul(mean_deriv, covarhat)
        xhatdot = mean + \
                  torch.matmul(K, y.t() - torch.matmul(self.C, xhat.t())).t()
        covarhatdot = S + S.t() + \
                      kwargs.get('prior_kwargs').get('EKF_process_covar') - \
                      torch.matmul(torch.matmul(K, kwargs.get(
                          'prior_kwargs').get('EKF_meas_covar')), K.t())
        return torch.cat((reshape_pt1(xhatdot),
                          reshape_pt1(torch.flatten(covarhatdot))), dim=1)