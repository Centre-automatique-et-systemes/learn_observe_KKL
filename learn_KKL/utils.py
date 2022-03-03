import numpy as np
import torch
import torch.nn as nn
from smt.sampling_methods import LHS
from scipy import linalg

# Set double precision by default
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)


def generate_mesh(limits: np.array, num_samples: int,
                  method: str = 'LHS') -> torch.tensor:
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
        Use 'LHS' or 'uniform'.

    Returns
    ----------
    mesh: torch.tensor
        Mesh in shape (num_samples, 2).
    """

    # Sample either a uniform grid or use latin hypercube sampling
    if method == 'uniform':
        # Linspace upper bound and cut additional samples at random (
        # otherwise all cut in the same region!)
        axes = np.linspace(limits[:, 0], limits[:, 1], int(np.ceil(np.power(
            num_samples, 1/len(limits)))))
        axes_list = [axes[:, i] for i in range(axes.shape[1])]
        mesh = np.array(np.meshgrid(*axes_list)).T.reshape(-1, axes.shape[1])
        idx = np.random.choice(np.arange(len(mesh)), size=(num_samples,),
                               replace=False)
        print(f'Computed the smallest possible uniform grid for the '
              f'given dimensions, then deleted {len(mesh) - num_samples} '
              f'samples randomly to match the desired number of samples'
              f' {num_samples}.')
        mesh = mesh[idx]
    elif method == 'LHS':
        sampling = LHS(xlimits=limits)
        mesh = sampling(num_samples)
    else:
        raise NotImplementedError(f'The method {method} is not implemented')

    return torch.as_tensor(mesh)


def compute_h_infinity(A: np.array, B: np.array, C: np.array, epsilon: float = 1e-5) -> int:
    """
    Computes the H_infinity norm from a given system A, B, C with D being zero,
    for an given accucarcy epsilon.

    Parameters
    ----------
    A: np.array
    B: np.array
    C: np.array
    epsilon: float

    Returns
    -------
    singular_value: int
    """
    C_g = linalg.solve_continuous_lyapunov(A, -B.dot(B.T))
    O_g = linalg.solve_continuous_lyapunov(A.T, -C.T.dot(C))

    dim = 3
    r_lb = np.sqrt(np.trace(np.matmul(C_g, O_g))/dim)
    r_ub = 2*np.sqrt(dim*np.trace(np.matmul(C_g, O_g)))
    r = 0

    while(not r_ub - r_lb <= 2*epsilon*r_lb):
        r = (r_lb+r_ub)/2
        r_inv = 1/r
        M_r = np.block([[A, r_inv*B.dot(B.T)], [-r_inv*C.T.dot(C), -A.T]])
        eigen = np.linalg.eig(M_r)[0]
        image = np.where(np.abs(eigen.real) < 1e-14)
        if len(*image) == 0:
            r_ub = r
        else:
            r_lb = r

    return r


def MSE(x, y, dim=None):
    """
    Compute the mean squared error between x and y along dimension dim.

    Parameters
    ----------
    x: torch.tensor
    y: torch.tensor
    dim: int
        Dimension along which to compute the mean.

    Returns
    -------
    error: torch.tensor
        Computed RMSE.
    """
    error = torch.nn.functional.mse_loss(x, y, reduction='none')
    if dim is None:
        return torch.mean(error)
    else:
        return torch.mean(error, dim=dim)


def RMSE(x, y, dim=None):
    """
    Compute the root mean squared error between x and y along dimension dim.

    Parameters
    ----------
    x: torch.tensor
    y: torch.tensor
    dim: int
        Dimension along which to compute the mean.

    Returns
    -------
    error: torch.tensor
        Computed RMSE.
    """
    return torch.sqrt(MSE(x=x, y=y, dim=dim))


# Replaces sklearn StandardScaler()
# https://discuss.pytorch.org/t/pytorch-tensor-scaling/38576
class StandardScaler:
    def __init__(self, X, device):
        self._mean = torch.mean(X, dim=0)#.to(device)
        self._var = torch.var(X, dim=0, unbiased=False)#.to(device)
        # If var = 0., i.e. values all same, make it 1 so unchanged!
        idx = torch.nonzero(self._var == 0.)
        self._var[idx] += 1.
        self._scale = torch.sqrt(self._var)#.to(device)
        self.n_samples_seen_ = len(X)

    def fit(self, X):
        self._mean = torch.mean(X, dim=0)
        self._var = torch.var(X, dim=0, unbiased=False)
        # If var = 0., i.e. values all same, make it 1 so unchanged!
        idx = torch.nonzero(self._var == 0.)
        self._var[idx] += 1.
        self._scale = torch.sqrt(self._var)
        self.n_samples_seen_ = len(X)

    def transform(self, X):
        if torch.is_tensor(X):
            return (X - self._mean) / self._scale
        else:
            return (X - self._mean.numpy()) / self._scale.numpy()

    def inverse_transform(self, X):
        if torch.is_tensor(X):
            return self._scale * X + self._mean
        else:
            return self._scale.numpy() * X + self._mean.numpy()

    def set_scaler(self, mean, var):
        self._mean = mean
        self._var = var
        # If var = 0., i.e. values all same, make it 1 so unchanged!
        idx = torch.nonzero(self._var == 0.)
        self._var[idx] += 1.
        self._scale = torch.sqrt(self._var)

    def __str__(self):
        return f'Standard scaler of mean {self._mean} and var {self._var}\n'

# Simple MLP model with n hidden layers. Can pass StandardScaler to
# normalize input and output in forward function.
class MLPn(nn.Module):

    def __init__(self, num_hl, n_in, n_hl, n_out, activation=nn.Tanh(),
                 init=None, init_args={}, scaler_X=None, scaler_Y=None):
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        super(MLPn, self).__init__()
        # Initialize weights using "He Init" if ReLU after, "Xavier" otherwise
        if not init:
            init = nn.init.xavier_uniform_
        # Create ModuleList and add first layer with input dimension
        # Layers: input * activation, hidden * activation, output
        if isinstance(n_hl, int):
            n_hl = [n_hl] * (num_hl + 1)
        layers = nn.ModuleList()
        layers.append(nn.Linear(n_in, n_hl[0]))
        init(layers[-1].weight, *init_args)
        if 'xavier' not in init.__name__:
            init(layers[-1].bias, *init_args)  # not for tensors dim < 2
        # Add num_hl layers of size n_hl with chosen activation
        for i in range(num_hl):
            layers.append(activation)
            layers.append(nn.Linear(n_hl[i], n_hl[i + 1]))
            init(layers[-1].weight, *init_args)
            if 'xavier' not in init.__name__:
                init(layers[-1].bias, *init_args)  # not for tensors dim < 2
        # Append last layer with output dimension (linear activation)
        layers.append(nn.Linear(n_hl[-1], n_out))
        init(layers[-1].weight, *init_args)
        if 'xavier' not in init.__name__:
            init(layers[-1].bias, *init_args)  # not for tensors dim < 2
        self.layers = layers

    def set_scalers(self, scaler_X=None, scaler_Y=None):
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y

    def __call__(self, x):
        # Compute output through all layers. Normalize in, denormalize out
        if self.scaler_X:
            x = self.scaler_X.transform(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        if self.scaler_Y:
            x = self.scaler_Y.inverse_transform(x)
        return x

    def forward(self, x):
        return self(x)

    def freeze(self):
        # Freeze all model parameters
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        # Unfreeze all model parameters
        for param in self.parameters():
            param.requires_grad = True
