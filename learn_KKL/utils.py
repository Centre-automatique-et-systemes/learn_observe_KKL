import torch

# Set double precision by default
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)


def MSE(x, y, dim=None):
    """
    Compute the mean squared between x and y along dimension dim.

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
    Compute the root mean squared between x and y along dimension dim.

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
    def __init__(self, X):
        self._mean = torch.mean(X, dim=0)
        self._var = torch.var(X, dim=0, unbiased=False)
        # If var = 0., i.e. values all same, make it 1 so unchanged!
        idx = torch.nonzero(self._var == 0.)
        self._var[idx] += 1.
        self._scale = torch.sqrt(self._var)
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
