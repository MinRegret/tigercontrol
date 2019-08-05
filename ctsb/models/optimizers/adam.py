'''
Adam optimizer
'''

from ctsb.models.optimizers.core import Optimizer
from ctsb.models.optimizers.losses import mse
from jax import jit, grad
import jax.numpy as np

class Adam(Optimizer):
    """
    Description: Ordinary Gradient Descent optimizer.
    Args:
        pred (function): a prediction function implemented with jax.numpy 
        loss (function): specifies loss function to be used; defaults to MSE
        learning_rate (float): learning rate
    Returns:
        None
    """
    def __init__(self, pred=None, loss=mse, learning_rate=1.0, hyperparameters={}):
        self.initialized = False
        self.lr = learning_rate

        self.hyperparameters = {'beta_1': 0.9, 'beta_2': 0.999, 'eps': 0.00000001}
        self.hyperparameters.update(hyperparameters)
        self.beta_1, self.beta_2 = self.hyperparameters['beta_1'], self.hyperparameters['beta_2']
        self.beta_1_t, self.beta_2_t = self.beta_1, self.beta_2
        self.eps = self.hyperparameters['eps']

        self.max_norm = 1.

        self.m, self.v = None, None

        self.pred = pred
        self.loss = loss
        if self._is_valid_pred(pred, raise_error=False) and self._is_valid_loss(loss, raise_error=False):
            self.set_predict(pred, loss=loss)

    def update(self, params, x, y, loss=None):
        """
        Description: Updates parameters based on correct value, loss and learning rate.
        Args:
            params (list/numpy.ndarray): Parameters of model pred method
            x (float): input to model
            y (float): true label
            loss (function): loss function. defaults to input value.
        Returns:
            Updated parameters in same shape as input
        """
        assert self.initialized

        grad = self.gradient(params, x, y, loss=loss) # defined in optimizers core class

        if(self.m is None):
            self.m = np.zeros(grad.shape[0])
            self.v = 0

        self.m = self.beta_1 * self.m + (1. - self.beta_1) * grad
        self.v = self.beta_2 * self.m + (1. - self.beta_1) * grad.T @ grad

        # bias-corrected estimates
        m_t = self.m / (1 - self.beta_1_t)
        v_t = self.v / (1 - self.beta_2_t)

        # maintain current power of betas
        self.beta_1_t, self.beta_2_t = self.beta_1_t * self.beta_1, self.beta_2_t * self.beta_2

        self.max_norm = np.maximum(self.max_norm, np.linalg.norm(grad))
        lr = self.lr / self.max_norm

        return params - lr / (np.sqrt(v_t) + self.eps) * m_t