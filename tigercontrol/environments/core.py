# Environment class
# Author: John Hallman

from tigercontrol import error
import inspect
import jax
import jax.numpy as np

# class for online control tests
class Environment(object):
    spec = None

    def __init__(self):
        self.initialized = False

    def initialize(self, **kwargs):
        ''' Description: resets environment to time 0 '''
        raise NotImplementedError

    def step(self, **kwargs):
        ''' Description: run one timestep of the environment's dynamics. '''
        raise NotImplementedError

    def rollout(self, baby_controller, T, dynamics_grad=False, loss_grad=False, loss_hessian=False):
        """ Description: Roll out and return trajectory of given baby_controller. """
        raise NotImplementedError

""" # OLD CODE
    def rollout(self, baby_controller, T, dynamics_grad=False, loss_grad=False, loss_hessian=False):
        # Description: Roll out trajectory of given baby_controller.
        request_grad = dynamics_grad or loss_grad or loss_hessian
        if not hasattr(self, "compiled") and request_grad: # on first call, compile gradients
            if '_dynamics' not in vars(self):
                raise NotImplementedError("rollout not possible: no dynamics in {}".format(self))
            if '_loss' not in vars(self):
                raise NotImplementedError("rollout not possible: no loss in {}".format(self))
            try:
                # stack the jacobians of environment dynamics gradient
                jacobian = jax.jacrev(self._dynamics, argnums=(0,1))
                self._dynamics_jacobian = jax.jit(lambda x, u: np.hstack(jacobian(x, u)))
                # stack the gradients of environment loss
                loss_grad = jax.grad(self._loss, argnums=(0,1))
                self._loss_grad = jax.jit(lambda x, u: np.hstack(loss_grad(x, u)))
                # block the hessian of environment loss
                block_hessian = lambda A: np.vstack([np.hstack([A[0][0], A[0][1]]), np.hstack([A[1][0], A[1][1]])])
                hessian = jax.hessian(self._loss, argnums=(0,1))
            except Exception as e:
                print(e)
                raise error.JAXCompilationError("jax.jit failed to compile environment dynamics or loss")
            self.compiled = True

        transcript = {'x': [], 'u': []} # return transcript
        if dynamics_grad: transcript['dynamics_grad'] = [] # optional derivatives
        if loss_grad: transcript['loss_grad'] = []
        if loss_hessian: transcript['loss_hessian'] = []

        x_origin, x = self._state, self._state
        for t in range(T):
            u = baby_controller.get_action(x)
            transcript['x'].append(x)
            transcript['u'].append(u)
            if dynamics_grad: transcript['dynamics_grad'].append(self._dynamics_jacobian(x, u))
            if loss_grad: transcript['loss_grad'].append(self._loss_grad(x, u))
            if loss_hessian: transcript['loss_hessian'].append(self._loss_hessian(x, u))
            x = self.step(u)[0] # move to next state
        self._state = x_origin # return to original state
        return transcript
"""

    def get_loss(self):
        return self._loss

    def close(self):
        ''' Description: closes the environment and returns used memory '''
        pass

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)

    def __repr__(self):
        return self.__str__()
