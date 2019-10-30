"""
Test for PyBullet cartpole environment
"""
import time
import tigercontrol
import jax.numpy as np


# cartpole test
def test_cartpole(verbose=False):
    environment = tigercontrol.environment("CartPole-v0")
    C_x, C_u = np.diag(np.array([0.1, 0.0, 1.0, 0.0])), np.diag(np.array([0.1]))
    L = lambda x, u: x.T @ C_x @ x + u.T @ C_u @ u
    dim_x, dim_u = 4, 1
    update_period = 75
    obs = environment.initialize()

    T = 75 # horizon
    threshold = 0.01
    lamb = 0.1
    max_iterations = 25

    method = tigercontrol.method("ILQR")
    method.initialize(environment, L, dim_x, dim_u, update_period, max_iterations, lamb, threshold)

    if verbose:
        print("Running iLQR...")
    # u = method.plan(obs, T, max_iterations, lamb, threshold)
    # u = method.plan_trajectory(obs, T, max_iterations, lamb, threshold)

    index = 0
    for t in range(10 * T):
        if verbose: 
            environment.render()
            time.sleep(1. / 50.)
        u = method.plan(obs)
        # obs, r, done, _ = environment.step(u[index])
        obs, r, done, _ = environment.step(u)
        index += 1

        
        if done:
            if verbose:
                print("lasted {} time steps".format(t+1))
            obs = environment.initialize()
        '''
        if done or index == T:
            if verbose:
                print("recomputing u...")
            u = method.plan(obs, T, max_iterations, lamb, threshold)
            index = 0'''

    environment.close()
    print("test_cartpole passed")


if __name__ == "__main__":
    test_cartpole(verbose=True)