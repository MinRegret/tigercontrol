# test the crypto problem class

import ctsb
from ctsb.problems.time_series.crypto import Crypto
import jax.numpy as np
import matplotlib.pyplot as plt


def test_ctrl_indices(steps=1000, show_plot=False, verbose=False):
    T = steps
    problem = CtrlIndices()
    problem.initialize()
    assert problem.T == 0

    test_output = []
    for t in range(T):
        x_t, y_t = problem.step()
        test_output.append(y_t)

    assert problem.T == T
    
    info = problem.hidden()
    if verbose:
        print(info)
    if show_plot:
        plt.plot(test_output)
        plt.title("Control CtrlIndices")
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    print("test_ctrl_indices passed")
    return


if __name__=="__main__":
    test_ctrl_indices(show_plot=True)