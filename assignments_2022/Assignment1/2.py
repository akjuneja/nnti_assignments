import numpy as np

np.random.seed(42)


def func1():
    """
    Creates a random 4x4 matrix and finds it's transpose and inverses it. Prints each step.
    :return: None
    """
    matrix = np.random.random((4, 4))
    print("Random 4x4 matrix:", matrix)

    transposed = matrix.T
    print("Transposed matrix:", transposed)

    inverse_transpose = np.linalg.inv(transposed)
    print("Inverse matrix:", inverse_transpose)

    return


def func2():
    """
    Creates a random 4x3 matrix and finds e-values and e-vectors of that matrix. Prints each step.
    :return: None
    """
    matrix = np.random.random((4, 4))
    print("Random 4x4 matrix:", matrix)

    w, v = np.linalg.eig(matrix)

    print("Eigenvalues:", w)
    print("Eigenvectors:", v)

    return
