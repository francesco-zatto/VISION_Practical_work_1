from horn import horn
import error_functions
import matplotlib.pyplot as plt
import numpy as np
from middlebury import readflo, computeColor

IM1_PATH = '../data/mysine/mysine9.png'
IM2_PATH = '../data/mysine/mysine10.png'
GT_PATH = '../data/mysine/correct_mysine.flo'

def find_optimal_alpha(I1, I2, GT, alphas, N=1000):
    errors = []
    for alpha in alphas:
        u, v = horn(I1, I2, alpha, N)
        mean, _ = error_functions.end_point_error((u, v), GT)
        computeColored = computeColor(u, v, True)
        errors.append(mean)
    optimal_alpha = alphas[np.argmin(errors)]
    print(f'Optimal alpha: {optimal_alpha} with EPE: {min(errors)}')
    return optimal_alpha

if __name__ == "__main__":
    I1 = plt.imread(IM1_PATH)
    I2 = plt.imread(IM2_PATH)
    GT = readflo(GT_PATH)
    GT = GT[:-1, :-1].transpose(2, 0, 1)

    alphas = 10.0 ** np.linspace(-5, 1, 7)
    optimal_alpha = find_optimal_alpha(I1, I2, GT, alphas)

    Ns = [100, 500, 1000, 5000]
    for N in Ns:
        u, v = horn(I1, I2, optimal_alpha, N)
        mean_epe, std_epe = error_functions.end_point_error((u, v), GT)
        mean_ae, std_ae = error_functions.angular_error((u, v), GT)
        mean_ne, std_ne = error_functions.norm_error((u, v), GT)
        mean_rne, std_rne = error_functions.relative_norm_error((u, v), GT)
        print(f'N={N}: EPE={mean_epe:.4f}±{std_epe:.4f}, AE={mean_ae:.4f}±{std_ae:.4f}, '
              f'NE={mean_ne:.4f}±{std_ne:.4f}, RNE={mean_rne:.4f}±{std_rne:.4f}')
        computeColored = computeColor(u, v, True)
        plt.imshow(computeColored)
        plt.title(f'Horn Method Optical Flow (N={N})')
        plt.show()
    