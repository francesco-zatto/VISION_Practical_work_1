from lucas import run_lucas
from middlebury import readflo
import matplotlib.pyplot as plt

IM1_PATH = '../data/square/square9.png'
IM2_PATH = '../data/square/square10.png'
GT_PATH = '../data/square/correct_square.flo'

if __name__ == "__main__":
    I1 = plt.imread(IM1_PATH)
    I2 = plt.imread(IM2_PATH)
    GT = readflo(GT_PATH)

    window_sizes = [33,35,37,39,41,43,45,47]

    print("Square window: ")
    run_lucas(I1, I2, GT=GT, window_sizes=window_sizes, window_type='square', plot=True, data_name='lucas_square')
    print("Gaussian window: ")
    run_lucas(I1, I2, GT=GT, window_sizes=window_sizes, window_type='gaussian', plot=True, data_name='lucas_square_gaussian')
    print("Circular window: ")
    run_lucas(I1, I2, GT=GT, window_sizes=window_sizes, window_type='circular', plot=True, data_name='lucas_square_circular')
    
    
