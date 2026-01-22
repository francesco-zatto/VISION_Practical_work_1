from nagel import run_nagel
from middlebury import readflo
import matplotlib.pyplot as plt

IM1_PATH = 'data/mysine/mysine9.png'
IM2_PATH = 'data/mysine/mysine10.png'
GT_PATH = 'data/mysine/correct_mysine.flo'

if __name__ == "__main__":
    I1 = plt.imread(IM1_PATH)
    I2 = plt.imread(IM2_PATH)

    alphas = [0.01, 0.1, 1.0, 10.0]

    run_nagel(I1, I2, N=1000, alphas=alphas, plot=True, data_name='mysine')