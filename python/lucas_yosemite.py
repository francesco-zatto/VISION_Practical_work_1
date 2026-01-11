from lucas import run_lucas
from middlebury import readflo
import matplotlib.pyplot as plt

IM1_PATH = '../data/yosemite/yos9.png'
IM2_PATH = '../data/yosemite/yos10.png'
GT_PATH = '../data/yosemite/correct_yos.flo'

if __name__ == "__main__":
    I1 = plt.imread(IM1_PATH)
    I2 = plt.imread(IM2_PATH)
    GT = readflo(GT_PATH)

    window_sizes = [7,9,11,13,15]

    run_lucas(I1, I2, GT=GT, window_sizes=window_sizes, plot=True, data_name='lucas_yosemite')
