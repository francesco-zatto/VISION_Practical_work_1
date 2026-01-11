from lucas import run_lucas
import matplotlib.pyplot as plt

IM1_PATH = '../data/rubic/rubic9.png'
IM2_PATH = '../data/rubic/rubic10.png'

if __name__ == "__main__":
    I1 = plt.imread(IM1_PATH)
    I2 = plt.imread(IM2_PATH)

    window_sizes = [3,5,7,9,11]

    run_lucas(I1, I2, window_sizes=window_sizes, plot=True, data_name='lucas_rubic')
