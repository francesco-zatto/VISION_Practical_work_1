from horn_ad import run_horn_ad
import matplotlib.pyplot as plt
import utils

IM1_PATH = '../data/taxi/taxi9.png'
IM2_PATH = '../data/taxi/taxi10.png'

if __name__ == "__main__":
    I1 = plt.imread(IM1_PATH)
    I2 = plt.imread(IM2_PATH)

    u, v = run_horn_ad(I1, I2, alpha=15, max_iter=500, lr=1e-2)
    utils.plot_flow_results(u, v)
