from gradhorn import gradhorn
import error_functions
import matplotlib.pyplot as plt

IM1_PATH = '../data/rubberwhale/frame10.png'
IM2_PATH = '../data/rubberwhale/frame11.png'

if __name__ == "__main__":
    I1 = plt.imread(IM1_PATH)
    I2 = plt.imread(IM2_PATH)