from lucas import run_lucas
from middlebury import readflo
import numpy as np
from PIL import Image

IM1_PATH = '../data/rubberwhale/frame10.png'
IM2_PATH = '../data/rubberwhale/frame11.png'
GT_PATH = '../data/rubberwhale/correct_rubberwhale10.flo'

if __name__ == "__main__":
    I1 = np.array(Image.open(IM1_PATH).convert('L'), dtype=np.float32)
    I2 = np.array(Image.open(IM2_PATH).convert('L'), dtype=np.float32)
    GT = readflo(GT_PATH)

    window_sizes = [3,5,7,9,11]

    run_lucas(I1, I2, GT=GT, window_sizes=window_sizes, plot=True, data_name='lucas_rubberwhale')

    