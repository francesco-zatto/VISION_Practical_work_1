from gradhorn import gradhorn
import error_functions
import matplotlib.pyplot as plt


IM1_PATH = '../data/square/square9.png'
IM2_PATH = '../data/square/square10.png'

def debug_gradient(I1, I2, show=True):
    Ix, Iy, It = gradhorn(I1, I2)
    if show:
        plt.imshow(Ix, cmap='gray')
        plt.show()
        plt.imshow(Iy, cmap='gray')
        plt.show()
        plt.imshow(It, cmap='gray')
        plt.show()
    

if __name__ == "__main__":
    I1 = plt.imread(IM1_PATH)
    I2 = plt.imread(IM2_PATH)

    debug_gradient(I1, I2)