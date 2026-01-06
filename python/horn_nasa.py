from gradhorn import gradhorn
import error_functions
import matplotlib.pyplot as plt

IM1_PATH = '../data/nasa/nasa9.png'
IM2_PATH = '../data/nasa/nasa10.png'

if __name__ == "__main__":
    I1 = plt.imread(IM1_PATH)
    I2 = plt.imread(IM2_PATH)
    print(I1.shape, I2.shape)
    Ix, Iy, It = gradhorn(I1, I2)
    print(Ix.shape, Iy.shape, It.shape)
    plt.imshow(Ix, cmap='gray')
    plt.imshow(Iy, cmap='gray')
    plt.imshow(It, cmap='gray')
    plt.show()