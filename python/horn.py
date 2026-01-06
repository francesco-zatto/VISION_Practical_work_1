# Write the function horn(I1,I2,alpha,N) that returns compute the velocity map between images I1
# and I2 according to the Horn and Schunck method.

from gradhorn import gradhorn
import numpy as np
from scipy.signal import convolve2d

def horn(I1,I2,alpha,N):
    Ix, Iy, It = gradhorn(I1, I2)
    u = v = np.zeros_like(Ix)
    A = np.array([
        [1/12, 1/6, 1/12],
        [1/6, 0, 1/6],
        [1/12, 1/6, 1/12]
    ])
    for _ in range(N):
        u_avg = convolve2d(u, A, mode='same', boundary='symm')
        v_avg = convolve2d(v, A, mode='same', boundary='symm')
        update_term = (Ix * u_avg + Iy * v_avg + It) / (alpha + Ix**2 + Iy**2)
        u = u_avg - Ix * update_term
        v = v_avg - Iy * update_term
    return u, v

