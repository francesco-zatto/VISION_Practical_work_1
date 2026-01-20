import torch
import numpy as np
from gradhorn import gradhorn
torch.nn.HuberLoss
def gradient(f: torch.Tensor):
    fx_valid = f[:, 2:] - f[:, :-2]
    fx = torch.zeros_like(f)
    fx[:, 1:-1] = fx_valid / 2

    fy_valid = f[2:, :] - f[:-2, :]
    fy = torch.zeros_like(f)
    fy[1:-1, :] = fy_valid / 2

    return torch.stack([fx, fy], dim=-1)

def horn_ad_loss(u: torch.Tensor, v: torch.Tensor, Ix: torch.Tensor, Iy: torch.Tensor, It: torch.Tensor, alpha=1e-3, norm=2):
    data_term = torch.norm(Ix*u + Iy*v + It, p=norm)
    u_grad = gradient(u)
    v_grad = gradient(v)
    reg_term = alpha * (torch.norm(u_grad, p=norm)**2 + torch.norm(v_grad, p=norm)**2)
    return data_term + reg_term

def run_horn_ad(I1: np.ndarray, I2: np.ndarray, alpha=1e-1, lr=5e-3, max_iter=1000):
    Ix, Iy, It = gradhorn(I1, I2)

    Ix = torch.tensor(Ix, dtype=torch.float32)
    Iy = torch.tensor(Iy, dtype=torch.float32)
    It = torch.tensor(It, dtype=torch.float32)

    u = torch.zeros(I1.shape, dtype=torch.float32, requires_grad=True)
    v = torch.zeros(I1.shape, dtype=torch.float32, requires_grad=True)

    optimizer = torch.optim.LBFGS([u,v], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        L = horn_ad_loss(u, v, Ix, Iy, It)
        L.backward()
        return L

    optimizer.step(closure)
    
    return u.detach().numpy(), v.detach().numpy()