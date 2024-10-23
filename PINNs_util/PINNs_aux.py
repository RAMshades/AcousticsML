import torch 
import torch.nn as nn
import numpy as np

class FCN(nn.Module):
        def __init__(self, n_in, n_out, n_ffeatures, n_hidden, n_layers):
            super().__init__()
            self.pi = torch.acos(torch.zeros(1)).item() * 2
            self.n_ffeatures = n_ffeatures
            if n_ffeatures != 0:
                self.B = torch.nn.Parameter(torch.randn((n_in, n_ffeatures)))
                self.b = torch.nn.Parameter(torch.randn((1, n_ffeatures)))
                n_in = n_ffeatures

            activation = nn.Tanh
            self.fcs = nn.Sequential(*[
                            nn.Linear(n_in, n_hidden),
                            activation()])
            self.fch = nn.Sequential(*[
                            nn.Sequential(*[
                                nn.Linear(n_hidden, n_hidden),
                                activation()]) for _ in range(n_layers-1)])
            self.fce = nn.Linear(n_hidden, n_out)

        def forward(self, r):
            if self.n_ffeatures != 0:
                r = torch.cos( 2*self.pi*r @ self.B + self.b ) # Fourier mapping
            r = self.fcs(r)
            r = self.fch(r)
            r = self.fce(r)
            return r

def pde_residual(p, r, c):
    p_r = torch.autograd.grad(p, r, torch.ones_like(p), create_graph=True)[0]
    p_xx = torch.autograd.grad(p_r[:,0], r, torch.ones_like(p_r[:,0]), create_graph=True)[0][:,0:1]
    p_yy = torch.autograd.grad(p_r[:,1], r, torch.ones_like(p_r[:,1]), create_graph=True)[0][:,1:2]
    p_tt = torch.autograd.grad(p_r[:,2], r, torch.ones_like(p_r[:,2]), create_graph=True)[0][:,2:3]
    pde_res = p_xx + p_yy - p_tt/c**2
    return pde_res

def absorbing_boundary(p, r_bound, id_bound, c):
    p_r = torch.autograd.grad(p, r_bound, torch.ones_like(p), create_graph=True)[0]
    p_t =  p_r[:,2]
    if id_bound == 1:
        p_y = p_r[:,1]
        res_bound = -p_y + p_t/c
    elif id_bound == 2:
        p_y = p_r[:,1]
        res_bound = p_y + p_t/c
    elif id_bound == 3:
        p_x = p_r[:,0]
        res_bound = -p_x + p_t/c
    elif id_bound == 4:
        p_x = p_r[:,0]
        res_bound = p_x + p_t/c
    return res_bound

def loss_grad_norm(loss, model):
    loss_grad_norm = 0
    loss_clone = loss.clone()
    for params in model.parameters():
        loss_grad = torch.autograd.grad(loss_clone, params, retain_graph=True, allow_unused=True, materialize_grads=True)[0]
        loss_grad_norm += torch.sum(loss_grad**2)
    loss_grad_norm = loss_grad_norm**0.5
    loss_grad_norm = loss_grad_norm.detach()
    return loss_grad_norm

def update_lambda(model, loss_lst, lamb_lst, alpha):
    grad = []
    for loss in loss_lst:
        grad.append(loss_grad_norm(loss, model))
    grad_sum = sum(grad)
    lamb = []
    for i in range(len(grad)):
        lamb_hat = grad_sum / grad[i]
        if torch.isnan(lamb_hat) or torch.isinf(lamb_hat):
            lamb_hat = torch.ones_like(lamb_hat)
        lamb_new = alpha*lamb_lst[i] + (1-alpha)*lamb_hat
        lamb.append(lamb_new)
    return lamb

def xyt_tensor(rxy, t, device):
    n_t = len(t)
    n_xy = len(rxy)
    r = np.column_stack(
        (np.repeat(rxy, n_t, axis=0),
         np.tile(t, n_xy)))
    r = torch.tensor(r).view(-1,3).requires_grad_(True)
    r = r.to(device)
    return r