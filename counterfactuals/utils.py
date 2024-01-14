import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib import cm, ticker
from numpy import ma

def plot_x_point(x, x_origin, model):
    model.eval()
    fig, ax = plt.subplots()
    x_res = x.detach().numpy().squeeze()
    x_origin = x_origin.numpy().squeeze()
    dist = np.linalg.norm(x_res - x_origin)

    with torch.no_grad():
        samples_zero, log_probs = model.sample_and_log_prob(512, context=torch.Tensor([[0]]))
        samples_one, log_probs = model.sample_and_log_prob(512, context=torch.ones(1, 1))


    ax.set_title(f"{x.detach().numpy()}, dist: {dist}")
    ax.scatter(samples_zero.squeeze()[:, 0], samples_zero.squeeze()[:, 1], c="g")
    ax.scatter(samples_one.squeeze()[:, 0], samples_one.squeeze()[:, 1], c="b")
    ax.arrow(x_origin[0], x_origin[1], x_res[0]-x_origin[0], x_res[1]-x_origin[1], width=0.025, length_includes_head=True, color="C3")
    ax.scatter(x_origin[0], x_origin[1], c="r")

def plot_model_distribution(model):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8,5)

    xline = torch.linspace(-1.5, 2.5, 200)
    yline = torch.linspace(-.75, 1.25, 200)
    xgrid, ygrid = torch.meshgrid(xline, yline)
    xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)
    
    with torch.no_grad():
        zgrid = model.log_prob(xyinput, torch.zeros(40000, 1)).exp().reshape(200, 200) \
                - model.log_prob(xyinput, torch.ones(40000, 1)).exp().reshape(200, 200) \

    cs = ax.contourf(xgrid.numpy(), ygrid.numpy(), zgrid, levels=50, cmap=cm.PuBu_r) # locator=ticker.LogLocator()
    cbar = fig.colorbar(cs)
    plt.show()

def plot_loss_space(x, x_origin, optim_function, **optim_function_params):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8,5)

    xline = torch.linspace(-1.5, 2.5, 200)
    yline = torch.linspace(-.75, 1.25, 200)
    xgrid, ygrid = torch.meshgrid(xline, yline)
    xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

    x_res = x.detach().numpy().squeeze()
    context_origin = torch.zeros((xyinput.shape[0], 1))
    context_target = torch.ones((xyinput.shape[0], 1))
    
    with torch.no_grad():
        zgrid, _, _, _ = optim_function(xyinput, x_origin, context_origin, context_target, **optim_function_params)
        zgrid = zgrid.log().reshape(200, 200)

    # zgrid[zgrid > 2] = 2
    cs = ax.contourf(xgrid.numpy(), ygrid.numpy(), zgrid, levels=100, cmap=cm.PuBu_r) #  locator=ticker.LogLocator()
    cbar = fig.colorbar(cs)
    ax.scatter(x_origin[0,0], x_origin[0,1], c="r")
    min_i = np.argmin(zgrid)
    print(torch.min(zgrid))

    ax.scatter(xyinput[min_i, 0], xyinput[min_i, 1], c="y")
    ax.arrow(x_origin[0, 0], x_origin[0, 1], x_res[0]-x_origin[0, 0], x_res[1]-x_origin[0, 1], width=0.025, length_includes_head=True, color="C3")
    plt.show()




def plot_distributions(x, x_orig, model, optim_function, alpha):
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(12,5)

    xline = torch.linspace(-1.5, 2.5, 200)
    yline = torch.linspace(-.75, 1.25, 200)
    xgrid, ygrid = torch.meshgrid(xline, yline)
    xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

    x_res = x.detach().numpy().squeeze()
    
    with torch.no_grad():
        zgrid0 = - model.log_prob(xyinput, torch.ones(40000, 1)).exp().reshape(200, 200) \
                - model.log_prob(xyinput, torch.zeros(40000, 1)).exp().reshape(200, 200)

        zgrid1 = optim_function(xyinput, x_orig, model, alpha).reshape(200, 200)

    # zgrid1 = ma.masked_where(zgrid1 <= 0, zgrid1)
    # zgrid1[zgrid1 > 2] = 2
    ax[0].contourf(xgrid.numpy(), ygrid.numpy(), zgrid0.numpy(), levels=20, cmap=cm.PuBu_r)
    cs = ax[1].contourf(xgrid.numpy(), ygrid.numpy(), zgrid1, levels=50, cmap=cm.PuBu_r) # locator=ticker.LogLocator()
    cbar = fig.colorbar(cs)
    ax[0].scatter(x_orig[0,0], x_orig[0,1], c="r")
    ax[1].scatter(x_orig[0,0], x_orig[0,1], c="r")

    ax[0].arrow(x_orig[0, 0], x_orig[0, 1], x_res[0]-x_orig[0, 0], x_res[1]-x_orig[0, 1], width=0.025, length_includes_head=True, color="C3")
    ax[1].arrow(x_orig[0, 0], x_orig[0, 1], x_res[0]-x_orig[0, 0], x_res[1]-x_orig[0, 1], width=0.025, length_includes_head=True, color="C3")
    log_p = model.log_prob(x, context=torch.Tensor([[1]])).exp().item()
    log_p_origin = model.log_prob(x_orig, context=torch.Tensor([[0]])).exp().item()
    ax[0].set_title(f"log_p_orig: {log_p_origin:.2e}")
    ax[1].set_title(f"Distance: {np.linalg.norm(x_orig - x_res):0.2f}, log_p: {log_p:.2e}")
    plt.show()

def flatten_dict(dd, separator ='_', prefix =''):
    return { prefix + separator + k if prefix else k : v
             for kk, vv in dd.items()
             for k, v in flatten_dict(vv, separator, kk).items()
             } if isinstance(dd, dict) else { prefix : dd }

def add_prefix_to_dict(d: dict, prefix: str) -> dict:
    return {prefix + "/" + k : v for k,v in d.items()}

def process_classification_report(report: dict, prefix: str) -> dict:
    report = flatten_dict(report)
    report = add_prefix_to_dict(report, prefix)
    return report