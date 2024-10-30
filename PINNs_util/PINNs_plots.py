import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plot_field(p, L, title):
    p_max = np.max(np.abs(p))
    plt.rcParams["animation.html"] = "jshtml"
    plt.ioff()
    frames = []
    fig, [ax1, ax2] = plt.subplots(1, 2, gridspec_kw={"width_ratios":[50,1]})
    cmap = matplotlib.cm.seismic
    norm = matplotlib.colors.Normalize(vmin=-p_max, vmax=p_max)
    cbar = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, orientation='vertical')
    ax1.set_title(title)
    for i in range(p.shape[-1]):   
        p_plot = p[:,:,i]
        img = ax1.imshow(p_plot, vmin = -p_max, vmax = p_max, origin='lower', cmap='seismic', extent=[0,L,0,L], animated=True)
        frames.append([img])
        ani = animation.ArtistAnimation(fig, frames, interval=200, blit=True)
    plt.close()
    return ani

def plot_data(p_ref, p_data, x_data, y_data, L, T):
    p_max = np.max(np.abs(p_ref))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('initial condition')
    ax2.set_title('data')
    ax1.set_ylabel('y')
    ax1.set_xlabel('x')
    ax2.set_ylabel('time')
    ax2.set_xlabel('x_receiver')
    ax1.imshow(p_ref[:,:,0], vmin = -p_max, vmax = p_max, origin='lower', cmap='seismic', extent=[0,L,0,L])
    ax1.scatter(x_data, y_data, s=10, c='black', marker='^')
    ax2.imshow(p_data.T, vmin = -p_max, vmax = p_max, origin='lower', cmap='seismic', extent=[0,L,0,T])
    plt.show()

def plot_inital_estimation(p_ref, p_est, L):
    p_max = np.max(np.abs(p_ref))
    n_L = p_ref.shape[0]
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.title.set_text('Initial condition')
    ax1.imshow(p_ref[:,:,0], vmin = -p_max, vmax = p_max, origin='lower', cmap='seismic', extent=[0,L,0,L])
    ax2.title.set_text('Estimation - no FF')
    ax2.imshow(np.reshape(p_est[:,1], (n_L, n_L)), vmin = -p_max, vmax = p_max, origin='lower', cmap='seismic', extent=[0,L,0,L])
    plt.show()

def plot_estimation(p_ref, p_est, L):
    p_max = np.max(np.abs(p_ref))
    n_L = p_ref.shape[0]
    n_T = p_ref.shape[-1]
    plt.rcParams["animation.html"] = "jshtml"
    plt.ioff()
    frames = []
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.title.set_text('Reference')
    ax2.title.set_text('Estimated')
    for i in range(0, n_T):   
        img1 = ax1.imshow(p_ref[:,:,i], vmin = -p_max, vmax = p_max, origin='lower', cmap='seismic', extent=[-L/2,L/2,-L/2,L/2], animated=True)
        img2 = ax2.imshow(np.reshape(p_est[:,i], (n_L, n_L)), vmin = -p_max, vmax = p_max, origin='lower', cmap='seismic', extent=[-L/2,L/2,-L/2,L/2], animated=True)
        frames.append([img1, img2])
        ani = animation.ArtistAnimation(fig, frames, interval=200, blit=True)
    plt.close()
    return ani

def plot_train_log(loss_data_hist, loss_ini_hist, loss_pde_hist, lamb_data_hist, lamb_ini_hist, lamb_pde_hist):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8, 3))
    ax1.title.set_text('loss')
    ax1.plot(np.asarray(loss_pde_hist)*np.asarray(lamb_pde_hist)/np.asarray(lamb_data_hist), label="pde loss")
    ax1.plot(np.asarray(loss_ini_hist)*np.asarray(lamb_ini_hist)/np.asarray(lamb_data_hist), label="ic loss")
    ax1.plot(loss_data_hist, label="data loss")
    ax1.legend()
    ax1.set_yscale("log")   
    ax1.set_xlabel("epochs")
    ax2.title.set_text('lambda')
    ax2.plot(lamb_pde_hist, label="lambda_pde")
    ax2.plot(lamb_ini_hist, label="lambda_ic")
    ax2.plot(lamb_data_hist, label="lambda_data")
    ax2.legend()
    ax2.set_yscale("log")   
    ax2.set_xlabel("epochs")
    plt.show()

def plot_train_log_bound(loss_data_hist, loss_ini_hist, loss_pde_hist, loss_bound_hist, lamb_data_hist, lamb_ini_hist, lamb_pde_hist, lamb_bound_hist):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8, 3))
    ax1.title.set_text('loss')
    ax1.plot(np.asarray(loss_pde_hist)*np.asarray(lamb_pde_hist)/np.asarray(lamb_data_hist), label="pde loss")
    ax1.plot(np.asarray(loss_ini_hist)*np.asarray(lamb_ini_hist)/np.asarray(lamb_data_hist), label="ic loss")
    ax1.plot(np.asarray(loss_bound_hist)*np.asarray(lamb_bound_hist)/np.asarray(lamb_data_hist), label="bound loss")
    ax1.plot(loss_data_hist, label="data loss")
    ax1.legend()
    ax1.set_yscale("log")   
    ax1.set_xlabel("epochs")
    ax2.title.set_text('lambda')
    ax2.plot(lamb_pde_hist, label="lambda_pde")
    ax2.plot(lamb_ini_hist, label="lambda_ic")
    ax2.plot(lamb_bound_hist, label="lambda_bound")
    ax2.plot(lamb_data_hist, label="lambda_data")
    ax2.legend()
    ax2.set_yscale("log")   
    ax2.set_xlabel("epochs")
    plt.show()