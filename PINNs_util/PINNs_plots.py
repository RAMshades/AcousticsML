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
    t = np.linspace(0, T, p_data.shape[-1]) 
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.tight_layout()
    ax1.set_title('initial condition')
    ax2.set_title('data')
    ax1.set_ylabel('y')
    ax1.set_xlabel('x')
    ax2.set_xlabel('time')
    ax1.imshow(p_ref[:,:,0], vmin = -p_max, vmax = p_max, origin='lower', cmap='seismic', extent=[0,L,0,L])
    ax1.scatter(x_data, y_data, s=10, c='black', marker='^')
    ax2.plot(t, p_data.T)
    ax2.set_xlim([0, T])
    asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
    ax2.set_aspect(asp)
    plt.show()

def plot_inital_estimation(p_ref, p_est, L, title):
    p_max = np.max(np.abs(p_ref))
    n_L = p_ref.shape[0]
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.title.set_text('Initial condition')
    ax1.imshow(p_ref[:,:,0], vmin = -p_max, vmax = p_max, origin='lower', cmap='seismic', extent=[0,L,0,L])
    ax2.title.set_text(title)
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

def plot_train_log(loss, lamb, label):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8, 3))
    ax1.title.set_text('loss')
    for i in range(len(loss)):
        ax1.plot(np.asarray(loss[i])*np.asarray(lamb[i])/np.asarray(lamb[0]), label=label[i])
    ax1.legend()
    ax1.set_yscale("log")   
    ax1.set_xlabel("epochs")
    ax2.title.set_text('lambda')
    for i in range(len(lamb)):
        ax2.plot(lamb[i], label=label[i])
    ax2.legend()
    ax2.set_yscale("log")   
    ax2.set_xlabel("epochs")
    plt.show()