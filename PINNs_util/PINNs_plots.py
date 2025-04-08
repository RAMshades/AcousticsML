import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plot_field(p, L, title):
    p_max = np.max(np.abs(p))
    plt.rcParams["animation.html"] = "jshtml"
    plt.ioff()
    frames = []
    fig, [ax1, ax2] = plt.subplots(1, 2, gridspec_kw={"width_ratios":[50,1]}, figsize=(4,3))
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

def plot_estimation(p_ref, p_est, L):
    p_max = np.max(np.abs(p_ref))
    n_L = p_ref.shape[0]
    n_T = p_ref.shape[-1]
    p_est = p_est.reshape(n_L, n_L, n_T)
    plt.rcParams["animation.html"] = "jshtml"
    plt.ioff()
    frames = []
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(6,3))
    ax1.title.set_text('Reference')
    ax2.title.set_text('Estimated')
    for i in range(0, n_T):   
        img1 = ax1.imshow(p_ref[:,:,i], vmin = -p_max, vmax = p_max, origin='lower', cmap='seismic', extent=[-L/2,L/2,-L/2,L/2], animated=True)
        img2 = ax2.imshow(p_est[:,:,i], vmin = -p_max, vmax = p_max, origin='lower', cmap='seismic', extent=[-L/2,L/2,-L/2,L/2], animated=True)
        frames.append([img1, img2])
        ani = animation.ArtistAnimation(fig, frames, interval=200, blit=True)
    plt.close()
    return ani

def plot_train_log(loss, lamb, label):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8, 3))
    ax1.title.set_text('loss')
    for i in range(len(loss)-1,-1,-1):
        ax1.plot(np.asarray(loss[i])*np.asarray(lamb[i])/np.asarray(lamb[0]), label=label[i])
    ax1.legend()
    ax1.set_yscale("log")   
    ax1.set_xlabel("epochs")
    ax2.title.set_text('lambda')
    for i in range(len(lamb)-1,-1,-1):
        ax2.plot(lamb[i], label=label[i])
    ax2.legend()
    ax2.set_yscale("log")   
    ax2.set_xlabel("epochs")
    plt.show()

def plot_speed(c, L):
    c_max = np.max(np.abs(c))
    c_min = np.min(np.abs(c))
    fig,ax = plt.subplots(figsize=(4,3))
    img = ax.imshow(c, vmin = c_min, vmax = c_max, origin='lower', cmap='viridis', extent=[0,L,0,L])
    fig.colorbar(img, ax=ax)
    ax.set_title('wave speed c(x,y)')
    plt.show()

def plot_speed_log(c_hist, c_ref, L):
    c_max = np.max(np.abs(c_ref))
    c_min = np.min(np.abs(c_ref))
    n_L = c_ref.shape[0]
    plt.rcParams["animation.html"] = "jshtml"
    plt.ioff()
    frames = []
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(6,3))
    ax1.title.set_text('Reference')
    ax2.title.set_text('Estimated')
    for i in range(len(c_hist)):   
        img1 = ax1.imshow(c_ref, vmin = c_min, vmax = c_max, origin='lower', cmap='viridis', extent=[-L/2,L/2,-L/2,L/2], animated=True)
        img2 = ax2.imshow(np.reshape(c_hist[i], (n_L, n_L)), vmin = c_min, vmax = c_max, origin='lower', cmap='viridis', extent=[-L/2,L/2,-L/2,L/2], animated=True)
        frames.append([img1, img2])
        ani = animation.ArtistAnimation(fig, frames, interval=200, blit=True)
    plt.close()
    return ani

def plot_comparison(p_ref, p_est, L, T, t):
    n_L = p_ref.shape[0]
    n_T = p_ref.shape[-1]
    p_est = p_est.reshape(n_L, n_L, n_T)
    p_ref_max = np.max(p_ref)
    p_ref_min = -p_ref_max

    time_indices = [0, int(n_T * 0.33), int(n_T * 0.66), n_T - 1]
    time_labels = [t[idx]*5/3 for idx in time_indices]

    fig, axes = plt.subplots(2, 6, figsize=(22, 10), gridspec_kw={'width_ratios': [0.2, 1, 1, 1, 1, 0.2]})

    axes[0, 0].text(0.5, 0.5, 'Reference', fontsize=24, ha='center', va='center', rotation=0)
    axes[1, 0].text(0.5, 0.5, 'PINN', fontsize=24, ha='center', va='center', rotation=0)

    for ax in axes[:, 0]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    for ax in axes[:, -1]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    for i, idx in enumerate(time_indices):
        ax = axes[0, i + 1]
        im = ax.imshow(p_ref[:, :, idx], extent=[0, 5, 0, 5], origin='lower', cmap='seismic', vmin=p_ref_min, vmax=p_ref_max)
        ax.set_title(f"t={time_labels[i]:.2f} s", fontsize=22)
        ax.set_xlabel('x (km)', fontsize=20)
        ax.set_ylabel('y (km)', fontsize=20)

    for i, idx in enumerate(time_indices):
        ax = axes[1, i + 1]
        im = ax.imshow(p_est[:, :, idx], extent=[0, 5, 0, 5], origin='lower', cmap='seismic', vmin=p_ref_min, vmax=p_ref_max)
        ax.set_xlabel('x (km)', fontsize=20)
        ax.set_ylabel('y (km)', fontsize=20)

    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7]) 
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Normalized pressure', fontsize=24)
    cbar.ax.tick_params(labelsize=20) 

    for ax in axes[0, 1:]:
        ax.tick_params(axis='both', which='major', labelsize=20)

    for ax in axes[1, 1:]:
        ax.tick_params(axis='both', which='major', labelsize=20)

    plt.tight_layout()
    plt.show()