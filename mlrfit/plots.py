import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import pandas as pd
from collections import OrderedDict

from matplotlib.colors import ListedColormap
import random



def plot_ra_evolution_col(rhs0, figsize=(6,4), dpi=120, filename='', width=1, shuffle=False):
    plt.rcParams['text.usetex'] = True
    max_iters = max([rh.shape[0] for rh in rhs0])
    rhs = []
    # pad rhs
    for rh in rhs0:
        if rh.shape[0] < max_iters:
            rh = np.concatenate([rh, np.zeros((max_iters-rh.shape[0], rh.shape[1]))], axis=0)
        rhs += [rh]
        
    fig, axs = plt.subplots(len(rhs), 1, figsize=figsize, dpi=dpi, sharex=True)
    # cmap = sns.color_palette("Spectral", rhs[0].shape[1]+1, as_cmap=True)
    num_colors = rhs[0].shape[1]+1
    a = sns.color_palette("Spectral", num_colors).as_hex()
    a = np.array(list(a))
    if shuffle:
        idx = np.concatenate([np.arange(0, num_colors, 3), np.arange(1, num_colors, 3), np.arange(2, num_colors, 3)], axis=0)
        a = a[idx]
    cmap = ListedColormap(a)
    for i, rh in enumerate(rhs):
        if len(rhs) >= 2: ax = axs[i]
        else: ax = axs
        # assert np.allclose(rh.sum(axis=1), rh[0].sum())
        num_levels = rh.shape[1]
        df = pd.DataFrame(OrderedDict({r'$l=%d$'%(num_levels-i):val for i, val in \
                                        enumerate(reversed(np.array_split(rh.flatten('F'), rh.shape[1])))}),
                        index=list(map(str, range(rh.shape[0]))))
        sns.set_theme(style="white")

        # create stacked bar chart for monthly temperatures
        df.plot(kind='bar', stacked=True, ax=ax, width=width, legend=False, colormap=cmap)
            
        # labels for x & y axis
        ax.set_xlabel('iteration')
        ax.set_ylabel('rank allocation')
        ax.set_xticks(range(0, rh.shape[0], 5))
        ax.set_xticklabels(range(0, rh.shape[0], 5), rotation=90)

        ax.set_ylim([0, rh[0].sum()])
        ax.set_xlim([-0.5, rh.shape[0]-0.5])
    if len(rhs) >= 2: handles, labels = axs[1].get_legend_handles_labels()
    else: handles, labels = axs.get_legend_handles_labels()
    fig.legend(reversed(handles), reversed(labels), loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight')


def plot_loss_all_info(info, labels=None, figsize=(7, 5), dpi=120, logscale=False, hline=False, ylim=None, filename=''):
    sns.set_theme(style="whitegrid")
    plt.rcParams['text.usetex'] = True
    if labels is None:
        labels = sorted(info.keys())
    cmp = sns.color_palette("hls", len(labels))
    fig, axs = plt.subplots(1, figsize=figsize, dpi=dpi, sharey='row')
    for i, method in enumerate(labels):
        if method == 'LR':
            axs.axhline(y = info[method]['loss'][0], label=method, lw=1, color=cmp[i], ls='-')
        else:
            axs.plot(info[method]['loss'], label=method, lw=1, color=cmp[i])
            if 'epochs' in info[method]:
                epochs = info[method]['epochs'][:-1]
                axs.plot(epochs, np.array(info[method]['loss'])[epochs], \
                    marker='.', markersize=3, ls="", color=cmp[i])
            if hline:
                axs.axhline(y = info[method]['loss'][-1], linewidth=1, color=cmp[i], ls='--', lw=0.5)

    if logscale:
        axs.set_yscale('log')
    axs.grid(True)
    axs.set_ylabel(r'${\|A-\hat A\|_F}/{\|A\|_F}$')
    axs.set_xlabel('iteration')
    handles, labels = axs.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    if ylim is not None:
        plt.ylim(ylim)
    axs.tick_params(axis='both', which='major', labelsize=13)
    axs.tick_params(axis='both', which='minor', labelsize=13)
    fig.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_ff_loss_methods(info, figsize=(7, 6), dpi=120, iterations=True, filename=''):
    plt.rcParams['text.usetex'] = True
    sns.set_theme(style="whitegrid")
    labels = sorted(info.keys())
    cmp = sns.color_palette("hls", len(labels)+1)
    fig, axs = plt.subplots(2, figsize=figsize, dpi=dpi, sharex=True)
    for i, ax in enumerate(axs):
        m_ep = labels[i]
        num_iters = len(info[m_ep]['loss'])
        t = info[m_ep]['time']
        loss = info[m_ep]['loss'][-1]
        print(f"{m_ep:18s}, {num_iters=:3d}, {loss=:.8f}, {t=}")
        if iterations:
            x_range = np.arange(len(info[m_ep]['loss']))
        else:
            x_range = np.linspace(0, t+t/num_iters, num_iters)
        ax.plot(x_range, info[m_ep]['loss'], label=m_ep.upper(), marker='.', markersize=3, lw=1, color=cmp[i])
        ax.grid(True)
        ax.legend()
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.tick_params(axis='both', which='minor', labelsize=11)
        ax.set_ylabel(r'${\|A-\hat A\|_F}/{\|A\|_F}$')
        if i == len(axs)-1:
            ax.set_xlabel('iteration' if iterations else 'second')
    fig.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_ff_loss_methods_join(info, figsize=(7, 6), dpi=120, iterations=True, filename=''):
    plt.rcParams['text.usetex'] = True
    sns.set_theme(style="whitegrid")
    labels = sorted(info.keys())
    cmp = sns.color_palette("hls", len(labels)+1)
    fig, ax = plt.subplots(1, figsize=figsize, dpi=dpi)
    i = 0
    for i, m_ep in enumerate(labels):
        num_iters = len(info[m_ep]['loss'])
        t = info[m_ep]['time']
        loss = info[m_ep]['loss'][-1]
        print(f"{m_ep:18s}, {num_iters=:3d}, {loss=:.8f}, {t=}")
        if iterations:
            x_range = np.arange(len(info[m_ep]['loss']))
        else:
            x_range = np.linspace(0, t+t/num_iters, num_iters)
        ax.plot(x_range, info[m_ep]['loss'], label=m_ep.upper(), marker='.', markersize=3, lw=1, color=cmp[i])
    ax.grid(True)
    ax.legend()
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.tick_params(axis='both', which='minor', labelsize=11)
    ax.set_ylabel(r'${\|A-\hat A\|_F}/{\|A\|_F}$')
    ax.set_xlabel('iteration' if iterations else 'second')

    fig.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_np_sparse_linop_time(times, sizes, ns, figsize=(7, 7), dpi=120, filename=''):
    lw = 1
    plt.rcParams['text.usetex'] = True
    methods = sorted(times.keys())
    cmp = sns.color_palette("hls", len(methods)+1)
    fig, axs = plt.subplots(2, 1, figsize=figsize, dpi=dpi, sharex=True)
    for a, method in enumerate(methods):
        axs[0].plot(ns, times[method], marker='.', markersize=3, label=method, lw=lw, color=cmp[a])
        if method != 'np':
                axs[1].plot(ns, np.divide(np.array(times['np']),np.array(times[method])), \
                                marker='.', markersize=3, label='time np/'+method[-4:], lw=lw, color=cmp[a])
                axs[1].plot(ns, np.divide(np.array(sizes['np']),np.array(sizes[method])), \
                                marker='.', markersize=3, label='size np/'+method[-4:], lw=lw, ls=':', color=cmp[a])     
    for i in [0,1]:
        axs[i].set_yscale('log')
        axs[i].grid(True)
        axs[i].set_xlabel(r'$m$')
        axs[i].legend()
    axs[0].set_ylabel('ms')
    axs[1].set_ylabel("ratios  np / mlr")
    fig.tight_layout()
    if filename:
            plt.savefig("plots/%s.pdf"%filename, bbox_inches='tight')
    plt.show()


def plot_frob_error_rank(frob_losses):
    dpi = 120
    plt.rcParams['text.usetex'] = True
    fig, axs = plt.subplots(1, figsize=(5, 3), dpi=dpi, sharey='row')
    axs.plot(frob_losses)
    axs.set_xlabel(r'$r$')
    axs.set_ylabel(r'$\|A - A_r\|_F/\|A\|_F$')
    axs.set_yscale('log')
    axs.set_xscale('log')
    axs.grid(True)


def plot_ra_evolution_all(rhs, figsize=(6,4), dpi=120, filename=''):
    plt.rcParams['text.usetex'] = True
    fig, axs = plt.subplots(1, len(rhs), figsize=figsize, dpi=dpi, \
                             gridspec_kw={'width_ratios':[rh.shape[0] for rh in rhs]}, sharey=True)

    for i, rh in enumerate(rhs):
        ax = axs[i]
        assert np.allclose(rh.sum(axis=1), rh[0].sum())
        num_levels = rh.shape[1]
        df = pd.DataFrame(OrderedDict({r'$l=%d$'%(num_levels-i):val for i, val in \
                                        enumerate(reversed(np.array_split(rh.flatten('F'), rh.shape[1])))}),
                        index=list(map(str, range(rh.shape[0]))))
        sns.set_theme(style="white")

        # create stacked bar chart for monthly temperatures
        df.plot(kind='bar', stacked=True, ax=ax, width=1, legend=False)
            
        # labels for x & y axis
        ax.set_xlabel('iteration')
        ax.set_ylabel('rank allocation')
        ax.set_xticks(range(0, rh.shape[0], 5))
        ax.set_xticklabels(range(0, rh.shape[0], 5), rotation=90)

        ax.set_ylim([0, rh[0].sum()])
        ax.set_xlim([-0.5, rh.shape[0]-0.5])

    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(reversed(handles), reversed(labels), loc='upper right')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight')


def plot_ra_evolution(rh, figsize=(6,4), dpi=120, filename=''):
    assert np.allclose(rh.sum(axis=1), rh[0].sum())
    plt.rcParams['text.usetex'] = True
    num_levels = rh.shape[1]
    df = pd.DataFrame(OrderedDict({r'$l=%d$'%(num_levels-i):val for i, val in \
                                    enumerate(reversed(np.array_split(rh.flatten('F'), rh.shape[1])))}),
                    index=list(map(str, range(rh.shape[0]))))
    sns.set_theme(style="whitegrid")
    labels = [r'$l=%d$'%i for i in range(rh.shape[0])]

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.subplot(111)
    
    # create stacked bar chart for monthly temperatures
    df.plot(kind='bar', stacked=True, ax=ax, width=1)

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), loc='center left', bbox_to_anchor=(1, 0.5))
        
    # labels for x & y axis
    ax.set_xlabel('iteration')
    ax.set_ylabel('rank allocation')

    ax.set_ylim([0, rh[0].sum()])
    ax.set_xlim([-0.5, rh.shape[0]-0.5])
    if filename:
        plt.savefig(filename, bbox_inches='tight')


def plot_all_losses_methods(all_losses, title, dpi=120, save=False):
    cl_types = sorted(all_losses.keys())
    plt.rcParams['text.usetex'] = True
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(1, figsize=(10, 7), dpi=dpi)
    num_cl = len(cl_types)+1
    cmp = sns.color_palette("hls", num_cl)
    labels = sorted(all_losses[cl_types[0]].keys())
    labels.remove('LR')
    
    loss = all_losses[cl_types[0]]['LR']
    ax.axhline(y = loss[0], label="LR", linewidth=1, color=cmp[num_cl-1])
    
    for i, mode in enumerate(labels):
        for j, cl_type in enumerate(cl_types):
            loss, epochs = all_losses[cl_type][mode]
            ax.plot(loss, label=cl_type, linewidth=0.9, color=cmp[j])

            ax.set_ylabel(r'$\|A-\hat A\|_F~/~\|A\|_F$')
            ax.grid(True)
            ax.set_yscale('log')
            ax.vlines(x=epochs, ymin=0, ymax=1, color=cmp[j], lw=0.8, ls='--')

    fig.suptitle(title)
    fig.legend(fontsize=15)
    fig.tight_layout()
    if save:
        plt.savefig("%s.pdf"%title, bbox_inches='tight')


def plot_block_coord_epochs(all_losses, title, dpi=120, sharey=False):
    cl_types = sorted(all_losses.keys())
    plt.rcParams['text.usetex'] = True
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(1, len(cl_types), figsize=(10, 7/len(cl_types)), dpi=dpi, sharey=sharey)
    num_algo = len(all_losses[cl_types[0]])
    cmp = sns.color_palette("hls", num_algo)
    labels = sorted(all_losses[cl_types[0]].keys())
    
    for i, cl_type in enumerate(cl_types):
        epochs = np.array(all_losses[cl_type][labels[-1]][1])
        if len(cl_types)>=2:
            ax[i].vlines(x=epochs, ymin=0, ymax=1, color='k', lw=0.3, ls='--')
        else:
            ax.vlines(x=epochs, ymin=0, ymax=1, color='k', lw=0.3, ls='--')
        
        for j, mode in enumerate(labels):
            loss = all_losses[cl_type][mode]
            if len(cl_types)>=2:
                if "LR" in mode: ax[i].axhline(y = loss[0], label=mode, linewidth=1, color=cmp[j]) 
                else:
                    loss = loss[0]
                    ax[i].plot(loss, label=mode, linewidth=0.9, color=cmp[j])
            else:
                if "LR" in mode: ax.axhline(y = loss[0], label=mode, linewidth=1, color=cmp[j]) 
                else:
                    loss = loss[0]
                    ax.plot(loss, label=mode, linewidth=0.9, color=cmp[j])
        if len(cl_types)>=2:
            ax[i].set_ylabel(r'$\|A-\hat A\|_F~/~\|A\|_F$')
            ax[i].set_title("%s clustering"%cl_type)
            ax[i].grid(True)
            ax[i].set_xticks(epochs, np.arange(epochs.size))
            ax[i].set_yscale('log')
        else:
            ax.set_ylabel(r'$\|A-\hat A\|_F~/~\|A\|_F$')
            ax.set_title("%s clustering"%cl_type)
            ax.grid(True)
            ax.set_xticks(epochs, np.arange(epochs.size))
            ax.set_yscale('log')
       
    if len(cl_types)>=2:
        handles, labels = ax[-1].get_legend_handles_labels()
    else:
        handles, labels = ax.get_legend_handles_labels()
    fig.suptitle(title)
    fig.legend(handles, labels)
    fig.tight_layout()


def plot_hist_rad_kern_matrices(A, Dist, La, kern_func, kern_type, d, n, sigma):
    plt.rcParams['text.usetex'] = True
    fig = plt.figure(figsize=(7, 12), dpi=120)
    axes = fig.subplots(nrows=La+2, ncols=2)
    l = 0
    cax = axes[l, 0].matshow(Dist, cmap='hot')
    axes[l, 0].set_title(f"Dist")
    cbar = fig.colorbar(cax, ax=axes[l,0])
    cbar.ax.tick_params(labelsize=6)
    axes[l, 1].hist(Dist.flatten(), bins=30, color='blue')
    axes[l, 1].set_title('histogram Dist')
    axes[l, 1].set_yscale('log')
    for l in range(1, La+1):
        Kl = kern_func(Dist / (sigma / 2**(l-1)))
        cax = axes[l, 0].matshow(Kl, cmap='hot')
        axes[l, 0].set_title(f"Kl, l={l-1}")
        cbar = fig.colorbar(cax, ax=axes[l,0])
        cbar.ax.tick_params(labelsize=6)
        axes[l, 1].hist(Kl.flatten(), bins=30, color='blue')
        axes[l, 1].set_title('histogram Kl')
        axes[l, 1].set_yscale('log')
    l = La + 1
    cax = axes[l, 0].matshow(A, cmap='hot')
    axes[l, 0].set_title(f"A")
    cbar = fig.colorbar(cax, ax=axes[l,0])
    cbar.ax.tick_params(labelsize=6)
    axes[l, 1].hist(A.flatten(), bins=30, color='blue')
    axes[l, 1].set_title('histogram A')
    axes[l, 1].set_yscale('log')
    for ax in axes.flat:
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.tick_params(axis='both', which='minor', labelsize=6)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    fig.suptitle(f"{d=}, {n=}, {sigma=}, {kern_type=}", fontsize=16, y=0.98)
    plt.savefig(f"radkern_Al_Kl:{d=}, {n=}, {sigma=}, {kern_type=}.pdf", format="pdf")
    plt.show()
    