import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

def plot_global(redshifts, summary, 
                summary_pred=None, summary_fe=None,
                save=None, N=10, label=None):
    """
    Plot a global quantity as a function of redshift.

    Parameters
    ----------
    redshifts : array
        Redshift values.
    summary : array
        Global quantity to plot.
    summary_pred : array, optional
        Quantity to compare against the `summary`. Put quantities predicted by the NN here.
        Default is None.
    summary_fe : array, optional
        Fractional error between `summary` and `summary_pred`. Default is None.
    save : str, optional
        Path to save the plot. Default is None.
    N : int, optional
        Number of samples to plot. Default is 10.
    label : str, optional
        Name of the summary, used as label for the y-axis. Default is None.
    
    Returns
    -------
    None
    """
    rcParams.update({'font.size': 20})
    if summary is not None:
        if summary_pred is not None and summary_fe is not None:
            fig, ax = plt.subplots(nrows = 3, ncols = 1, figsize = (10,18),sharex=True, layout="constrained")
        if summary_pred is not None and summary_fe is None:
            fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (10, 12), sharex=True, layout="constrained")
        else:
            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,6), sharex=True, layout="constrained")
            ax = [ax]
    for i in range(N):
        ax[0].plot(redshifts, summary[i], color = 'k')
        if summary_pred is not None:
            ax[0].plot(redshifts, summary_pred[i], color = 'r')
            ax[-1].plot(redshifts, summary_pred[i]- summary_true[i], color = 'r')
            ax[-1].plot(redshifts, np.zeros_like(redshifts), color = 'k')
        if summary_fe is not None:
            ax[1].plot(redshifts, summary_fe[i], color = 'r')
            ax[1].plot(redshifts, np.zeros_like(redshifts), color = 'k')     
    ax[-1].set_xlabel('Redshift')
    if summary_fe is not None:
        ax[1].set_ylabel('FE (%)')
    ax[-1].set_ylabel('Abs. Diff')
    ax[0].set_ylabel(label)
    if save is not None:
        plt.savefig(save)
    else:
        plt.show()
    fig.clf()
    plt.close()
    del fig, ax, summary_pred, summary

def plot_uvlfs(M_UV, lfs, lfs_pred=None, 
               save=None, N=10):
    """
    Plot UV LFs for 7 redshift values as a function of UV magnitude.

    Parameters
    ----------
    M_UV : array
        UV magnitude values.
    lfs : array
        UV LFs to plot
    lfs_pred : array, optional
        LFs to compare against the `lfs`. Put LFs predicted by the NN here.
        Default is None.
    save : str, optional
        Path to save the plot. Default is None.
    N : int, optional
        Number of samples to plot. Default is 10.
    
    Returns
    -------
    None
    """
    rcParams.update({'font.size': 20})
    LF_zs = np.array([6.,7.,8.,9.,10.,12.,15.])
    fig, ax = plt.subplots(nrows = len(LF_zs), figsize = (13,25),ncols = 1, sharex=True, layout="constrained")
    for i in range(N):
        for j in range(len(LF_zs)):
            if lfs_pred is not None:
                ax[j].plot(M_UV, lfs_pred[i,:,j], color = 'r')
            ax[j].plot(M_UV, lfs[i,:,j], color = 'k')
            if j == len(LF_zs)-1:
                ax[j].set_xlabel(r'$M_{\rm UV}$')
            ax[j].set_ylabel(r'$\log_{10}\phi$ [Mpc$^{-3}$ mag$^{-1}$]')
    for i in range(len(LF_zs)):
        ax[i].text(0.1, 0.9, r'$z = $' + str(LF_zs[i]),
                   horizontalalignment='center',
                   verticalalignment='center', transform = ax[i].transAxes)
    if save is not None:
        plt.savefig(save)
    else:
        plt.show()
    fig.clf()
    plt.close()
    del fig, ax, lfs_pred, lfs

def plot_tau(tau, tau_pred=None, save=None, N=10):
    """
    Plot the true Thomson scattering optical depth to the CMB against the predicted one.

    Parameters
    ----------
    tau : array
        True optical depth values.
    tau_pred : array
        Predicted optical depth values.
    save : str, optional
        Path to save the plot. Default is None.
    N : int, optional
        Number of samples to plot. Default is 10.
    
    Returns
    -------
    None
    """
    if tau_pred is not None:
        fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex=True, layout="constrained")
        ax.scatter(tau, tau_pred, marker = '.', color = 'k')
        line = np.linspace(0,max(tau_true), 10)
        ax.plot(line, line, color = 'r', ls = '--')
        ax.set_xlabel(r'True $\tau_e$')
        ax.set_ylabel(r'Emulated $\tau_e$')
        if save is not None:
            plt.savefig(save)
        else:
            plt.show()
        fig.clf()
        plt.close()
        del fig, ax, tau_pred, tau
    else:
        fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex=True, layout="constrained")
        ax.hist(tau, histtype='step',bins=20, color = 'k')
        ax.set_xlabel(r'$\tau_e$')
        if save is not None:
            plt.savefig(save)
        else:
            plt.show()
        fig.clf()
        plt.close()
        del fig, ax, tau

def plot(redshifts=None, M_UV=None,xhi=None, tb=None, ts=None, lfs=None, tau=None, 
         xhi_pred=None, tb_pred=None, ts_pred=None, lfs_pred=None, tau_pred=None, 
         xhi_fe=None, tb_fe=None, ts_fe=None, lfs_fe=None, tau_fe=None,
         save=None, N=10):
    if (xhi is not None or tb is not None or ts is not None) and redshifts is None:
        raise ValueError("Must supply redshifts to plot a redshift-dependent quantity.")
    if xhi is not None and redshifts is not None:
        plot_global(redshifts, xhi, xhi_pred, xhi_fe, 
                save=str(save)+'xhi' if save is not None else None,
                 N=N, label = r'$\overline{x}_{\rm HI}$')
    if tb is not None and redshifts is not None:
        plot_global(redshifts, tb, tb_pred, tb_fe, 
                save=str(save)+'tb' if save is not None else None,
                 N=N, label = r'$\delta \overline{T}_b$ [mK]')
    if ts is not None and redshifts is not None:
        plot_global(redshifts, ts, ts_pred, ts_fe, 
                save=str(save)+'ts' if save is not None else None,
                 N=N, label = r'$\log_{10} \overline{T}_s$ [mK]')
    if lfs is not None:
        plot_uvlfs(M_UV, lfs, lfs_pred=lfs_pred, 
               save=str(save) + '/lfs' if save is not None else None,
               N=N)
    if tau is not None:
        plot_tau(tau, tau_pred, 
             save=str(save) + '/tau' if save is not None else None,
             N=N)