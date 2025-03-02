import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
import sys
sys.path.insert(1, '../Data/')


class Summaries_Dataset(Dataset):
    def __init__(self, params, xHI, Tb, Ts, LFs, tau):
        self.params = params
        self.xHI = xHI
        self.Tb = Tb
        self.Ts = Ts
        self.LFs = LFs
        self.tau = tau
        
    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        params = self.params[idx]
        xHI = self.xHI[idx]
        Tb = self.Tb[idx]
        Ts = self.Ts[idx]
        LFs = self.LFs[idx]
        tau = self.tau[idx]
        
        return (np.array(params, dtype = np.float32), 
                np.array(xHI, dtype = np.float32),
                np.array(Tb, dtype = np.float32),
                np.array(Ts, dtype = np.float32),
                np.array(LFs, dtype = np.float32),
                np.array(tau, dtype = np.float32),
               )
def get_normed_data(lstm=False):
    """
    Load the data and normalise it to [0,1].
    
    Returns
    -------
    normed_params : np.ndarray
        Normalised parameters.
    xHI : np.ndarray
        Neutral fraction (already normalised).
    normed_Tbs : np.ndarray
        Normalised 21-cm brightness temperature.
    normed_Tss : np.ndarray
        Normalised log10 of spin temperature.
    normed_UVLFs : np.ndarray
        Normalised log10 of UV luminosity functions.
    normed_tau : np.ndarray
        Normalised log10 of optical depth.
    """
    data_path = Path('../Data/db.npz')
    with np.load(data_path, allow_pickle = True) as f:
        param_labels = f['param_labels'] # last one is cosmo sigma_8
        redshifts = f['redshifts']
        xHIs = f['xHI']
        Tbs = f['Tb']
        params = f['params']
        if lstm:
            params = np.repeat(f['params'],len(redshifts), axis = 0).reshape((xHI.shape[0], len(redshifts), f['params'].shape[-1]))
            all_redshifts = np.array(list(redshifts) * f['xHI'].shape[0]).reshape((xHI.shape[0], len(redshifts)))
            params = np.append(params, all_redshifts[:,:,np.newaxis], axis = -1)
        params[:,[0,3,5,6,7,8]] = np.log10(params[:,[0,3,5,6,7,8]])
        UVLFs = f['UVLFs']
        Tss = f['Ts']
        LF_zs = f['LF_zs']
        M_UV = f['M_UV']
        taus = np.log10(f['tau_e'])
    # Normalise the data
    normed_params = normalize(params, bias = 0.0, scale = 1.0)
    Tb_bias = np.min(Tbs)
    Tb_scale = np.max(Tbs) - Tb_bias
    normed_Tbs = normalize(Tbs, bias = Tb_bias, scale = Tb_scale)

    Ts_bias = np.nanmin(Tss)
    Ts_scale = np.nanmax(Tss) - Ts_bias
    normed_Tss = normalize(Tss, bias = Ts_bias, scale = Ts_scale)

    nonans = np.where(~np.isnan(np.nanmean(UVLFs, axis=(1,2))))[0]
    UVLFs_bias = np.nanmin(UVLFs[nonans], axis = (0,1))
    UVLFs_scale = np.nanmax(UVLFs[nonans], axis = (0,1)) - UVLFs_bias
    normed_UVLFs = normalize(UVLFs, bias = UVLFs_bias, scale = UVLFs_scale)

    tau_bias = np.min(taus)
    tau_scale = np.max(taus) - tau_bias
    normed_tau = normalize(taus, bias = tau_bias, scale = tau_scale)
    norms = (Tb_bias, Tb_scale,
            Ts_bias, Ts_scale,
            UVLFs_bias, UVLFs_scale,
            tau_bias, tau_scale)
    return (normed_params, 
            xHIs, normed_Tbs, 
            normed_Tss, normed_UVLFs, normed_tau,
            norms)

def get_dataloaders(f_train=0.8, f_valid=0.1, batch_size=64, lstm=False):
    """
    Get the DataLoader objects for training, validation, and test sets.

    Parameters
    ----------
    f_train : float, optional
        Fraction of the database to use for training. Default is 0.8.
    f_valid : float, optional
        Fraction of the database to use for validation. Default is 0.1.
    batch_size : int, optional
        Batch size. Default is 64.
    
    Returns
    -------
    train_dataloader : DataLoader
        DataLoader object for training set.
    valid_dataloader : DataLoader
        DataLoader object for validation set.
    test_dataloader : DataLoader
        DataLoader object for test set
    """
    if f_train + f_valid >= 1.0:
        raise ValueError("f_train + f_valid must be strictly less than 1.0 to leave some data for the test set.")
    # Load data and normalise \in [0,1]
    normed_params, xHI, normed_Tbs, normed_Tss, normed_UVLFs, normed_tau, norms = get_normed_data(lstm=lstm)
    
    # Split into training, validation, and test sets
    N_samples = len(xHI)
    N_train = int(np.round(N_samples * f_train))
    N_valid = int(np.round(N_samples * f_valid))
    # Shuffle it first
    idx = np.arange(N_samples)
    np.random.shuffle(idx)
    theta = torch.Tensor(normed_params)[idx]
    theta_train = theta[:N_train]
    theta_valid = theta[N_train:N_train+N_valid]
    theta_test = theta[N_train+N_valid:]
    train_DS = Summaries_Dataset(theta_train, 
                                xHI[idx][:N_train], 
                                normed_Tbs[idx][:N_train], 
                                normed_Tss[idx][:N_train],
                                normed_UVLFs[idx][:N_train],
                                normed_tau[idx][:N_train],
                                )
    valid_DS = Summaries_Dataset(theta_valid, 
                                xHI[idx][N_train:N_train+N_valid], 
                                normed_Tbs[idx][N_train:N_train+N_valid], 
                                normed_Tss[idx][N_train:N_train+N_valid], 
                                normed_UVLFs[idx][N_train:N_train+N_valid],
                                normed_tau[idx][N_train:N_train+N_valid]
                                )
    test_DS = Summaries_Dataset(theta_test, 
                                xHI[idx][N_train+N_valid:], 
                                normed_Tbs[idx][N_train+N_valid:], 
                                normed_Tss[idx][N_train+N_valid:], 
                                normed_UVLFs[idx][N_train+N_valid:],
                                normed_tau[idx][N_train+N_valid:]
                                )
    
    train_dataloader = DataLoader(train_DS, 
                                  batch_size=batch_size, 
                                  shuffle=True, # shuffle the training set after every epoch
                                  generator = torch.Generator(device=device) # put the shuffling generator on the GPU if applicable
                                  )
    # for deterministic NNs, there is no need to shuffle the validation set
    # for generative NNs, we generate samples from a small subset of the validation set every epoch as a diagnostic
    # so it's useful to shuffle the validation set only for the generative NN.
    valid_dataloader = DataLoader(valid_DS, 
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                  generator = torch.Generator(device=device))
    test_dataloader = DataLoader(test_DS, batch_size=batch_size, shuffle=False)
    return train_dataloader, valid_dataloader, test_dataloader, norms

def normalize(data, bias=0.0, scale=1.0):
    data = data - bias
    data = data / scale
    return data

def unnormalize(data, bias=0.0, scale=1.0):
    data = data * scale
    data = data + bias
    return data
