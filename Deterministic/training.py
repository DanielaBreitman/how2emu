
from tqdm.auto import tqdm
import numpy as np

# Training loop

def train(model, 
          train_dataloader, 
          optimizer, 
          criterion, 
          epoch, 
          weights=None,
          device='cpu'):
    pbar = tqdm(range(len(train_dataloader)),
                              total=len(train_dataloader),
                              desc ="Epoch " + str(epoch))
    model.train()
    tmp = []
    for step in pbar:
        optimizer.zero_grad()
        theta, xhi_true, tb_true, ts_true, lfs_true, tau_true = next(iter(train_dataloader))

        theta = theta.to(device)
        
        pred = model(theta)

        loss = criterion((xhi_true, tb_true, ts_true, lfs_true, tau_true), pred, weights=weights)
        loss.backward()
        tmp.append(loss.item())
        optimizer.step()
        pbar.set_postfix(total_loss=np.mean(tmp))
    return torch.mean(tmp), model, optimizer


def validate(model, valid_dataloader, weights=None):
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        pbar = tqdm(range(len(valid_dataloader)),
                              total=len(valid_dataloader),
                              desc ="Epoch " + str(epoch))
        model.eval()
        vtmp = []
        for step in pbar:
            theta, xhi_true, tb_true, ts_true, lfs_true, tau_true = next(iter(valid_dataloader))

            theta = theta.to(device)
            
            pred = model(theta)

            vloss = criterion((xhi_true, tb_true, ts_true, lfs_true, tau_true), pred, weights=weights)

            vtmp.append(vloss.item())
            xhi_fe, tb_fe, ts_fe, lf_fe, tau_fe = get_FE(true,
                                                    pred,
                                                    floor=1e-2, plot = True if step == 0 else False)
            pbar.set_postfix(vloss=np.mean(vtmp))

    return torch.mean(vtmp)

def lr_schedule(optimizer, epoch_vlosses, scheduler, plateau=0, patience=8):
    this_loss = epoch_vlosses[-1]
    prev_best_loss = np.min(epoch_vlosses[:-1])
    lr = optimizer.param_groups[0]["lr"]
    if prev_best_loss - this_loss > 0:
        plateau = 0
    if prev_best_loss - this_loss < 0:
        plateau += 1
    if plateau > patience and lr > 1e-10:
            scheduler.step()
            neg = -1*patience

    return scheduler, plateau

