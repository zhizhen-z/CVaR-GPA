# Metrics to monitor convergence
# 1. kinetic energy
#    calc_ke(velocity_field, samplesize)
# 2. average particle speed
#    calc_grad_phi(velocity_field)
# 3. sinkhorn divergence
#    calc_sinkhorn(P, Q, reg=0.2)

import numpy as np

def calc_ke(dP_dt, N_samples_P):
    return np.linalg.norm(dP_dt)**2/N_samples_P/2
    
def calc_grad_phi(dP_dt):
    return np.mean(np.linalg.norm(dP_dt, axis=1))
    
def calc_sinkhorn(P, Q, reg=0.2):
    from geomloss import SamplesLoss
    import torch
    N_P = P.shape[0]
    N_Q = Q.shape[0]
    
    X = torch.from_numpy(P).type(torch.float32)
    Y = torch.from_numpy(Q).type(torch.float32)
    
    return SamplesLoss(loss='sinkhorn', p=2)(X,Y).numpy()
