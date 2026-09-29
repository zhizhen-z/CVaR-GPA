import tensorflow as tf
import numpy as np
from lib.construct_NN import spectral_normalization
from lib.losses import divergence_mb, gradient_penalty, wasserstein1_mb

def sgd(lr_phi, NN_par, W, dW, b, db, nu=None, dnu=None, descent=True, calc_dW_norm=False):
    dW_norm = 0
    if descent == False:
        lr = - lr_phi
    else:
        lr = lr_phi
    
    for l in range(len(W)):
        if calc_dW_norm == True:
            dW_norm = max(dW_norm, tf.norm(dW[l]))
        
        W[l].assign(W[l] - lr * dW[l])
        if NN_par['constraint'] == 'hard' and NN_par['L'] != None: # spectral normalization
            spectral_normalization(W[l], NN_par['L']**(1/len(W)))
        if db[l] != None: # fnn
            b[l].assign(b[l] - lr * db[l])
            
    if dnu != None:
        nu.assign(nu - lr_phi*dnu)
            
    return dW_norm

def adam_update(grad, iter, m, v, beta1=0.9, beta2=0.999, eps=1e-8):
    grad = grad.numpy()
    if iter == 0:
        m, v = np.zeros_like(grad), np.zeros_like(grad)
        
    # reweight m, v
    m = beta1*m + (1-beta1)*grad
    v = beta2*v + (1-beta2)*grad**2
    
    m_hat = m/(1-beta1**(iter+1))
    v_hat = v/(1-beta2**(iter+1))
    
    grad_hat = m_hat/(np.sqrt(v_hat)+eps)
    
    return grad_hat, m, v
 
def adam(lr_phi, NN_par, iter, W, dW, m_W, v_W, b, db, m_b, v_b, nu=None, dnu=None, m_nu=None, v_nu=None, descent=True, calc_dW_norm=False):
    dW_norm = 0
    if descent == False:
        lr = - lr_phi
    else:
        lr = lr_phi
    
    for l in range(len(W)):
        if calc_dW_norm == True:
            dW_norm = max(dW_norm, tf.norm(dW[l]))
        dW_hat, m_W[l], v_W[l] = adam_update(dW[l], iter, m_W[l], v_W[l])
        W[l].assign(W[l] - lr * dW_hat)
        
        if NN_par['constraint'] == 'hard' and NN_par['L'] != None: # spectral normalization
            spectral_normalization(W[l], NN_par['L']**(1/len(W)))
            
        if db[l] != None: # fnn
            db_hat, m_b[l], v_b[l] = adam_update(db[l], iter, m_b[l], v_b[l])
            b[l].assign(b[l] - lr * db_hat)
            
    if dnu != None:
        dnu_hat, m_nu, v_nu = adam_update(dnu, iter, m_nu, v_nu)
        nu.assign(nu - lr * dnu_hat)
        
    if type(nu) == type(None):
        return dW_norm, m_W, v_W, m_b, v_b
    else:
        return dW_norm, m_W, v_W, m_b, v_b, m_nu, v_nu
   

# -------------------------------------------
def train_disc(parameters, phi, P, Q, lr_phi, epochs_phi, loss_par, NN_par, data_par, optimizer, print_vals=False, track_loss=False):
    
    W = parameters['W']
    b = parameters['b']
    nu = parameters['nu']
    
    if track_loss == True:
        track_losses = []
        
    
    #current_loss = -1e+6
    for in_it in range(1, epochs_phi+1): # Loop for training NN discriminator phi*
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch([W, b,])# nu])
            
            penalty = gradient_penalty(phi, P, Q, W, b, NN_par, data_par, loss_par['lamda'])
            loss = divergence_mb(phi, nu, P, Q, W, b, NN_par, loss_par, data_par) + penalty
            #print(loss.numpy())
            if track_loss == True:
                track_losses.append(loss.numpy())
        
        dW, db, dnu = tape.gradient(loss, [W,b,nu])
                
        if optimizer == 'sgd':
            dW_norm = sgd(lr_phi, NN_par, W, dW, b, db, nu, dnu, descent=False, calc_dW_norm=print_vals) # update phi(W,b)
        elif optimizer == 'adam':
            if in_it == 1:
                m_W, v_W, m_b, v_b, m_nu, v_nu = [0]*len(W), [0]*len(W), [0]*len(W), [0]*len(W), 0, 0
            dW_norm, m_W, v_W, m_b, v_b, m_nu, v_nu = adam(lr_phi, NN_par, in_it, W, dW, m_W, v_W, b, db, m_b, v_b, nu, dnu, m_nu, v_nu, descent=False, calc_dW_norm=print_vals) # update phi(W,b)
       
    current_loss = loss.numpy()
            
    parameters['W'] = W
    parameters['b'] = b
    parameters['nu'] = nu

    if print_vals == True:
        if track_loss == True:
            return parameters, current_loss, dW_norm, track_losses
        else:
            return parameters, current_loss, dW_norm
    else:
        return parameters
        
        
def train_wasserstein1(parameters, phi, P, Q, lr_phi, epochs_phi, NN_par, data_par, optimizer, print_vals=False):    
    W = parameters['W']
    b = parameters['b']
    
    #current_loss = -1e+6
    for in_it in range(1, epochs_phi+1): # Loop for training NN discriminator phi*
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch([W, b])

            loss = wasserstein1_mb(phi, P, Q, W, b, NN_par, data_par)
            #print(loss.numpy())
        
        dW, db = tape.gradient(loss, [W,b])
                
        if optimizer == 'sgd':
            dW_norm = sgd(lr_phi, NN_par, W, dW, b, db, descent=False, calc_dW_norm=print_vals) # update phi(W,b)
        elif optimizer == 'adam':
            if in_it == 1:
                m_W, v_W, m_b, v_b = [0]*len(W), [0]*len(W), [0]*len(W), [0]*len(W)
            dW_norm, m_W, v_W, m_b, v_b = adam(lr_phi, NN_par, in_it, W, dW, m_W, v_W, b, db, m_b, v_b, descent=False, calc_dW_norm=print_vals) # update phi(W,b)
       
    current_loss = loss.numpy()
            
    parameters['W'] = W
    parameters['b'] = b

    if print_vals == True:
        return parameters, current_loss, dW_norm
    else:
        return parameters

