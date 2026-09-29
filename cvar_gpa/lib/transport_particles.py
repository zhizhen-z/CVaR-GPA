import tensorflow as tf
import numpy as np
from lib.train_NN import train_disc
from lib.losses import f_star

# This module consists of
# - vectorfield calculator using the gradient of loss first variation
# - ode solver for the particle update

# ----------------------------------------------------------
# obtain vectorfield from taking gradient of first_variation
# ----------------------------------------------------------
@tf.function
def first_variation(phi, P, W, b, nu, NN_par, loss_par, data_par):
    P_label = data_par['P_label']
    g = phi(P, P_label, W, b, NN_par)
    if loss_par['reverse'] == False: # D(P||Q)
        return g
    else: # D(Q||P)
        if loss_par['formulation'] == 'DV':
            g_max = tf.reduce_max(g)
            logsumexp = tf.math.log(tf.math.reduce_mean(tf.math.exp(g - g_max))) + g_max
            return -(logsumexp + tf.math.exp(g - logsumexp)-1)
        else:
            return -(f_star(g-nu, loss_par) + nu)
       
            
@tf.function
def calc_vectorfield(phi, P, parameters, NN_par, loss_par, data_par):
    W = parameters['W']
    b = parameters['b']
    nu = parameters['nu']
    
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(P)
        loss_first_variation = first_variation(phi, P, W, b, nu, NN_par,loss_par, data_par)
    dP = tape.gradient(loss_first_variation, P)
    return dP
    
# ----------------------------------------------------------
# solve ode
# ----------------------------------------------------------
# 1 step explicit
def forward_euler(x, deltat, f_xt):
    vf = f_xt[-1]
    x.assign(x - deltat*vf)
    return x, [], vf
    
# 2 step explicit
def AB2(x, deltat, f_xt):
    if len(f_xt) < 2:
        vf = f_xt[-1]
    else:
        vf = (-1*f_xt[0] + 3*f_xt[1])/2
        f_xt.pop(0)
    x.assign(x - deltat*vf)
    return x, f_xt, vf
    
# 3 step explicit
def AB3(x, deltat, f_xt):
    if len(f_xt) < 3:
        vf = f_xt[-1]
    else:
        vf = (5*f_xt[0] - 16*f_xt[1] + 23*f_xt[2])/12
        f_xt.pop(0)
    x.assign(x - deltat*vf)
    return x, f_xt, vf
    
# 4 step explicit
def AB4(x, deltat, f_xt):
    if len(f_xt) < 4:
        vf = f_xt[-1]
    else:
        vf = (-9*f_xt[0] + 37*f_xt[1] - 59*f_xt[2] + 55*f_xt[3])/24
        f_xt.pop(0)
    x.assign(x - deltat*vf)
    return x, f_xt, vf

# 5 step explicit
def AB5(x, deltat, f_xt):
    if len(f_xt) < 5:
        vf = f_xt[-1]
    else:
        vf = (251*f_xt[0] - 1274*f_xt[1] + 2616*f_xt[2] - 2774*f_xt[3] + 1901*f_xt[4])/720
        f_xt.pop(0)
    x.assign(x - deltat*vf)
    return x, f_xt, vf
    
    
# 1 step AB-predictor AM-corrector
def ABM1(x, deltat, f_xt, aux_params):
    parameters = aux_params['parameters']
    phi = aux_params['phi']
    Q = aux_params['Q']
    lr_phi = aux_params['lr_phi']
    epochs_phi = aux_params['epochs_phi']
    loss_par = aux_params['loss_par']
    NN_par = aux_params['NN_par']
    data_par = aux_params['data_par']
    optimizer = aux_params['optimizer']
    
    x_tilde = x - deltat*f_xt[0]
    f_xt.pop(0)
    parameters = train_disc(parameters, phi, x_tilde, Q, lr_phi, epochs_phi, loss_par, NN_par, data_par, optimizer)
    f_x_tilde_t = calc_vectorfield(phi, x_tilde, parameters, NN_par, loss_par, data_par)
    
    vf = f_x_tilde_t
    x.assign(x - deltat*vf)
    return x, f_xt, vf
    
# 2 step FE-predictor  trapezoidal-corrector
def Heun(x, deltat, f_xt, aux_params):
    parameters = aux_params['parameters']
    phi = aux_params['phi']
    Q = aux_params['Q']
    lr_phi = aux_params['lr_phi']
    epochs_phi = aux_params['epochs_phi']
    loss_par = aux_params['loss_par']
    NN_par = aux_params['NN_par']
    data_par = aux_params['data_par']
    optimizer = aux_params['optimizer']
    
    x_tilde = x - deltat*f_xt[-1]
    parameters = train_disc(parameters, phi, x_tilde, Q, lr_phi, epochs_phi, loss_par, NN_par, data_par, optimizer)
    f_x_tilde_t = calc_vectorfield(phi, x_tilde, parameters, NN_par, loss_par, data_par)
    
    vf = (f_xt[0] + f_x_tilde_t)/2
    f_xt.pop(0)
    x.assign(x - deltat*vf)
    return x, f_xt, vf
    
# 2 step AB-predictor-evaluation AM-corrector-evaluation
def ABM2(x, deltat, f_xt, aux_params):
    parameters = aux_params['parameters']
    phi = aux_params['phi']
    Q = aux_params['Q']
    lr_phi = aux_params['lr_phi']
    epochs_phi = aux_params['epochs_phi']
    loss_par = aux_params['loss_par']
    NN_par = aux_params['NN_par']
    data_par = aux_params['data_par']
    optimizer = aux_params['optimizer']
    
    if len(f_xt) < 2:
        vf = f_xt[-1]
    else:
        x_tilde = x - deltat * (-f_xt[0] + 3*f_xt[1])/2 # AB
        f_xt.pop(0)
        parameters = train_disc(parameters, phi, x_tilde, Q, lr_phi, epochs_phi, loss_par, NN_par, data_par, optimizer)
        f_x_tilde_t = calc_vectorfield(phi, x_tilde, parameters, NN_par, loss_par, data_par)
        
        vf = (f_xt[0] + f_x_tilde_t)/2 # AM
    
    x.assign(x - deltat*vf)
    return x, f_xt, vf
    
# 3 step AB-predictor-evaluation AM-corrector-evaluation
def ABM3(x, deltat, f_xt, aux_params):
    parameters = aux_params['parameters']
    phi = aux_params['phi']
    Q = aux_params['Q']
    lr_phi = aux_params['lr_phi']
    epochs_phi = aux_params['epochs_phi']
    loss_par = aux_params['loss_par']
    NN_par = aux_params['NN_par']
    data_par = aux_params['data_par']
    optimizer = aux_params['optimizer']
    
    if len(f_xt) < 3:
        vf = f_xt[-1]
    else:
        x_tilde = x - deltat * (5*f_xt[0] -16*f_xt[1] + 23*f_xt[2])/12 # AB
        f_xt.pop(0)
        parameters = train_disc(parameters, phi, x_tilde, Q, lr_phi, epochs_phi, loss_par, NN_par, data_par, optimizer)
        f_x_tilde_t = calc_vectorfield(phi, x_tilde, parameters, NN_par, loss_par, data_par)
        vf = (-1*f_xt[0] + 8*f_xt[1] + 5*f_x_tilde_t)/12 # AM
        
    x.assign(x - deltat*vf)
    return x, f_xt, vf
    
# 4 step AB-predictor-evaluation AM-corrector-evaluation
def ABM4(x, deltat, f_xt, aux_params):
    parameters = aux_params['parameters']
    phi = aux_params['phi']
    Q = aux_params['Q']
    lr_phi = aux_params['lr_phi']
    epochs_phi = aux_params['epochs_phi']
    loss_par = aux_params['loss_par']
    NN_par = aux_params['NN_par']
    data_par = aux_params['data_par']
    optimizer = aux_params['optimizer']
    
    if len(f_xt) < 4:
        vf = f_xt[-1]
    else:
        x_tilde = x - deltat * (-9*f_xt[0] + 37*f_xt[1] - 59*f_xt[2] + 55*f_xt[3])/24 # AB
        f_xt.pop(0)
        parameters = train_disc(parameters, phi, x_tilde, Q, lr_phi, epochs_phi, loss_par, NN_par, data_par, optimizer)
        f_x_tilde_t = calc_vectorfield(phi, x_tilde, parameters, NN_par, loss_par, data_par)
        
        vf = (1*f_xt[0] - 5/24*f_xt[1] + 19*f_xt[2] + 9*f_x_tilde_t)/24 # AM
        
    x.assign(x - deltat*vf)
    return x, f_xt, vf

# 5 step AB-predictor-evaluation AM-corrector-evaluation
def ABM5(x, deltat, f_xt, aux_params):
    parameters = aux_params['parameters']
    phi = aux_params['phi']
    Q = aux_params['Q']
    lr_phi = aux_params['lr_phi']
    epochs_phi = aux_params['epochs_phi']
    loss_par = aux_params['loss_par']
    NN_par = aux_params['NN_par']
    data_par = aux_params['data_par']
    optimizer = aux_params['optimizer']
    
    if len(f_xt) < 5:
        vf = f_xt[-1]
    else:
        x_tilde = x - deltat * (251*f_xt[0] - 1274*f_xt[1] + 2616*f_xt[2] - 2774*f_xt[3] + 1901*f_xt[4])/720 # AB
        f_xt.pop(0)
        parameters = train_disc(parameters, phi, x_tilde, Q, lr_phi, epochs_phi, loss_par, NN_par, data_par, optimizer)
        f_x_tilde_t = calc_vectorfield(phi, x_tilde, parameters, NN_par, loss_par, data_par)
        
        vf = (-19*f_xt[0] + 106*f_xt[1] - 264*f_xt[2] + 646*f_xt[3] + 251*f_x_tilde_t)/720
    x.assign(x - deltat*vf)
    return x, f_xt, vf
    
def RK4(x, deltat, f_xt, aux_params):
    parameters = aux_params['parameters']
    phi = aux_params['phi']
    Q = aux_params['Q']
    lr_phi = aux_params['lr_phi']
    epochs_phi = aux_params['epochs_phi']
    loss_par = aux_params['loss_par']
    NN_par = aux_params['NN_par']
    data_par = aux_params['data_par']
    optimizer = aux_params['optimizer']
    
    # y1 = x # f_xt[0]
    
    y2 = x - deltat/2*f_xt[-1]
    parameters, current_loss1, dW_norm1 = train_disc(parameters, phi, y2, Q, lr_phi, epochs_phi, loss_par, NN_par, data_par, optimizer, print_vals=True)
    f_xt.append( calc_vectorfield(phi, y2, parameters, NN_par, loss_par, data_par) ) # f_xt[1]
    
    y3 = x - deltat/2*f_xt[-1]
    parameters, current_loss2, dW_norm2 = train_disc(parameters, phi, y3, Q, lr_phi, epochs_phi, loss_par, NN_par, data_par, optimizer, print_vals=True)
    f_xt.append( calc_vectorfield(phi, y3, parameters, NN_par, loss_par, data_par) ) # f_xt[2]
    
    y4 = x - deltat*f_xt[-1]
    parameters, current_loss3, dW_norm3 = train_disc(parameters, phi, y4, Q, lr_phi, epochs_phi, loss_par, NN_par, data_par, optimizer, print_vals=True)
    f_xt.append( calc_vectorfield(phi, y4, parameters, NN_par, loss_par, data_par) ) # f_xt[3]
    
    #print("Losses: ", current_loss1, current_loss2, current_loss3)
    #print("dW_norms: ",dW_norm1, dW_norm2, dW_norm3)
    
    vf = (f_xt[0] + 2*f_xt[1] + 2*f_xt[2] + f_xt[3])/6
    x.assign(x - deltat*vf)
    f_xt = []
    return x, f_xt, vf
    
def ode45(x, deltat, f_xt, aux_params, tol=1e-5):
    parameters = aux_params['parameters']
    phi = aux_params['phi']
    Q = aux_params['Q']
    lr_phi = aux_params['lr_phi']
    epochs_phi = aux_params['epochs_phi']
    loss_par = aux_params['loss_par']
    NN_par = aux_params['NN_par']
    data_par = aux_params['data_par']
    optimizer = aux_params['optimizer']
    
    # y1 = x # f_xt[0]
    
    y2 = x - deltat/5*f_xt[0]
    parameters = train_disc(parameters, phi, y2, Q, lr_phi, epochs_phi, loss_par, NN_par, data_par, optimizer)
    f_xt.append( calc_vectorfield(phi, y2, parameters, NN_par, loss_par, data_par) ) # f_xt[1]
    
    y3 = x - deltat*(3/40*f_xt[0] + 9/40*f_xt[1])
    y3_tilde = x - deltat*3/10*f_xt[0]
    parameters = train_disc(parameters, phi, y3, Q, lr_phi, epochs_phi, loss_par, NN_par, data_par, optimizer)
    f_xt.append( calc_vectorfield(phi, y3, parameters, NN_par, loss_par, data_par) ) # f_xt[2]
    
    y4 = x - deltat*(44/45*f_xt[0] - 56/15*f_xt[1] + 32/9*f_xt[2])
    y4_tilde = x - deltat*4/5*f_xt[0]
    parameters = train_disc(parameters, phi, y4, Q, lr_phi, epochs_phi, loss_par, NN_par, data_par, optimizer)
    f_xt.append( calc_vectorfield(phi, y4, parameters, NN_par, loss_par, data_par) ) # f_xt[3]
    
    y5 = x - deltat*(19372/6561*f_xt[0] - 25360/2187*f_xt[1] + 64448/6561*f_xt[2] - 212/729*f_xt[3])
    y5_tilde = x - deltat*8/9*f_xt[0]
    parameters = train_disc(parameters, phi, y5, Q, lr_phi, epochs_phi, loss_par, NN_par, data_par, optimizer)
    f_xt.append( calc_vectorfield(phi, y5, parameters, NN_par, loss_par, data_par) ) # f_xt[4]
    
    y6 = x - deltat*(9017/3168*f_xt[0] - 355/33*f_xt[1] + 46732/5247*f_xt[2] + 49/176*f_xt[3] - 5103/18656*f_xt[4])
    y6_tilde = x - deltat*f_xt[0]
    parameters = train_disc(parameters, phi, y6, Q, lr_phi, epochs_phi, loss_par, NN_par, data_par, optimizer)
    f_xt.append( calc_vectorfield(phi, y6, parameters, NN_par, loss_par, data_par) ) # f_xt[5]
    
    a1, a2, a3, a4, a5, a6, a7 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0
    y7 = x - deltat*(a1*f_xt[0] + a2*f_xt[1] + a3*f_xt[2] + a4*f_xt[3] + a5*f_xt[4] + a6*f_xt[5])
    y7_tilde = x - deltat*f_xt[0]
    parameters = train_disc(parameters, phi, y7 , Q, lr_phi, epochs_phi, loss_par, NN_par, data_par, optimizer)
    f_xt.append( calc_vectorfield(phi, y7, parameters, NN_par, loss_par, data_par) ) # f_xt[6]
    
    # adjust stepsize by size of error
    if tf.rank(x) == 2:
        error = tf.math.sqrt(tf.math.reduce_sum(tf.math.square((a1-5179/57600)*f_xt[0] + (a3-7571/16695)*f_xt[2] + (a4-393/640)*f_xt[3] + (a5-92097/339200)*f_xt[4] + (a6-187/2100)*f_xt[5] + (a7-1/40)*f_xt[6]), axis = 1, keepdims=True))
    elif tf.rank(x) == 4:
        error = tf.math.sqrt(tf.math.reduce_sum(tf.math.square((a1-5179/57600)*f_xt[0] + (a3-7571/16695)*f_xt[2] + (a4-393/640)*f_xt[3] + (a5-92097/339200)*f_xt[4] + (a6-187/2100)*f_xt[5] + (a7-1/40)*f_xt[6]), axis = [1, 2, 3], keepdims=True))
    
    #print(error< tol)
    delta = 0.84*(tol/error)**(1/5)
    #
    #print(tol)
    #print(error)
    #vf = tf.cast(error < tol, tf.float32) * (a1*f_xt[0] + a3*f_xt[2] + a4*f_xt[3] + a5*f_xt[4] + a6*f_xt[5])
    vf = (a1*f_xt[0] + a3*f_xt[2] + a4*f_xt[3] + a5*f_xt[4] + a6*f_xt[5])
    x.assign(x - deltat*vf)
    
    #deltat = tf.cast(delta <= 0.1, tf.float32) * deltat/10 + tf.cast(delta >= 40, tf.float32) * deltat*4 + tf.cast(tf.logical_and(delta>0.1, delta<40), tf.float32)* delta*deltat
    
    f_xt = []
    return x, f_xt, vf, deltat
    
    
def ode113(x, deltat, f_xt, aux_params, tol=1e-5):
    parameters = aux_params['parameters']
    phi = aux_params['phi']
    Q = aux_params['Q']
    lr_phi = aux_params['lr_phi']
    epochs_phi = aux_params['epochs_phi']
    loss_par = aux_params['loss_par']
    NN_par = aux_params['NN_par']
    data_par = aux_params['data_par']
    optimizer = aux_params['optimizer']
    
    
    
    
# learning rate decay schedule
def decay_learning_rate(lr, schedule, param_dict):
    if schedule == 'rational':
        maxiter = param_dict['epochs']
        it = param_dict['epoch']
        lr.assign(lr*(1-it/maxiter))
    elif schedule == 'step':
        KE_P = param_dict['KE_P']
        KE_P_lim = 1e-4
        KE_P_alpha_lim = 0.001
        if KE_P <= KE_P_lim + 2*KE_P_alpha_lim:
            lr.assign(lr/2)
    return lr


# ------------------------------------------------------
# solver of ode of the form dx/dt=f(x,t)
def solve_ode(x, deltat, f_xt, ode_solver, aux_params):
    if ode_solver == "forward_euler":
        return forward_euler(x, deltat, f_xt)
    elif ode_solver == "AB2":
        return AB2(x, deltat, f_xt)
    elif ode_solver == "AB3":
        return AB3(x, deltat, f_xt)
    elif ode_solver == "AB4":
        return AB4(x, deltat, f_xt)
    elif ode_solver == "AB5":
        return AB5(x, deltat, f_xt)
        
    elif ode_solver == "ABM1":
        return ABM1(x, deltat, f_xt, aux_params)
    elif ode_solver == "Heun":
        return Heun(x, deltat, f_xt, aux_params)
    elif ode_solver == "ABM2":
        return ABM2(x, deltat, f_xt, aux_params)
    elif ode_solver == "ABM3":
        return ABM3(x, deltat, f_xt, aux_params)
    elif ode_solver == "ABM4":
        return ABM4(x, deltat, f_xt, aux_params)
    elif ode_solver == "ABM5":
        return ABM5(x, deltat, f_xt, aux_params)
        
    elif ode_solver == "RK4":
        return RK4(x, deltat, f_xt, aux_params)
    elif ode_solver == "ode45":
        return ode45(x, deltat, f_xt, aux_params)
        

