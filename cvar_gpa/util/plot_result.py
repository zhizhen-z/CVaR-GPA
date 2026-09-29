#!/usr/bin/env python3
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sys import platform, argv

def show_or_close_figure(show=False):
    if "linux" in platform or show == False:
        plt.clf()
        plt.close()
    else:
        plt.show() 

def load_pickle(filename):
    with open(filename, "rb") as fr:
        param, result = pickle.load(fr)
    return param, result
    
# -----------------------------------------
# General drawing functions
# -----------------------------------------
def handle_dt(lr_P=None, lr_Ps=None, physical_time=True):
# lr_P (old) : lr_P scalar value for the entire epochs/the first epoch
# lr_P (new) : [lr_P]*N_moving_particles for the first epoch
# lr_Ps (old) : [lr_P]*epochs for all different epochs
# lr_Ps (new) : [[lr_P]*N_moving_particles]*epochs for all different epochs
    if physical_time == True:
        try:
            dt = lr_P[0]
        except:
            dt = lr_P
        if lr_Ps != None:
            try:
                dt = lr_Ps
                dt = [x[0] for x in dt]
            except:
                dt = lr_Ps
    else:
        dt = 1
    return dt
    
def calculate_time_steps(dt, iter_nos, physical_time=True):
# calculate real_time / iteration count
    if physical_time == True:
        iter_nos_new = []
        if type(dt) == list: # decaying or varying dt
            for iter_no in iter_nos:
                if iter_no == 0:
                    iter_nos_new.append(0)
                else:
                    iter_nos_new.append(sum(dt[:iter_no]))
        else: # constant dt
            for iter_no in iter_nos:
                iter_nos_new.append(dt*iter_no)
        iter_nos = iter_nos_new
    return iter_nos
                
def proj_and_sample(snapshot, proj_axes=None, pick_samples=None):
# return chosen samples and chosen axes
    if type(snapshot) != np.ndarray:
        return None
    snapshot = np.array(snapshot)
    if proj_axes != None:
        snapshot = snapshot[:,proj_axes]
    if pick_samples != None:
        snapshot = snapshot[pick_samples,:]
    return snapshot 
    
def set_axis_lim(samples, mask=1, lb=None, ub=None):
# set axis limit using 1st-99th percentile to show tails without extreme outlier blow-up
# samples: list of arrays
    all_vals = np.concatenate([np.asarray(s).ravel() for s in samples])
    s_min = np.percentile(all_vals, 1) - mask
    s_max = np.percentile(all_vals, 99) + mask

    if lb != None:
        s_min = max(s_min, lb)
    if ub != None:
        s_max = min(s_max, ub)

    return (s_min, s_max)
    
    
    
# -----------------------------------------
# Additional features for Trajectories plot
# -----------------------------------------   
def add_quantile_contour(ax, dataset, r_param):
# gaussian) 50% quantile: solid line, 90% quantile: dashed line
# heavy-tailed) 25% quantile: solid line, 50% quantile: dashed line
# only applicable to dataset = gaussian/student_t/Stretched_exponential
    if 'gaussian' in dataset:
        r1 = 0.6745*r_param   # 50%
        r2 = 1.644854*r_param # 90%
    elif 'student_t' in dataset:
        if r_param == 0.5:
            r1 = 0.51856      # 25%
            r2 = 1.55377      # 50%
        elif r_param == 5:
            r1 = 0.33672      # 25%
            r2 = 0.72669      # 50%
    elif dataset == 'Stretched_exponential':
        if r_param == 0.7:
            r1 = 0.55         # 25%
            r2 = 1.2          # 50%
        elif r_param == 0.4:
            r1 = 2.8          # 25%
            r2 = 7            # 50% 
    else:
        return -1# irrelevant to plot quantiles
    
    if 'Mixture' not in dataset:
        centers=[(0, 0)]
    elif 'Mixture_of_gaussians' in dataset:
        centers=[(0, 0), (4,0), (0,4), (4,4)]
        if 'Mixture_of_gaussians2' in dataset:
            centers=[(0, 0), (8,0), (0,8), (8,8)]
    elif dataset == 'Mixture_of_student_t':
        centers=[(-10, -10), (10,-10), (-10,10), (10,10)]
    elif dataset == 'Mixture_of_student_t_submnfld':
        centers=[(-10, -10), (10,-10), (-10,10), (10,10)]
        
    circles1, circles2 = [], []
    for c in centers:
        circles1.append(plt.Circle(tuple(c), r1, color='k', linestyle='-', fill=False))
        circles2.append(plt.Circle(tuple(c), r2, color='k', linestyle='--', fill=False))
        ax.add_patch(circles1[-1])
        ax.add_patch(circles2[-1])
    return 0
        
def plot_arrow(ax, x, dx, scale=1):
# plot vectorfield of chosen samples
    if type(dx) != np.ndarray:
        return
    kwargs = {'color':'lawngreen'}
    ax.arrow(x=x[0], y=x[1], dx=-dx[0]*scale, dy=-dx[1]*scale, width=0.1, overhang=.5, **kwargs)
    

# -----------------------------------------
# Individual figures
# -----------------------------------------    

# Scatter plot for given data
def plot_initial_data(X_=None, Y_=None, proj_axes = [0,1], x_lim = [None,None],y_lim = [None,None], filename=None, show = False):
# plot X_ : target data, Y_ : initial data in 2D projected plane
    if 'epochs' not in filename:
        param, result = load_pickle(filename)
        X_ = param['X_']
        try: 
            Y_ = param['Y_']
        except:
            pass
            
    X_ = proj_and_sample(X_, proj_axes)
    Y_ = proj_and_sample(Y_, proj_axes)
    
    plt.scatter(X_[:, 0], X_[:, 1], label="target X")
    plt.scatter(Y_[:, 0], Y_[:, 1], label="initial Y")
    
    xlims = set_axis_lim([X_[:,0], Y_[:,0]], mask=1, lb=x_lim[0], ub=x_lim[1])
    ylims = set_axis_lim([X_[:,1], Y_[:,1]], mask=1, lb=y_lim[0], ub=y_lim[1])
    plt.xlim(xlims)
    plt.ylim(ylims)
    
    plt.legend()
    plt.tight_layout()
    
    f = filename.split('.pickle')
    plt.savefig(f[0]+"-initial_data.png",bbox_inches='tight')
    
    show_or_close_figure(show)
    
# Scatter plot for target vs output
def plot_output_target(X_=None, trajectories=None, proj_axes = [0,1], x_lim = [None,None],y_lim = [None,None], filename=None, show = False):
# plot X_ : target data, Y_ : initial data in 2D projected plane
    if 'epochs' not in filename:
        param, result = load_pickle(filename)
        X_ = param['X_']
        try:
            Y_ = result['trajectories'][-1]
        except:
            pass
            
    X_ = proj_and_sample(X_, proj_axes)
    Y_ = proj_and_sample(Y_, proj_axes)
    
    plt.scatter(X_[:, 0], X_[:, 1], label="target", s=5, alpha=0.7)
    plt.scatter(Y_[:, 0], Y_[:, 1], label="output", s=5, alpha=0.7)
    
    if x_lim[0] is not None and x_lim[1] is not None:
        xlims = (x_lim[0], x_lim[1])
    else:
        xlims = set_axis_lim([X_[:,0], Y_[:,0]], mask=1, lb=x_lim[0], ub=x_lim[1])
    if y_lim[0] is not None and y_lim[1] is not None:
        ylims = (y_lim[0], y_lim[1])
    else:
        ylims = set_axis_lim([X_[:,1], Y_[:,1]], mask=1, lb=y_lim[0], ub=y_lim[1])
    plt.xlim(xlims)
    plt.ylim(ylims)

    plt.legend()
    plt.tight_layout()

    f = filename.split('.pickle')
    plt.savefig(f[0]+"-output_target.png",bbox_inches='tight')
    
    show_or_close_figure(show)
 
# Scatter plots for time trajectories
def plot_trajectories(trajectories=None, dt=None, X_=None, Y_=None, dataset = None, r_param=None, vectorfields = [], proj_axes = [0,1], pick_samples =None, epochs = 0, iter_nos = None, physical_time=True, save_iter = 1, track_velocity=False, arrow_scale = 1, iscolor=False, quantile=True, exp_alias_ = None, x_lim = [None,None],y_lim = [None,None], filename = None, show = False):
# plot trajectories of one file
# exp_alias_: one of the keys in param dictionary as a string '...' (to specify title)
    print('plot_trajectories')
    # make frames
    if iter_nos == None:
        n_figs = 4
    else:
        n_figs = len(iter_nos)
    f, axs = plt.subplots(nrows=1, ncols=n_figs, figsize=(15, 3.5))  
    
    # load pickled data
    if 'epochs' not in filename:
        param, result = load_pickle(filename)
        if epochs == 0:
            epochs = param['epochs']
        if epochs > 100:
            save_iter = param['save_iter']
        X_ = param['X_']
        try:
            Y_ = param['Y']
        except:
            pass
        trajectories = [Y_] + result['trajectories']
        try:
            vectorfields = result['vectorfields']
        except:
            pass
            
        if 'lr_Ps' in param.keys():
            dt = handle_dt(param['lr_P'], lr_Ps=param['lr_Ps'], physical_time=physical_time)
        else:
            dt = handle_dt(param['lr_P'], lr_Ps=None, physical_time=physical_time)
        

        if quantile == True:
            dataset = param['dataset']
            if 'gaussian' in dataset:
                 r_param = param['sigma_Q']
            elif 'student_t' in dataset:
                r_param = param['nu']
            elif dataset == 'Extension_of_gaussian':
                r_param = param['a']   
                       
    else:
        trajectories = [Y_] + trajectories
        dt = handle_dt(lr_Ps=dt, physical_time=physical_time)
 
    
    # pre-processing
    X_ = proj_and_sample(X_, proj_axes, pick_samples)
    trajectories = [proj_and_sample(x, proj_axes, pick_samples) for x in trajectories] 
    
    if iter_nos == None:
        iter_nos = [int(epochs*x/(n_figs-1)) for x in range(n_figs)]
    if type(trajectories[iter_nos[0]]) != np.ndarray: # no initial samples
        iter_nos[0] = save_iter
      
    trajectories = [trajectories[int(i/save_iter)] for i in iter_nos]
    time_steps = calculate_time_steps(dt, iter_nos, physical_time=physical_time)  
    
    if track_velocity or iscolor:
        iter_nos_ = iter_nos
        if iter_nos_[-1] == epochs:
            iter_nos_[-1] = epochs-1
        if vectorfields != []:
            vectorfields = [vectorfields[int(i/save_iter)] for i in iter_nos_]
    
    if iscolor == True:
        cv_max, cv_min = 0, 1e+16
        for vf in vectorfields:
            cv_max = max(cv_max, max(np.linalg.norm(vf, axis=1)))
            cv_min = min(cv_min, min(np.linalg.norm(vf, axis=1)))  
            c_map = [np.linalg.norm(vectorfields[i] , axis=1) for i in range(len(vectorfields))]  
    if track_velocity == True:
        dP = [proj_and_sample(x, proj_axes, pick_samples) for x in vectorfields]
    
    # draw plots
    for i, ax in enumerate(axs):
        if physical_time == True:
            ax.set_title('T=%.3f'% time_steps[i])
        else:
            ax.set_title('T=%d' % iter_nos[i])
        # plot quantile contour or target data
        if quantile == True:
            flag = add_quantile_contour(ax, dataset, r_param)
            if flag == -1:
                ax.scatter(X_[:,0], X_[:,1], s=5)
        else:
            ax.scatter(X_[:,0], X_[:,1], s=5)
        
        # color/uncolor trajectories by speed of particles        
        if iscolor == True:
            ax.scatter(trajectories[i][ :, 0], trajectories[i][ :, 1], c=abs(c_map[i]), cmap='seismic', s=5, vmin=cv_min, vmax=cv_max)
        else:
            ax.scatter(trajectories[i][ :, 0], trajectories[i][ :, 1], s=5) 
        
        xlims = set_axis_lim([X_[:,0], trajectories[i][:,0]], mask=1, lb=x_lim[0], ub=x_lim[1])
        ylims = set_axis_lim([X_[:,1], trajectories[i][:,1]], mask=1, lb=y_lim[0], ub=y_lim[1])
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)   
        
        # draw arrow for particle speed
        if track_velocity == True:
            n_Samples_P = dP[i].shape[0]
            for s in range(min(3, n_Samples_P)):
                plot_arrow(ax, trajectories[i][s], dP[i][s], arrow_scale)
    plt.tight_layout()
    
    f = filename.split('.pickle')
    plt.savefig(f[0]+"-trajectories.png",bbox_inches='tight')
    
    show_or_close_figure(show)
    
def trajectories_to_animation(x_lim, y_lim, trajectories=None, N_samples_P=None, dt=None, physical_time=True, epochs=0, quantile = True, dataset=None, r_param = None, save_gif=True, filename = None, show = False):
    import matplotlib.animation as animation
    
    # load data
    if 'epochs' not in filename:
        param, result = load_pickle(filename)
        trajectories = result['trajectories']
        N_samples_P = param['N_samples_P']
        
        if epochs==0:
            epochs = param['epochs']
        else:
            param['epochs'] = epochs
        save_iter = param['save_iter']
        
        if 'lr_Ps' in param.keys():
            dt = handle_dt(param['lr_P'], lr_Ps=param['lr_Ps'], physical_time=physical_time)
        else:
            dt = handle_dt(param['lr_P'], lr_Ps=None, physical_time=physical_time)
            
        X_ = param['X_']
        
        dataset = param['dataset']
        if quantile == True:
            if 'gaussian' in dataset:
                 r_param = param['sigma_Q']
            elif 'student_t' in dataset:
                r_param = param['nu']
            elif dataset == 'Extension_of_gaussian':
                r_param = param['a']
            
    trajectories = result['trajectories'][:int(epochs/save_iter)]
    
    iter_nos = list(range(save_iter, save_iter+epochs, save_iter))
    time_steps = calculate_time_steps(dt, iter_nos, physical_time=physical_time)
        
    # make a frame
    fig, ax = plt.subplots()
    
    ims = []
    
    x1 = [x[:,0] for x in trajectories]
    x2 = [x[:,1] for x in trajectories]

    xlims = set_axis_lim([X_[:,0]] + x1, mask=1, lb=x_lim[0], ub=x_lim[1])
    ylims = set_axis_lim([X_[:,1]] + x2, mask=1, lb=y_lim[0], ub=y_lim[1])
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    for i, x in enumerate(trajectories):
        if physical_time == True:
            ttl = ax.text(0.5,1.05, "t = %.3f" % time_steps[i], bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},transform=ax.transAxes, ha="center")
        else:
            ttl = ax.text(0.5,1.05, "t = %d" % iter_nos[i], bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},transform=ax.transAxes, ha="center")

        im = ax.scatter(x[ :, 0], x[ :, 1], c='m', s=3, zorder=10, alpha=0.7)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        
        # plot quantile contour or target data
        if quantile == True:
            flag = add_quantile_contour(ax, dataset, r_param)
            if flag == -1:
                ax.scatter(X_[:,0], X_[:,1], c='c', s=3, zorder=1, alpha=0.7)
        else:
            ax.scatter(X_[:,0], X_[:,1], c='c', s=3, zorder=1, alpha=0.7)
            
        
        ims.append([im, ttl])
        
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=False, repeat_delay=200)
   
    if save_gif:
        writergif = animation.PillowWriter(fps=3)
        f = filename.split('.pickle')
        ani.save(f[0]+"-movie.gif", writer=writergif)
    
    show_or_close_figure(show)
    

# -----------------------------------------
# Additional features for Loss plot
# -----------------------------------------      
def fit_line(epochs, loss_states, plot_scale, save_iter, dt):
# calculate fitting line of loss
    import numpy as np
    start_idx = 2
    end_idx = min([20, int(epochs/save_iter)])
    if type(dt) == list : # decaying dt
        x_range = np.cumsum(dt[start_idx*save_iter+1:end_idx*save_iter+1])
    else: # constant dt
        x_range = np.arange(start_idx*save_iter+1, end_idx*save_iter+1, save_iter)*dt
    logB = np.log10(np.abs(loss_states[start_idx:end_idx]))
    #print(len(x_range), len(logB))
    
    if plot_scale == "semilogy":
        line_coefs = np.polyfit(x_range, logB, 1)
    elif plot_scale == "loglog":
        log_x = np.log10(x_range)
        line_coefs = np.polyfit(log_x, logB, 1)
    return x_range, line_coefs
    
def plot_fitting_line(x_range, line_coefs, plot_scale):
# plot fitting line
    m, y0 = line_coefs
    if plot_scale == "semilogy":
        plt.plot(x_range, 10**(x_range*m+y0), linestyle = 'dotted', label = 'exp[(%.4f)x+(%.2f)]' %(m,y0) )
    if plot_scale == "loglog":
        plt.plot(x_range, (10**y0)*x_range**m, linestyle = 'dotted', label = 'x^(%.4f)*10^(%.2f)' %(m,y0) )
    
def plot_loss(loss_states, epochs, physical_time, plot_scale, dt, exp_alias=None, save_iter=1, lty="solid", color = None, linewidth=1):
# plot setting for the certain type of loss (loss_states) with(out) fitting lines
# plot_scale = 'semilogy' or 'loglog'
    from numpy import arange
    if physical_time == True:
        xlabel_name = 'Time'
    else:
        xlabel_name = 'Iteration'
    
    iter_nos = range(save_iter, epochs+save_iter, save_iter)
    x_val = calculate_time_steps(dt, iter_nos, physical_time)
    '''
    if type(dt) == list : # decaying dt
        from numpy import cumsum
        try:
            dt = [x[0] for x in dt]
        except:
            pass
        x_val = cumsum(dt[0:epochs+save_iter:save_iter])
    else: # constant dt
        x_val = arange(save_iter,epochs+save_iter, save_iter)*dt
    '''
    if exp_alias == None:
        if plot_scale == "semilogy":
            plt.semilogy(x_val, [abs(x) for x in loss_states], color=color, linestyle=lty, linewidth=linewidth)
        if plot_scale == "loglog":
            plt.loglog(x_val, [abs(x) for x in loss_states], color=color, linestyle=lty, linewidth=linewidth)
            xlabel_name = xlabel_name+ ' (log scale)'
    else:
        if plot_scale == "semilogy":
            plt.semilogy(x_val, [abs(x) for x in loss_states], color=color, linestyle=lty, linewidth=linewidth, label=exp_alias)
        if plot_scale == "loglog":
            plt.loglog(x_val, [abs(x) for x in loss_states], color=color, linestyle=lty, linewidth=linewidth, label=exp_alias)
            xlabel_name = xlabel_name+ ' (log scale)'
    plt.xlabel(xlabel_name, fontsize="16")

# -----------------------------------
# plot generic loss
# -----------------------------------
def plot_losses(loss_type, loss_states=None, plot_scale='semilogy', fitting_line=False, save_iter = 1, dt = None, iter_nos = None, exp_alias_=None, epochs=0, ylims=None, physical_time=True, filename=None, show=False):
# from one file, plot a designated type of loss (loss_type) 
# iter_nos = [t_1, t_2, t_3,...] marks dots on the loss value of chosen epochs
    print(f'plot {loss_type}')
    
    # load data
    if 'epochs' not in filename:
        param, result = load_pickle(filename)        
        if epochs==0:
            epochs = param['epochs']
        if epochs > 100:
            save_iter = param['save_iter']
        else:
            save_iter = 1
        
        if 'lr_Ps' in param.keys():
            dt = handle_dt(param['lr_P'], lr_Ps=param['lr_Ps'], physical_time=physical_time)
        else:
            dt = handle_dt(param['lr_P'], lr_Ps=None, physical_time=physical_time)
            
        loss_states = result[loss_type]
        if np.isnan(result[loss_type]).any():
            print(f'{filename} loss diverged')   
            result[loss_type] = len(result[loss_type])*[0] 
        loss_states = np.array(result[loss_type])
        
        if exp_alias_ != None:
            exp_alias = '%s=%s' % (exp_alias_, param[exp_alias_])
    else:
        dt = handle_dt(lr_Ps=dt, physical_time=physical_time)
        
    for x in loss_states:
        if np.isnan(x):
            x = 0
        
    if loss_type != 'FIDs':
        save_iter = 1
    if fitting_line == True:
        x_range, line_coefs = fit_line(epochs, loss_states, plot_scale, save_iter, dt)
        
    loss_states = loss_states[:int(epochs/save_iter)]
    print("Last: %s = %f" % (loss_type, loss_states[-1]))
    
    # plot loss
    plot_loss(loss_states, epochs, physical_time, plot_scale, dt=dt, save_iter=save_iter)    
    
    # mark specific points
    if iter_nos != None:
        if type(dt) == list : # decaying dt
            x_val = [sum(dt[:iter_no]) for iter_no in iter_nos]
        else: # constant dt
            x_val = [iter_no*dt for iter_no in iter_nos]
        
        plt.plot(x_val, [abs(loss_states[x-1]) for x in iter_nos], '.', color='red')
        
    # fitting line
    if fitting_line == True:
        plot_fitting_line(x_range, line_coefs, plot_scale)
    if loss_type == 'divergences':
        plt.ylabel('Divergences (log scale)')
    elif loss_type == 'KE_Ps':    
        plt.ylabel(r'KE = $\frac{1}{2}\|\|dP\|\|_2^2 (log scale)$')
    elif loss_type == 'FIDs':    
        plt.ylabel('FID (log scale)')    
    
    '''
    if type(dt) == list : # decaying dt
        plt.xlim([dt[0]*save_iter, sum(dt)])
    else: # constant dt
        plt.xlim([dt*save_iter, epochs*dt])
    '''
    plt.xlim([dt[0]*save_iter, epochs*dt[0]])
    if ylims != None:
        plt.ylim(ylims)    
    if exp_alias_ != None:
        plt.legend()
    
    plt.tight_layout()
    f = filename.split('.pickle')
    plt.savefig(f[0]+"-"+loss_type+".png",bbox_inches='tight')
    
        
    show_or_close_figure(show)


# -----------------------------------------
# All figures of one run
# -----------------------------------------
# intermediate=True : called during a run with the current state in `data`; files get the suffix _<epochs>epochs
# intermediate=False: called after a run; everything is read from the result pickle `filename`
def plot_result(filename, intermediate=False, epochs = 0, iter_nos = None, data = [], show=False):
    if intermediate == True:
        trajectories = data['trajectories']
        divergences = data['divergences']
        KE_Ps = data['KE_Ps']
        X_ = data['X_']
        Y_ = data['Y_']
        dt = data['dt']
        save_iter = data['save_iter']
        dataset = data['dataset']
        r_param = data['r_param']
        vectorfields = data['vectorfields']
        f = filename.split('.pickle')
        filename_ = f[0] + "_%depochs" % epochs + f[1]
    else:
        trajectories, divergences, KE_Ps, X_, Y_, dt, save_iter = None, None, None, None, None, None, 1
        dataset, r_param, vectorfields = None, None, []
        filename_ = filename
        epochs = 0

    proj_axes = [0,1]
    x_lim = [None, None]
    y_lim = [None, None]

    if intermediate == False:
        trajectories_to_animation(physical_time=True, epochs=epochs, quantile = False, dataset=dataset, r_param=r_param, x_lim=x_lim, y_lim=y_lim, save_gif=True, filename = filename, show = show)
        plot_initial_data(proj_axes = proj_axes, x_lim = x_lim, y_lim = y_lim, filename=filename_, show = show)
        plot_output_target(proj_axes = proj_axes, x_lim = x_lim,y_lim = y_lim, filename=filename_, show = show)

    plot_trajectories(trajectories=trajectories, dt=dt, X_=X_, Y_=Y_, dataset = dataset, r_param=r_param, vectorfields = vectorfields, proj_axes = proj_axes, pick_samples =None, epochs = epochs, iter_nos = None, physical_time=True, save_iter = save_iter, track_velocity=False, arrow_scale = 1, iscolor=False, quantile=False, exp_alias_ = None, x_lim = x_lim, y_lim = y_lim, filename = filename_, show = show)
    plot_losses(loss_type='divergences', loss_states=divergences, plot_scale='semilogy',dt = dt,  fitting_line=False, save_iter = 1, iter_nos = None, exp_alias_=None, epochs=epochs, ylims=None, physical_time=True, filename=filename_, show = show)
    plot_losses(loss_type='KE_Ps', loss_states=KE_Ps, plot_scale='semilogy', dt = dt, fitting_line=False, save_iter = 1,  iter_nos = None, exp_alias_=None, epochs=epochs, ylims=None, physical_time=True, filename=filename_, show = show)


# plot data from loading pickled files
if __name__ == "__main__":
    if len(argv) == 2:
        filename = argv[1]
    else:
        print('Put filename for argv[1]!')
    plot_result(filename, intermediate=False, epochs = 0, iter_nos = None, show = True)
