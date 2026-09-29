import numpy as np
import tensorflow as tf
import sys

def generate_one_hot_encoding(
    N: int,
    N_classes: int,
    random_seed: int,
    data=[]
):
    if data == []:
        np.random.seed(random_seed)
        data = np.random.randint(N_classes, size=N)
        
    data = np.ndarray.flatten(data)
    X_label = np.zeros((data.size, data.max()+1))
    X_label[np.arange(data.size),data] = 1
    
    return X_label


def generate_data(param):
    # Input
    # param: parameters dictionary
    # Outputs
    # X_, Y_, [X_label, Y_label]
    # param
    if param['N_dim'] == None:
        param['N_dim'] = 2
        
    sys.path.append('../')
    
    if param['dataset'] == 'Learning_gaussian':
        from data.Random_samples import generate_gaussian
        param['expname'] = param['expname']+'_%.2f' % param['sigma_Q']
        X_ = generate_gaussian(size=(param['N_samples_Q'], param['N_dim']), m=0.0, std=param['sigma_Q'], random_seed=param['random_seed']) # target
        Y_ = generate_gaussian(size=(param['N_samples_P'], param['N_dim']), m=10.0, std=param['sigma_P'], random_seed=param['random_seed']+100) # initial
        
    elif param['dataset'] == 'Mixture_of_gaussians':
        from data.Random_samples import generate_gaussian, generate_four_gaussians
        param['expname'] = param['expname']+'_%.2f' % param['sigma_Q']
        X_ = generate_four_gaussians(size=(param['N_samples_Q'], param['N_dim']), dist=4.0, std=param['sigma_Q'], random_seed=param['random_seed']) # target
        Y_ = generate_gaussian(size=(param['N_samples_P'], param['N_dim']), m=0.0, std=param['sigma_P'], random_seed=param['random_seed']+100) # initial
    
    elif 'Aniso_student_t' in param['dataset'] or 'aniso_student_t' in param['dataset']:
        # Anisotropic Student-t: independent univariate-t per coordinate, vector of dof.
        # param['nu_vec'] is a list of length param['N_dim'].
        import pickle, os
        from data.Random_samples import generate_anisotropic_student_t, generate_gaussian
        nu_vec = list(param['nu_vec'])
        assert len(nu_vec) == param['N_dim'], (
            f"nu_vec has length {len(nu_vec)} but N_dim={param['N_dim']}"
        )
        # Encode the nu vector in the experiment name so per-coord runs are distinct.
        param['expname'] = param['expname'] + '_d{}_{}'.format(
            param['N_dim'],
            '-'.join(f"{float(v):g}" for v in nu_vec),
        )
        X_ = generate_anisotropic_student_t(
            size=(param['N_samples_Q'], param['N_dim']),
            nu_vec=nu_vec,
            random_seed=param['random_seed'],
        ).astype(np.float32)
        # Warm-start from a previously-saved Aniso pretrain pickle if it exists.
        pretrain_pkl = param.get('aniso_pretrain_pkl', None)
        if pretrain_pkl and os.path.exists(pretrain_pkl):
            with open(pretrain_pkl, 'rb') as _f:
                _, _pretrain_result = pickle.load(_f)
            Y_ = np.array(_pretrain_result['trajectories'][-1][:param['N_samples_P']],
                          dtype=np.float32)
        else:
            Y_ = generate_gaussian(
                size=(param['N_samples_P'], param['N_dim']),
                m=0.0, std=param['sigma_P'],
                random_seed=param['random_seed'] + 100,
            ).astype(np.float32)

    elif 'Learning_student_t' in param['dataset'] or 'CVaR_student_t' in param['dataset']:
        import pickle, os
        from data.Random_samples import generate_student_t, generate_gaussian
        param['expname'] = param['expname'] +'_%.2f' % param['nu']
        X_ = generate_student_t(size=(param['N_samples_Q'], param['N_dim']), m=0.0, nu=param['nu'], random_seed=param['random_seed']) # target
        pretrain_pkl = (
            f"../assets/Learning_student_t_nu{param['nu']:.1f}/"
            f"KL-Lipschitz_1.0000_{param['nu']:.2f}_10000_10000_00_pretrain.pickle"
        )
        if os.path.exists(pretrain_pkl):
            with open(pretrain_pkl, 'rb') as _f:
                _, _pretrain_result = pickle.load(_f)
            Y_ = np.array(_pretrain_result['trajectories'][-1][:param['N_samples_P']], dtype=np.float32)
        else:
            Y_ = generate_gaussian(size=(param['N_samples_P'], param['N_dim']), m=0.0, std=param['sigma_P'], random_seed=param['random_seed']+100) # initial
        
    elif param['dataset'] == 'Learning_translated_student_t':
        from data.Random_samples import generate_student_t, generate_gaussian
        param['expname'] = param['expname'] +'_%.2f' % param['nu']
        X_ = generate_student_t(size=(param['N_samples_Q'], param['N_dim']), m=0.0, nu=param['nu'], random_seed=param['random_seed']) # target
        Y_ = generate_gaussian(size=(param['N_samples_P'], param['N_dim']), m=10.0, std=param['sigma_P'], random_seed=param['random_seed']+100) # initial
        
    elif param['dataset'] == 'Lorenz63':        
        param['N_dim'] = 3
        filename = f"data/lorenzdataset_10.00-2.67-28.00_{param['N_samples_P']}.npy"
        Y_ = np.load(filename)
        
        filename = f"data/lorenzdataset_14.00-4.00-45.00_{param['N_samples_Q']}.npy"
        X_ = np.load(filename)
        
    elif param['dataset'] == 'Refine_heavytail':
        import pickle
        #import copy
        with open("data/alpha=02.00-Lipschitz_1.0000_GAN_df1.00_10000samples_0.pickle", 'rb') as file:
            dataset = pickle.load(file)
            Y_ = dataset[:5000]
            X_ = dataset[5000:]#copy.deepcopy(Y_)
            
    elif param['dataset'] == 'Refine_heavytail2':
        from scipy.stats import multivariate_t
        import pickle
        #import copy
        with open("data/alpha=02.00-Lipschitz_1.0000_GAN_df1.00_10000samples_0.pickle", 'rb') as file:
            dataset = pickle.load(file)
            Y_ = dataset[:5000]
        df = 1        # degrees of freedom (Student-t with df=1 is Cauchy)
        dim = 2       # 2D isotropic
        n_samples = 5000

        # Generate samples
        # Each coordinate is drawn from univariate t(df) independently (isotropic)
        X_ = multivariate_t.rvs(loc=np.zeros(dim), shape=np.eye(dim), df=df, size=n_samples, random_state=0)


    elif param['dataset'] == 'Refine_heavytail3':
        from scipy.stats import multivariate_t
        import pickle
        #import copy
        with open("data/alpha=02.00-Lipschitz_1.0000_GPA_df1.50_10000samples_1.pickle", 'rb') as file:
            dataset = pickle.load(file)
            Y_ = dataset[:5000]
        df = 1.5        # degrees of freedom (Student-t with df=1.5)
        dim = 2       # 2D isotropic
        n_samples = 5000

        # Generate samples
        # Each coordinate is drawn from univariate t(df) independently (isotropic)
        X_ = multivariate_t.rvs(loc=np.zeros(dim), shape=np.eye(dim), df=df, size=n_samples, random_state=0)

   

    elif param['dataset'] == 'Refine_heavytail4':
        from scipy.stats import multivariate_t
        import pickle
        #import copy
        with open("data/KL-Lipschitz_1.0000_GPA_df1.20_10000samples_1.pickle", 'rb') as file:
            dataset = pickle.load(file)
            Y_ = dataset[:5000]
        df = 1.2        # degrees of freedom (Student-t with df=1.2)
        dim = 2       # 2D isotropic
        n_samples = 5000

        # Generate samples
        # Each coordinate is drawn from univariate t(df) independently (isotropic)
        X_ = multivariate_t.rvs(loc=np.zeros(dim), shape=np.eye(dim), df=df, size=n_samples, random_state=0)
        
    elif 'FF25' in param['dataset']:
        import pickle, os
        from data.ff25_loader import load_ff25

        freq = 'daily' if 'daily' in param['dataset'] else 'monthly'
        param['N_dim'] = 25
        X_, Y_gaussian = load_ff25(
            freq=freq,
            N_Q=param['N_samples_Q'],
            N_P=param['N_samples_P'],
            random_seed=param['random_seed'],
            start_date=param.get('start_date'),
            end_date=param.get('end_date'),
            center=param.get('ff25_center', True),
            bootstrap_Q=param.get('ff25_bootstrap', False),
        )
        # Keep param in sync if loader shrank Q (e.g. no-bootstrap + N_Q>T)
        param['N_samples_Q'] = X_.shape[0]

        if 'CVaR_FF25' in param['dataset']:
            pretrain_pkl = (
                f"../assets/Learning_FF25_{freq}/"
                f"KL-Lipschitz_1.0000_5000_5000_00_pretrain.pickle"
            )
            if os.path.exists(pretrain_pkl):
                with open(pretrain_pkl, 'rb') as _f:
                    _, _pretrain_result = pickle.load(_f)
                Y_ = np.array(_pretrain_result['trajectories'][-1][:param['N_samples_P']], dtype=np.float32)
            else:
                Y_ = Y_gaussian
        else:
            Y_ = Y_gaussian

    elif 'Streamflow' in param['dataset']:
        # USGS Ohio-basin daily streamflow (64 gauges, 2005-2024, 7305 x 64), specific discharge in mm/day.
        from data.streamflow_loader import load_streamflow
        X_, Y_gaussian = load_streamflow(N_Q=param['N_samples_Q'],
                                         N_P=param['N_samples_P'],
                                         random_seed=param['random_seed'],
                                         scale='area')
        param['N_dim'] = X_.shape[1]
        param['N_samples_Q'] = X_.shape[0]
        Y_ = Y_gaussian


    elif 'custom' in param['dataset']:
        # Target samples of the user: --target_file, a .npy array of shape (N, d). All rows are used.
        # Initial sample: Gaussian with the mean and standard deviation of every coordinate of the target.
        if not param.get('target_file'):
            raise ValueError("--dataset %s needs --target_file <.npy array of shape (N, d)>" % param['dataset'])
        X_ = np.asarray(np.load(param['target_file']), dtype=np.float32)
        if X_.ndim != 2:
            raise ValueError("--target_file must hold an array of shape (N, d), got %s" % (X_.shape,))
        param['N_dim'] = X_.shape[1]
        param['N_samples_Q'] = X_.shape[0]
        rng = np.random.default_rng(param['random_seed'])
        Y_ = (rng.standard_normal((param['N_samples_P'], X_.shape[1])).astype(np.float32)
              * X_.std(axis=0)[None, :] + X_.mean(axis=0)[None, :]).astype(np.float32)

    elif param['dataset'] in ('Neal_funnel', 'Learning_Neal_funnel', 'CVaR_Neal_funnel'):
        # Neal's funnel (Neal 2003):  v ~ N(0,9),  x | v ~ N(0, exp(v/2))
        rng = np.random.default_rng(param['random_seed'])
        n_q = param['N_samples_Q']
        n_p = param['N_samples_P']
        v_q = rng.normal(0.0, 3.0, size=n_q)               # std=3 => var=9
        x_q = rng.normal(0.0, np.exp(v_q / 2.0), size=n_q)
        X_ = np.stack([v_q, x_q], axis=1).astype(np.float32)
        rng2 = np.random.default_rng(param['random_seed'] + 100)
        Y_ = rng2.normal(0.0, param['sigma_P'],
                         size=(n_p, 2)).astype(np.float32)


    if param['mb_size_P'] > param['N_samples_P']:
        param['mb_size_P'] = param['N_samples_P']
    if param['mb_size_Q'] > param['N_samples_Q']:
        param['mb_size_Q'] = param['N_samples_Q']
    
    if 'X_label' not in locals():
        X_label = None
    if 'Y_label' not in locals():
        Y_label = None
        
    if param['unseen'] == False:
        return param, X_, Y_, X_label, Y_label
    else:
        if 'Y_unseen_label' not in locals():
            Y_unseen_label = None
        param['Y_unseen_label'] = Y_unseen_label
        return param, X_, Y_, X_label, Y_label, Y_unseen, Y_unseen_label
