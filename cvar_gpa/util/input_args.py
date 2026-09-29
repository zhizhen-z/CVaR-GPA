import argparse
    
def input_params():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument(
        '-N_Q', '--N_samples_Q', type=int, help='total number of target samples', # default=200,
    )
    parser.add_argument(
        '-N_P', '--N_samples_P', type=int, help='total number of prior samples', #default=200,
    )
    parser.add_argument(
        '-N_dim', type=int, help='entire dimension of input data',
    )
    parser.add_argument(
        '-N_submnfld_dim', type=int, help='submanifold dimension of input data',
    )
    parser.add_argument(
        '-N_project_dim', type=int, help='dimension of PCA projected space on input',
    )
    parser.add_argument(
        '-sample_latent', type=bool, help='True: sample in the latent space, False: sample in the physical space', #default = False,
    )
    # Dataset property
    parser.add_argument(
        '--dataset', type=str, #choices=['Learning_gaussian', 'Mixture_of_gaussians', 'Mixture_of_gaussians2','Mixture_of_gaussians3','Mixture_of_gaussians4', 'Stretched_exponential', 'Learning_student_t', 'Mixture_of_student_t', 'Mixture_of_student_t_submnfld', 'Mixture_of_gaussians_submnfld','MNIST', 'CIFAR10', 'MNIST_switch', 'CIFAR10_switch', 'MNIST_ae', 'MNIST_ae_switch','CIFAR10_ae',  'Mixture_of_gaussians_submnfld_ae','BreastCancer', '1D_pts', '2D_pts','1D_dirac2gaussian', '1D_dirac2uniform','Lorenz63', 'Checkerboard'], #default='Learning_gaussian',
    )
    parser.add_argument(
        '-beta', type=float, help='gibbs distribution of -|x|^\beta',
    )
    parser.add_argument(
        '-sigma_P', type=float, help='std of initial gaussian distribution',
    )
    parser.add_argument(
        '-sigma_Q', type=float, help='std of target gaussian distribution',
    )
    parser.add_argument(
        '-nu', type=float, help='df of target student-t distribution',
    )
    parser.add_argument(
        '-nus', type=float, nargs="+", help='dfs of target student-t distribution',
    )
    parser.add_argument(
        '--nu_vec', type=str, default=None,
        help='comma-separated per-coord dof for anisotropic Student-t '
             '(e.g. "1.0,1.3,1.7,2.0,5.0"). Length must equal N_dim.',
    )
    parser.add_argument(
        '--aniso_pretrain_pkl', type=str, default=None,
        help='path to anisotropic-pretrain pickle for warm-start',
    )
    # Kinetic-energy-driven lr_P decay (see Theorem 2.10 connection).
    parser.add_argument(
        '--lr_decay_mode', type=str, default='none',
        choices=['none', 'ke', 'polyak'],
        help='lr_P decay schedule. '
             '"ke": lr_P = lr0 * clip(KE_ema/KE_ref, floor, 1). '
             '"polyak": lr_P = lr0 * clip(|diff/target|^p, floor, 1) where '
             'diff = CVaR_Q - CVaR_target; reactive to overshoot.',
    )
    parser.add_argument(
        '--lr_decay_p', type=float, default=0.5,
        help='exponent on |diff/target| in the polyak schedule '
             '(0.5 = sqrt, 1 = linear)',
    )
    parser.add_argument(
        '--lr_decay_beta', type=float, default=0.99,
        help='EMA smoothing factor for KE_P (closer to 1 = smoother)',
    )
    parser.add_argument(
        '--lr_decay_burnin', type=int, default=1000,
        help='iterations during which KE_ref is identified (peak of KE_ema)',
    )
    parser.add_argument(
        '--lr_decay_floor', type=float, default=0.01,
        help='floor on the lr_P / lr0 ratio (e.g. 0.01 = no smaller than lr0/100)',
    )
    parser.add_argument(
        '-interval_length', type=float, help='interval length of the uniform distribution',
    )
    parser.add_argument(
        '-label', type=int, nargs="+", help='class label of image data',
    )
    parser.add_argument(
        '-pts_P', type=float, nargs="+", #default=[10.0,]
    )
    parser.add_argument(
        '-pts_Q', type=float, nargs="+", #default=[0.0,]
    )
    parser.add_argument(
        '-pts_P_2', type=float, nargs="+", #default=[0.0,]
    )
    parser.add_argument(
        '-pts_Q_2', type=float, nargs="+", #default=[0.0,]
    )
    parser.add_argument(
        '-y0', type=float, nargs="+", #default=[1.0,2.0, 2.0]
    )
    parser.add_argument(
        '--random_seed', type=int, help='random seed for data generator', #default=0,
    )
    
    
    # (f, Gamma)-divergence
    parser.add_argument(
        '--f', type=str, choices=['KL', 'alpha', 'reverse_KL', 'reverse_alpha','Dinf','reverse_Dinf'], #default='KL',
    )
    parser.add_argument(
        '-alpha', type=float, help='parameter value for alpha divergence',
    )    
    parser.add_argument(
        '--formulation', type=str, choices=['LT', 'DV'], help='LT or DV in case of f=KL, otherwise, keep LT', #default='LT',
    )
    parser.add_argument(
        '--Gamma', type=str, choices=['Lipshitz'], #default='Lipshitz',
    )
    parser.add_argument(
        '-L', type=float, help='Lipshitz constant: default=inf w/o constraint',
    )
    parser.add_argument(
        '--reverse', type=bool,  help='True -> D(Q|P), False -> D(P|Q)', #default=False,
    )
    parser.add_argument(
        '--constraint', type=str, choices=['hard', 'soft'], #default='hard',
    )
    parser.add_argument(
        '-lamda', type=float,  help='coefficient on soft constraint', #default=100.0,
    )
    parser.add_argument(
        '--generative_model', type=str, default="GPA_NN", # choices=['GPA_NN', 'GAN'], 
    )
    
    # 'Coefficient for CVaR squared-difference vector field
    parser.add_argument("--lam", type=float, default=120.0,
                    help="CVaR weight in the convention lam = lambda/(1-alpha): coef_CVaR = lam*(1-alpha), "
                         "so the CVaR velocity is lam * 2*Delta * unit * gate. "
                         "The paper uses --lam 4e-3.")

    parser.add_argument('--sweep_summary', type=str, default="coef_CVaR_summary.csv")
    
    # Wasserstein-1 metric
    parser.add_argument(
        '-calc_Wasserstein1', type=bool,  help='True -> check Wasserstein-1 metric', #default=False,
    )
      
    
    # Neural Network <phi>
    parser.add_argument(
        '-NN', '--NN_model', type=str,  choices=['fnn', 'cnn', 'cnn-fnn'], #default='fnn',
    )
    parser.add_argument(
        '-N_fnn_layers', type=int, nargs='+', help='list of the number of FNN hidden layer units / the number of CNN feed-forward hidden layer units',
    )
    parser.add_argument(
        '-N_cnn_layers', type=int, nargs='+', help='list of the number of CNN channels',
    )
    parser.add_argument(
        '--activation_ftn', type=str, nargs='+',  choices=['relu', 'mollified_relu_cos3','mollified_relu_poly3','mollified_relu_cos3_shift','softplus', 'leaky_relu','leaky_relu_001','elu', 'bounded_relu', 'bounded_elu'], help='[0]: for the fnn/convolutional layer, [1]: for the cnn feed-forward layer, [2]: for the LAST cnn feed-forward layer', #default=['relu',],
    )
    parser.add_argument(
        '-eps', type=float,  help='Mollifier shape adjusting parameter when using mollified relu3 activations', #default = 0.5,
    )
    parser.add_argument(
        '--N_conditions', type=int, help='number of classes for the conditional setting', #default=1,
    )
    
    
    # RKHS <phi>
    parser.add_argument(
        '--kernel', type=str,  choices=['gaussian'], #default='gaussian',
    )
    parser.add_argument(
        '-bandwidth', type=float, help='gaussian kernel bandwidth', # default = 2.0,
    )
    
    
    # Discriminator training parameters
    parser.add_argument(
        '--lr_phi', type=float, help='lr for phi',# default=0.001,
    )
    parser.add_argument(
        '-ep_phi', '--epochs_phi', type=int, help='# updates for phi to find phi*',# default=3,
    )
    parser.add_argument(
        '--optimizer', type=str, choices=['sgd', 'adam',], help='optimizer for NN',# default='adam',
    )
    
    # Particles transportation parameters
    parser.add_argument(
        '-ep', '--epochs', type=int, help='# updates for P', #default=1000,
    )
    parser.add_argument(
        '--ode_solver', type=str, choices=['forward_euler', 'AB2', 'AB3', 'AB4', 'AB5', 'ABM1', 'Heun', 'ABM2', 'ABM3', 'ABM4', 'ABM5', 'RK4', 'ode45', 'Newton', 'BFGS','Gradient_Ascent'], help='ode solver for particle ode', #default='forward_euler',
    )
    parser.add_argument(
        '-mobility', type=str, help='problem dependent mobility function\nRecommendation: MNIST - bounded',
    )
    parser.add_argument(
        '-lr_P_decay', type=str, choices=['rational', 'step',], help='delta t decay',
    )
    parser.add_argument(
        '--lr_P', type=float, help='lr for P',#default=1.0,
    )
    
    parser.add_argument(
        '--exp_no', type=str, help='short experiment name under the same data', #default=0,
    )
    parser.add_argument(
        '--mb_size_P', type=int, help='mini batch size for the moving distribution P',# default=200,
    )
    parser.add_argument(
        '--mb_size_Q', type=int, help='mini batch size for the target distribution Q',# default=200,
    )
    
    
    # generated data
    parser.add_argument(
        '-unseen', type=bool, default = False, help='transport unseen data',
    )
    parser.add_argument(
        '-generator_status', type=str, # choices=['learn', 'eval', ], #default=None,
    )
    parser.add_argument(
        '-N_generated_samples', type=int, help='number of generated samples from generator', # default=200,
    )
    parser.add_argument(
        '-resample', type=bool, help='True -> resample P0 or R for each generator learning',# default=True,
    )
    parser.add_argument(
        '-resample_latent', type=bool, help='True -> resample at the latent space', #default = False,
    )
    parser.add_argument(
        '-N_gen_latent_dim', type=int, help='dimension of latent space',
    )
    
    #Band BVaR
    parser.add_argument('--beta_level', type=float, default=None)
    parser.add_argument('--ab_weight',  type=float, default=None)
    parser.add_argument('--no_cvar', action='store_true', default=False,
                        help='Disable CVaR term entirely (coef_CVaR = 0 exactly)')
    parser.add_argument('--report_beta_levels', type=float, nargs='+', default=None)
    parser.add_argument('--report_ab_weights', type=float, nargs='+', default=None)
    parser.add_argument('--cvar_gate', type=str, default=None,
                        choices=['hard', 'ramp'],
                        help='CVaR velocity gate (cvar_gpa_smooth.py; default ramp). '
                             '"ramp": smoothed CVaR: sigmoid '
                             'gate W((g - q_h)/h) at the unique smoothed quantile q_h, '
                             'softplus-smoothed RU hinge in the CVaR value. '
                             '"hard": strict indicator 1{g > eta_bar} (non-smoothed variant).')
    parser.add_argument('--ramp_h', type=float, default=None,
                        help='Smoothing parameter h (cvar_gpa_smooth.py). Fixed '
                             'hyperparameter; unset = 0.5. Must be > 0.')
    parser.add_argument('--varbar_rule', type=str, default=None,
                        choices=['legacy', 'varbar', 'var'],
                        help='Hard-gate eta_bar anchor in T(Qhat)=[VaR,VaRbar] '
                             '(cvar_gpa_smooth.py; default legacy). '
                             'VaR=g_(ceil(a*M)), VaRbar=g_(floor(a*M)+1). '
                             '"legacy": default behaviour; when a*M is an integer it '
                             'shifts one order statistic ABOVE VaRbar, i.e. outside '
                             'T(Qhat), pushing 3 particles at a=0.999/M=5000 and biasing '
                             'the reported CVaR high. "varbar": VaRbar, pushes 4, CVaR '
                             'exact. "var": VaR, pushes 5, CVaR exact.')
    parser.add_argument('--cvar_undershoot_only', action='store_true', default=False,
                        help='If set, CVaR loss is (min(CVaR_P - CVaR_target, 0))^2 instead of '
                             '(CVaR_P - CVaR_target)^2: force is zero when overshooting.')
    parser.add_argument('--lambda_cvar', type=float, default=None,
                        help='Alias of --lam (convention lam = lambda/(1-alpha)).')
    parser.add_argument('--lam_rule', type=str, default=None,
                        choices=['fixed', 'gain', 'init_gap'],
                        help="'init_gap' (DEFAULT): lam = min(scale/|Delta_0|, lam_init_gap_cap). 'gain': lam = lam_gain / lr_P, i.e. specify the "
                             "dimensionless gain g = lam*lr_P that alone governs the CVaR push "
                             "(decay factor 1-2g per iter); g=0.1 gives 4e-3 at lr_P=25. "
                             "'fixed': use --lam as given. 'init_gap': set "
                             "lam = lam_init_gap_scale / max(|Delta_0|, 2*lr_P) ONCE at start, "
                             "where Delta_0 = CVaR^tar - CVaR^{P_0} at the run's gate/h; lam "
                             "then stays constant (cvar_gpa_smooth.py only, joint_radial). The "
                             "2*lr_P floor is the parameter-free no-overshoot bound "
                             "(2*lam*lr_P <= 1); it only binds when the pretrain already sits "
                             "within 2*lr_P of the target CVaR.")
    parser.add_argument('--lam_gain', type=float, default=None,
                        help='Gain g* for --lam_rule gain: lam = g*/lr_P (default 0.1). Must be in (0, 0.5].')
    parser.add_argument('--lam_init_gap_cap', type=float, default=None,
                        help='Cap for --lam_rule init_gap: lam = min(scale/|Delta_0|, cap). Default 0.01. '
                             'The 2*lr_P no-overshoot floor is enforced as well.')
    parser.add_argument('--lam_init_gap_scale', type=float, default=None,
                        help='Numerator c in lam = c/|Delta_0| for --lam_rule init_gap '
                             '(default 1.0).')
    parser.add_argument('--stop_ke_tol', type=float, default=None,
                        help='If set, stop early once the kinetic energy KE_P stays '
                             '<= this tolerance for --stop_ke_window consecutive '
                             'iterations (the algorithm is dead: particles no longer '
                             'move). Use a tiny positive value (e.g. 1e-10); exact 0 '
                             'almost never occurs in float32 because the KL velocity '
                             'is a NN gradient. Default None = disabled.')
    parser.add_argument('--stop_ke_window', type=int, default=500,
                        help='Consecutive-iteration window for --stop_ke_tol.')
    parser.add_argument('--stop_tail_ke_median', type=float, default=None,
                        help='cvar_gpa_smooth.py only. If set to tol>0, stop (AND with '
                             '--stop_ke_median when both set) once the MEDIAN over the last W '
                             'iterations of the per-particle KL kinetic energy of the '
                             'k=(1-alpha)M pushed particles (k largest radii) is below tol: '
                             'the KL/CVaR standoff on the tail has released. '
                             'The paper uses 1e-10. Default None = off.')
    parser.add_argument('--stop_ke_median', type=float, default=None,
                        help='cvar_gpa_smooth.py only. If set to tol>0, stop (AND with '
                             '--stop_tail_ke_median when both set) once the MEDIAN of the '
                             'total kinetic energy KE_P over the last W iterations is below '
                             'tol: the whole system is quiet. Medians, not instantaneous '
                             'values, because per-iteration KE has critic spikes. '
                             'The paper uses 1e-10. Default None = off.')
    parser.add_argument('--stop_median_window', type=int, default=1000,
                        help='Window W (iterations) for the two median stop conditions.')

    parser.add_argument('--target_file', type=str, default=None,
                        help='Target samples for --dataset CVaR_custom or Learning_custom: '
                             'a .npy array of shape (N, d). A relative path is relative to cvar_gpa/.')

    # Progressive / warm-start training
    parser.add_argument('--init_P_file', type=str, default=None,
                        help='Path to a .pickle result file; use its final trajectory as initial P')
    
    # Autoencoder generator parameters
    parser.add_argument(
        '-NN_ae_model', type=str, choices=['fnn', 'cnn',], help='Neural network for the autoencoder',
    )
    parser.add_argument(
        '-ae_model', type=str, choices=['VAE', 'AE',], help='Autoencoder model',
    )
    parser.add_argument(
        '-encoder_channel_layers', type=int, nargs='+', help='weight matrices dimensions for layers (both fnn and cnn)', # default=[8,8,8],
    )
    parser.add_argument(
        '-encoder_kernel_shape_layers', type=int, nargs='+', help='kernel height and width for layers (for cnn) dim=dim(encoder_activation_ftn)',
    )
    parser.add_argument(
        '-encoder_activation_ftn', type=str, nargs='+',  choices=['relu', 'mollified_relu_cos3','mollified_relu_poly3','mollified_relu_cos3_shift','softplus', 'leaky_relu','elu', 'bounded_relu', 'bounded_elu'], help='[0]: for the fnn/convolutional layer, [1]: for the cnn feed-forward layer, [2]: for the LAST cnn feed-forward layer', #default=['relu',],
    )
    parser.add_argument(
        '-decoder_channel_layers', type=int, nargs='+', help='weight matrices dimensions for layers (both fnn and cnn)', # default=[8,8,8],
    )
    parser.add_argument(
        '-decoder_kernel_shape_layers', type=int, nargs='+', help='kernel height and width for layers (for cnn) dim=dim(encoder_activation_ftn)',
    )
    parser.add_argument(
        '-decoder_activation_ftn', type=str, nargs='+',  choices=['relu', 'mollified_relu_cos3','mollified_relu_poly3','mollified_relu_cos3_shift','softplus', 'leaky_relu','elu', 'bounded_relu', 'bounded_elu'], help='[0]: for the fnn/convolutional layer, [1]: for the cnn feed-forward layer, [2]: for the LAST cnn feed-forward layer', #default=['relu',],
    )
    parser.add_argument(
        '-encoder_L', type=float, help='Lipshitz constant for the encoder: default=inf w/o constraint',
    )
    parser.add_argument(
        '-decoder_L', type=float, help='Lipshitz constant for the decoder: default=inf w/o constraint',
    )
    
    
    
    # Autoencoder training parameters
    parser.add_argument(
        '-epochs_ae', type=int, help='# updates for AE to find psi_0*', #default=1000,
    )
    parser.add_argument(
        '-lr_ae', type=float, help='lr for ae',#default=1.0,
    )
    parser.add_argument(
        '-optimizer_ae', type=str, choices=['sgd', 'adam',], help='optimizer for AE',# default='adam',
    )
    parser.add_argument(
        '-epochs_dec', type=int, help='# updates for AE to find psi_t*', #default=1000,
    )
    parser.add_argument(
        '-lr_dec', type=float, help='lr for dec',#default=1.0,
    )
    parser.add_argument(
        '-optimizer_dec', type=str, choices=['sgd', 'adam',], help='optimizer for decoder',# default='adam',
    )
    parser.add_argument(
        '-dec_train_iter', type=int, help='train decoder per each dec_train_iter', #default=1000,
    )
    
    
    # save/display 
    parser.add_argument(
        '--save_iter', type=int, help='save results per each save_iter',# default=10,
    )
    parser.add_argument(
        '--plot_result', type=bool, help='True -> show plots',# default=False,
    )
    parser.add_argument(
        '--plot_intermediate_result', type=bool, help='True -> save intermediate plots',# default=False,
    )
    parser.add_argument(
        '-filename', type=str, help='filename for GPA_NN-Generator',# default='adam',
    )
    
    return parser.parse_known_args()
