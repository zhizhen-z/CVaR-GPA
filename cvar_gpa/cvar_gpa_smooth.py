#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# CVaR-GPA, smooth version: penalization by the smoothed CVaR (Algorithms 1 and 2 of the paper).
# This is the algorithm behind every reported experiment.
#
# Loss functional:
#     F_h(Q ; P^tar) = D^L_KL(Q || P^tar) + lambda * (CVaR_{alpha,h}^{P^tar,g} - CVaR_{alpha,h}^{Q,g})^2,   g(x) = ||x||.
#
# Per-particle velocity:
#     v_i = -grad phi*(Y_i) + lambda * (2 * Delta_k / (1 - alpha)) * w_i * (Y_i / ||Y_i||),
#     w_i = sigmoid((||Y_i|| - q_h) / h),   q_h = root of mean_i sigmoid((||Y_i|| - q) / h) = 1 - alpha.
#
# Hyperparameter mapping:
#   * --lam         -> lambda / (1 - alpha); the CVaR velocity is 2 * lam * Delta_k * w_i * Y_i/||Y_i||.
#   * --beta_level  -> the paper's alpha;  --ramp_h -> h;  -L -> Lipschitz constant;  --lr_P -> step size.
#   * --cvar_gate hard switches this driver to the indicator gate; cvar_gpa_hard.py is the stand-alone hard version.
#
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
import tensorflow as tf
import numpy as np
import random
import re
import sys
import copy
import functools

main_dir = os.getcwd()   # run from the cvar_gpa/ directory: configs/ and data/ are read from here, results go to ../assets/

# input parameters --------------------------------------
from util.input_args import input_params
p, _ = input_params()

import yaml
from yaml.loader import SafeLoader
yaml_file = "{main_dir}/configs/{dataset}-{generative_model}.yaml".format(
    main_dir=main_dir, dataset=p.dataset, generative_model=p.generative_model
)

with open(yaml_file, 'r') as f:
    param = yaml.load(f, Loader=SafeLoader)

updated_param = vars(p)
for param_key, param_val in updated_param.items():
    if type(param_val) == type(None):
        continue
    param[param_key] = param_val

if isinstance(param.get('nu_vec', None), str):
    param['nu_vec'] = [float(x) for x in param['nu_vec'].split(',') if x.strip()]

if param['alpha']:
    par = [param['alpha']]
    param['exptype'] = '%s=%05.2f-%s' % (param['f'], param['alpha'], param['Gamma'])
else:
    par = []
    param['exptype'] = '%s-%s' % (param['f'], param['Gamma'])

if param['L'] == None:
    param['expname'] = '%s_%s' % (param['exptype'], 'inf')
else:
    param['expname'] = '%s_%.4f' % (param['exptype'], param['L'])

if param['reverse'] == True:
    param['expname'] += '_reverse'

seed = int(param['random_seed'])
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.keras.utils.set_random_seed(seed)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

# Data generation ----------------------------------------
from util.generate_data import generate_data
if param['unseen'] == False:
    param, X_, Y_, X_label, Y_label = generate_data(param)
else:
    param, X_, Y_, X_label, Y_label, Y_unseen, Y_unseen_label = generate_data(param)

rescale_factor = 1.0
if param['dataset'] in ['BreastCancer', 'Labeled_disease']:
    rescale_factor = 10.0
elif param['dataset'] in ['Day34',]:
    rescale_factor = 4.0

Q = tf.constant(X_ / rescale_factor, dtype=tf.float32)
P = tf.Variable(Y_ / rescale_factor, dtype=tf.float32)

_init_P_file = param.get('init_P_file', None)
if _init_P_file:
    import pickle as _pkl
    with open(_init_P_file, 'rb') as _f:
        _prev_param, _prev_result = _pkl.load(_f)
    _init_arr = np.array(_prev_result['trajectories'][-1], dtype=np.float32) / rescale_factor
    if _init_arr.shape[0] != P.shape[0]:
        _init_arr = _init_arr[:P.shape[0]]
    assert _init_arr.shape == P.shape, \
        f"init_P shape {_init_arr.shape} != P shape {P.shape}"
    P.assign(_init_arr)
    print(f"Warm-start: P initialized from {_init_P_file}")
if param['unseen'] == True:
    P_unseen = tf.Variable(Y_unseen / rescale_factor, dtype=tf.float32)

if param['N_conditions'] > 1:
    Q_label = tf.constant(X_label, dtype=tf.float32)
    P_label = tf.constant(Y_label, dtype=tf.float32)
    if param['unseen'] == True:
        P_unseen_label = tf.constant(Y_unseen_label, dtype=tf.float32)
else:
    Q_label, P_label = None, None
    if param['unseen'] == True:
        P_unseen_label = None

data_par = {
    'P_label': P_label,
    'Q_label': Q_label,
    'mb_size_P': param['mb_size_P'],
    'mb_size_Q': param['mb_size_Q'],
    'N_samples_P': param['N_samples_P'],
    'N_samples_Q': param['N_samples_Q'],
}
if param['unseen'] == True:
    data_par.update({'P_unseen_label': P_unseen_label})

print("Data prepared.")

# Discriminator learning  -----------------------------------------
from lib.construct_NN import check_nn_topology, initialize_NN, model

N_fnn_layers, N_cnn_layers, param['activation_ftn'] = check_nn_topology(
    param['NN_model'], param['N_fnn_layers'], param['N_cnn_layers'], param['N_dim'], param['activation_ftn']
)

NN_par = {
    'NN_model': param['NN_model'],
    'activation_ftn': param['activation_ftn'],
    'N_dim': param['N_dim'],
    'N_cnn_layers': N_cnn_layers,
    'N_fnn_layers': N_fnn_layers,
    'N_conditions': param['N_conditions'],
    'constraint': param['constraint'],
    'L': param['L'],
    'eps': param['eps']
}

W, b = initialize_NN(NN_par)
phi = model(NN_par)

nu = tf.Variable(0.0, dtype=tf.float32)
parameters = {'W': W, 'b': b, 'nu': nu}

# =========================================================
# CVaR block.
# =========================================================

_alpha_val = float(param.get("beta_level", 0.999))       # paper's alpha
_ab_weight_val = float(param.get("ab_weight", 0.50))     # kept only for report

_tail_mass_val = 1.0 - _alpha_val                        # 1 - alpha

alpha_const  = tf.constant(_alpha_val,     dtype=tf.float32)
tail_mass_tf = tf.constant(_tail_mass_val, dtype=tf.float32)
inv_tail_tf  = tf.constant(1.0 / _tail_mass_val, dtype=tf.float32)

# CVaR risk: the radius, g(x) = ||x||.

# CVaR gate: 'ramp' (default; smoothed CVaR, sigmoid weights, Algorithms 1 and 2 of the paper) or
# 'hard' (strict indicator at eta_bar; the non-smoothed variant of the appendix).
cvar_gate = param.get('cvar_gate', None) or 'ramp'
param['cvar_gate'] = cvar_gate
if cvar_gate == 'ramp':
    param['expname'] += '_ramp'


varbar_rule = param.get('varbar_rule', None) or 'legacy'
if varbar_rule not in ('legacy', 'varbar', 'var'):
    raise ValueError("varbar_rule must be one of 'legacy', 'varbar', 'var'")
param['varbar_rule'] = varbar_rule

y = tf.Variable(0.0, dtype=tf.float32)   # eta_bar tracker for logs


def radius(P_):
    return tf.norm(P_, axis=1)           # (N,)


# --- numpy helpers (only used in the final CVaR report block) ---
def empirical_var_cvar_from_risk(risk_vals, beta):
    q_beta = float(np.quantile(risk_vals, beta))
    tail = risk_vals[risk_vals >= q_beta]
    cvar_beta = float(np.mean(tail)) if tail.size > 0 else float("nan")
    return q_beta, cvar_beta


def empirical_bcvar_from_risk(risk_vals, beta, a_val):
    q_beta, cvar_beta = empirical_var_cvar_from_risk(risk_vals, beta)
    ab_weight_local = a_val * beta
    low_mask = risk_vals < q_beta
    if np.any(low_mask):
        e_low = float(np.mean(risk_vals[low_mask]))
    else:
        e_low = float(np.mean(risk_vals[risk_vals <= q_beta]))
    bcvar_val = (1.0 - ab_weight_local) * cvar_beta + ab_weight_local * e_low
    return q_beta, cvar_beta, bcvar_val


# --- empirical tail statistics: sort-based eta_bar with boundary case ---
@tf.function
def empirical_var_bar(P_, risk_ftn):
    """Empirical quantile: sort risk descending, k = ceil(N*(1-alpha)).
    Generic case (k/N > 1-alpha): eta_bar = k-th largest.
    Boundary case (k/N == 1-alpha): eta_bar = (k-1)-th largest.
    """
    g = risk_ftn(P_)
    g_sorted = tf.sort(g)                        # ascending
    N = tf.shape(g_sorted)[0]
    N_float = tf.cast(N, tf.float32)

    tm_N = tail_mass_tf * N_float
    m = tf.cast(tf.math.ceil(tm_N), tf.int32)
    m = tf.clip_by_value(m, 1, N)
    k_asc = N - m                                # ascending-sort idx of m-th largest

    if varbar_rule == 'var':
        # VaR = g_(ceil(alpha*M)); 0-based idx = N - floor((1-alpha)*M) - 1
        m_lo = tf.clip_by_value(tf.cast(tf.math.floor(tm_N), tf.int32), 0, N - 1)
        return g_sorted[tf.maximum(N - m_lo - 1, 0)]
    if varbar_rule == 'varbar':
        # VaRbar = g_(floor(alpha*M)+1); 0-based idx = floor(alpha*M) = N - m.
        # No integer special case: the paper's formula already covers it.
        return g_sorted[k_asc]

    # 'legacy': default behaviour, incl. the off-by-one when (1-alpha)*M is an
    # integer (see the varbar_rule comment above).
    is_boundary = tf.equal(tf.cast(m, tf.float32), tm_N)
    k_bd = tf.minimum(k_asc + 1, N - 1)          # (k-1)-th largest
    eta_bar = tf.cond(is_boundary,
                      lambda: g_sorted[k_bd],
                      lambda: g_sorted[k_asc])
    return eta_bar


# --- empirical CVaR from eta_bar via the Rockafellar-Uryasev formula ---
@tf.function
def empirical_cvar(eta_bar, P_, risk_ftn):
    """CVaR = eta_bar + (1/(1-alpha)) * (1/N) * sum_i max(g(Z_i) - eta_bar, 0)."""
    g = risk_ftn(P_)
    pos = tf.nn.relu(g - eta_bar)
    return eta_bar + inv_tail_tf * tf.reduce_mean(pos)


# --- closed-form per-particle CVaR velocity (indicator gate) ---
@tf.function
def calc_cvar_velocity_paper(P_, eta_bar, cvar_target):
    g = radius(P_)
    cvar_current = empirical_cvar(eta_bar, P_, radius)
    Delta = cvar_target - cvar_current                       
    coef = 2.0 * Delta * inv_tail_tf                         # 2 Delta / (1-alpha)
    unit = P_ / (tf.expand_dims(g, 1) + 1e-12)               # (N, d) radial unit
    mask = tf.cast(g > eta_bar, tf.float32)                  # (N,)
    b = coef * unit * tf.expand_dims(mask, 1)                # (N, d)

    dP_band = -b                                             # sign for P -= lr*vf
    return dP_band, cvar_current


# =========================================================
# Smoothed-CVaR (ramp) gate
# "smoothed-CVaR gate".  Kernel W = logistic sigmoid; smoothed hinge
# rho_h(u) = h*softplus(u/h), so rho_h'(u) = W(u/h) -> 1{u>0} as h -> 0.
# The smoothed alpha-quantile q_h is the unique root of
#     mean_i W((g_i - q_h)/h) = 1 - alpha
# (strict convexity of the smoothed RU objective), so the inward/outward
# selection ambiguity does not arise and the velocity is the genuine
# gradient of  D_KL^L + lambda * (CVaR_h^tar - CVaR_h^Q)^2.
# =========================================================

def smoothed_quantile(g, h, n_bisect=60):
    """Bisection for q_h: mean(sigmoid((g - q)/h)) = 1 - alpha.  The l.h.s.
    is strictly decreasing in q, and the root is bracketed since sigmoid(+-30)
    is 1/0 to float32 precision.  g: (N,) risk values; h: scalar tensor."""
    lo = tf.reduce_min(g) - 30.0 * h
    hi = tf.reduce_max(g) + 30.0 * h
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        frac = tf.reduce_mean(tf.sigmoid((g - mid) / h))
        root_above = frac > tail_mass_tf
        lo = tf.where(root_above, mid, lo)
        hi = tf.where(root_above, hi, mid)
    return 0.5 * (lo + hi)


def smoothed_cvar(q_h, g, h):
    """CVaR_h = q_h + (1/(1-alpha)) * mean_i rho_h(g_i - q_h)."""
    return q_h + inv_tail_tf * tf.reduce_mean(h * tf.math.softplus((g - q_h) / h))


@tf.function
def calc_cvar_velocity_ramp(P_, q_h, cvar_target, h):
    """Velocity of the smoothed CVaR penalty:
        b_i = (2 Delta_h / (1-alpha)) * (Y_i/||Y_i||) * W((g_i - q_h)/h)
    with Delta_h = CVaR_h^tar - CVaR_h^Q.  Same sign convention as the hard
    version: dP_band = -b_i, caller does vf += coef_CVaR * dP_band."""
    g = radius(P_)
    cvar_current = smoothed_cvar(q_h, g, h)
    Delta = cvar_target - cvar_current
    coef = 2.0 * Delta * inv_tail_tf
    unit = P_ / (tf.expand_dims(g, 1) + 1e-12)               # (N, d) radial unit
    gate = tf.sigmoid((g - q_h) / h)                         # W((g - q_h)/h)
    b = coef * unit * tf.expand_dims(gate, 1)                # (N, d)
    dP_band = -b                                             # sign for P -= lr*vf
    return dP_band, cvar_current


# Train setting -----------------------------------------------------
from lib.train_NN import train_disc
lr_phi = tf.Variable(param['lr_phi'], trainable=False)

loss_par = {
    'f': param['f'],
    'formulation': param['formulation'],
    'par': par,
    'reverse': param['reverse'],
    'lamda': param['lamda']
}

from lib.transport_particles import calc_vectorfield, solve_ode
dPs = []
if param['unseen'] == True:
    unseen_dPs = []
if param['ode_solver'] in ['forward_euler', 'AB2', 'AB3', 'AB4', 'AB5']:
    aux_params = []
else:
    aux_params = {
        'parameters': parameters,
        'phi': phi,
        'Q': Q,
        'lr_phi': lr_phi,
        'epochs_phi': param['epochs_phi'],
        'loss_par': loss_par,
        'NN_par': NN_par,
        'data_par': data_par,
        'optimizer': param['optimizer']
    }

if param['mobility'] == 'bounded':
    from lib.construct_NN import bounded_relu

lr_P_init = param['lr_P']
lr_P = tf.Variable(lr_P_init, trainable=False)
lr_Ps = []

if param['calc_Wasserstein1'] == True:
    NN_par2 = {
        'NN_model': param['NN_model'],
        'activation_ftn': param['activation_ftn'],
        'N_dim': param['N_dim'],
        'N_cnn_layers': N_cnn_layers,
        'N_fnn_layers': N_fnn_layers,
        'N_conditions': param['N_conditions'],
        'constraint': param['constraint'],
        'L': 1.0,
        'eps': param['eps']
    }
    W2, b2 = initialize_NN(NN_par2)
    phi2 = model(NN_par2)
    parameters2 = {'W': W2, 'b': b2}
    from lib.train_NN import train_wasserstein1

# Save & plot settings -----------------------------------------------
from util.evaluate_metric import calc_ke, calc_grad_phi
if np.prod(param['N_dim']) <= 12:
    from util.evaluate_metric import calc_sinkhorn

trajectories = []
vectorfields = []
QoIs = []
divergences = []
wasserstein1s = []
KE_Ps = []
FIDs = []   # kept empty so that the result files keep their fields
comp_times = []
eta_history = []
cvar_vf_snapshots = []
kl_vf_snapshots = []
cvar_signed_diff_history = []       # signed Delta = CVaR_Q - CVaR_target per iter
if param['unseen'] == True:
    unseen_trajectories = []

if param['save_iter'] >= param['epochs']:
    param['save_iter'] = 1

if param['plot_result'] == True:
    from util.plot_result import plot_result

os.makedirs('../assets/' + param['dataset'], exist_ok=True)

# --lambda_cvar is an alias of --lam. Convention: --lam = lambda / (1 - alpha), with lambda the weight of
# the paper. Below, coef_CVaR = lam * (1 - alpha) multiplies the velocity term b_i = 2 Delta / (1 - alpha) * w_i * Y_i/||Y_i||,
# so the CVaR velocity is 2 * lam * Delta * w_i * Y_i/||Y_i||.
if param.get('lambda_cvar', None) is not None:
    param['lam'] = float(param['lambda_cvar'])

param['expname'] = param['expname'] + '_%04d_%04d_%02d_%s' % (
    param['N_samples_Q'], param['N_samples_P'], param['random_seed'], param['exp_no']
)
filename = '../assets/' + param['dataset'] + '/%s.pickle' % (param['expname'])

if param['plot_intermediate_result'] == True:
    if 'gaussian' in param['dataset'] and 'Extension' not in param['dataset']:
        r_param = param['sigma_Q']
    elif 'student_t' in param['dataset']:
        r_param = param['nu']
    elif param['dataset'] == 'Extension_of_gaussian':
        r_param = param['a']
    else:
        r_param = None

if param['N_dim'] == 1:
    xx = np.linspace(-3, 13, 300)
    xx = tf.constant(np.reshape(xx, (-1, 1)), dtype=tf.float32)
    phis = []
elif param['N_dim'] == 2:
    if "student_t" in param['dataset']:
        nx = 50
        xx = np.linspace(-30, 30, nx)
        yy = np.linspace(-30, 30, nx)
    else:
        nx = 40
        xx = np.linspace(-3, 9, nx)
        yy = np.linspace(-3, 9, nx)
    XX, YY = np.meshgrid(xx, yy)
    xx = np.concatenate((np.reshape(XX, (-1, 1)), np.reshape(YY, (-1, 1))), axis=1)
    xx = tf.constant(xx, dtype=tf.float32)
    phis = []

# Train ---------------------------------------------------------------
import matplotlib.pyplot as plt
import time

# -------- smoothing parameter h ----------------
# h is a fixed hyperparameter: --ramp_h, default 0.5.
if cvar_gate == 'ramp':
    _ramp_h_param = param.get('ramp_h', None)
    _ramp_h_val = 0.5 if _ramp_h_param is None else float(_ramp_h_param)
    if _ramp_h_val <= 0.0:
        raise ValueError("--ramp_h must be > 0 (default 0.5)")
    ramp_h_tf = tf.constant(_ramp_h_val, dtype=tf.float32)
    param['ramp_h_used'] = _ramp_h_val
    print("Ramp gate: h =", param['ramp_h_used'])

# Target CVaR on Q (computed once).  Hard gate: Rockafellar-Uryasev value.  Ramp gate:
# smoothed CVaR_h with the same h, so Delta_h compares like with like.
if cvar_gate == 'ramp':
    _g_Q = radius(Q)
    _q_h_Q = smoothed_quantile(_g_Q, ramp_h_tf)
    bcvar_target = tf.stop_gradient(smoothed_cvar(_q_h_Q, _g_Q, ramp_h_tf))
else:
    eta_Q_val = empirical_var_bar(Q, radius)
    bcvar_target = tf.stop_gradient(empirical_cvar(eta_Q_val, Q, radius))

t0 = time.time()
coef_f = 1
lam = float(param.get("lam", 10))
_lam_rule = param.get('lam_rule', None) or 'init_gap'
if _lam_rule not in ('fixed', 'gain', 'init_gap'):
    raise ValueError("lam_rule must be 'fixed', 'gain' or 'init_gap'")
param['lam_rule'] = _lam_rule
if _lam_rule == 'gain':
    # lam is specified through the dimensionless gain g = lam*lr_P, the only
    # quantity the Phase-1 CVaR dynamics depend on: Delta_{t+1} = (1 - 2g) Delta_t, tau = 1/(2g) iters,
    # stable and non-overshooting for g <= 1/2. g = 0.1 gives lam = 4e-3 at lr_P = 25.
    _lam_gain = float(param.get('lam_gain', 0.1) or 0.1)
    if not (0.0 < _lam_gain <= 0.5):
        raise ValueError("lam_gain must be in (0, 0.5]: g > 1/2 overshoots, g >= 1 diverges")
    lam = _lam_gain / float(param['lr_P'])
    param['lam_cli'] = param.get('lam', None)      # whatever was passed, for provenance
    param['lam'] = lam
    param['lam_gain'] = _lam_gain
    print("lam_rule=gain: lam = g*/lr_P = %.6g / %.6g = %.6g   (gain %.4g, gap decay factor per iter %.4g)"
          % (_lam_gain, float(param['lr_P']), lam, _lam_gain, abs(1.0 - 2.0 * _lam_gain)))
elif _lam_rule == 'init_gap':
    if cvar_gate == 'ramp':
        _g_P0 = radius(P)
        _c_P0 = smoothed_cvar(smoothed_quantile(_g_P0, ramp_h_tf), _g_P0, ramp_h_tf)
    else:
        _c_P0 = empirical_cvar(empirical_var_bar(P, radius), P, radius)
    _delta_init = float(bcvar_target.numpy()) - float(_c_P0.numpy())
    _lam_scale = float(param.get('lam_init_gap_scale', 1.0) or 1.0)
    # No-overshoot guard (parameter-free). The gap obeys Delta_{t+1} = (1 - 2*lam*lr_P) Delta_t
    # exactly while the gate is saturated, so one step moves the CVaR by 2*lam*lr_P*Delta.
    # Requiring a single step not to carry past the target gives 2*lam*lr_P <= 1, i.e.
    # lam <= 1/(2*lr_P). Combined with the init-gap rule:  lam = scale / max(|Delta_0|, 2*lr_P).
    # The bound introduces no new hyperparameter: the 2 is the same factor in the CVaR velocity
    # and lr_P is an existing run setting. Without it, a pretrain that already sits close to the
    # target (small |Delta_0|) yields lam*lr_P > 1 and a permanent limit cycle.
    _lam_ceiling_den = 2.0 * float(param['lr_P'])
    # Explicit cap: lam = min(scale/|Delta_0|, lam_init_gap_cap), default cap 0.01.
    # Implemented as a floor on the denominator (1/cap) so the printed form stays lam = scale/max(...).
    _lam_cap = float(param.get('lam_init_gap_cap', 0.01) or 0.01)
    _cap_den = _lam_scale / _lam_cap
    _den = max(abs(_delta_init), _lam_ceiling_den, _cap_den, 1e-12)
    lam = _lam_scale / _den
    param['lam_init_gap_cap'] = _lam_cap
    _lam_capped = bool(abs(_delta_init) < max(_lam_ceiling_den, _cap_den))
    param['lam_cli'] = param.get('lam', None)      # whatever was passed, for provenance
    param['lam'] = lam
    param['delta_init'] = _delta_init
    param['lam_no_overshoot_capped'] = _lam_capped
    print("lam_rule=init_gap: Delta_0 = %.6g (target CVaR %.6g, init CVaR %.6g) -> lam = min(%.6g/|Delta_0|, cap %.6g) = %.6g"
          % (_delta_init, float(bcvar_target.numpy()), float(_c_P0.numpy()), _lam_scale, _lam_cap, lam))
    if _lam_capped:
        print("[info] lam CAPPED: uncapped %.6g/|Delta_0| = %.6g; cap %.6g (2*lr_P floor %.6g also enforced)."
              % (_lam_scale, _lam_scale / max(abs(_delta_init), 1e-12), _lam_cap, 1.0 / _lam_ceiling_den))
    print("           gain lam*lr_P = %.4g (gap decay factor per iter = %.4g)"
          % (lam * float(param['lr_P']), abs(1.0 - 2.0 * lam * float(param['lr_P']))))

if param.get('no_cvar', False):
    coef_CVaR = 0.0
else:
    coef_CVaR = lam * _tail_mass_val

eta_P = tf.Variable(0.0, dtype=tf.float32)
if cvar_gate == 'ramp':
    eta_P.assign(smoothed_quantile(radius(P), ramp_h_tf))   # q_h tracker
else:
    eta_P.assign(empirical_var_bar(P, radius))

_ke_stop_streak = 0  # for --stop_ke_tol early stop (KE-dead runs)
_gated_kl_ke_history = []
_tail_ke_median_history = []
_ke_median_history = []
_med_W = int(param.get('stop_median_window', 1000) or 1000)
_tkm_enabled = param.get('stop_tail_ke_median', None) is not None
_kem_enabled = param.get('stop_ke_median', None) is not None
if _tkm_enabled:
    if coef_CVaR == 0.0:
        raise ValueError("--stop_tail_ke_median needs an active CVaR term (no pushed set otherwise)")
    _tkm_tol = float(param['stop_tail_ke_median'])
    _gkl_k = max(1, int(round(float(_tail_mass_val) * int(P.shape[0]))))   # (1-alpha)*M
    param['stop_tail_ke_median_k'] = _gkl_k
    print("stop_tail_ke_median: stop when median per-particle KL KE of the k=%d pushed "
          "particles over the last %d iters < %g" % (_gkl_k, _med_W, _tkm_tol))
if _kem_enabled:
    _kem_tol = float(param['stop_ke_median'])
    print("stop_ke_median: stop when median KE_P over the last %d iters < %g" % (_med_W, _kem_tol))
param['stop_median_window'] = _med_W
_any_stop = _tkm_enabled or _kem_enabled

def _save_snapshot():
    """Append the current particle state (and velocity snapshots) to the
    saved histories. Called every save_iter, and once more at an early stop so
    trajectories[-1] is the true final state."""
    trajectories.append(P.numpy() * rescale_factor)
    if param['unseen'] == True:
        unseen_trajectories.append(P_unseen.numpy() * rescale_factor)
    if np.prod(param['N_dim']) < 500:
        vectorfields.append(dP.numpy())
        kl_vf_snapshots.append(vf_kl.numpy())
        if vf_cvar is not None:
            cvar_vf_snapshots.append(vf_cvar.numpy())
        else:
            cvar_vf_snapshots.append(np.zeros_like(vf_kl.numpy()))

for it in range(1, param['epochs'] + 1):
    parameters, current_loss, dW_norm = train_disc(
        parameters, phi, P, Q, lr_phi, param['epochs_phi'],
        loss_par, NN_par, data_par, param['optimizer'], print_vals=True
    )

    if param['calc_Wasserstein1'] == True:
        parameters2, current_wass1, _ = train_wasserstein1(
            parameters2, phi2, P, Q, lr_phi, param['epochs_phi'],
            NN_par2, data_par, param['optimizer'], print_vals=True
        )

    vf_kl = coef_f * calc_vectorfield(phi, P, parameters, NN_par, loss_par, data_par)
    vf = vf_kl

    if _tkm_enabled:
        # pushed set = k largest radii of the CURRENT particles (saturated gate ~1);
        # per-particle KL kinetic energy on it, then its median over the last W
        _r_np = radius(P).numpy()
        _top_idx = np.argpartition(_r_np, -_gkl_k)[-_gkl_k:]
        _vkl_top = vf_kl.numpy()[_top_idx]
        _gated_kl_ke_history.append(float(0.5 * np.mean(np.sum(_vkl_top ** 2, axis=1))))
        _tail_ke_median_history.append(float(np.median(_gated_kl_ke_history[-_med_W:]))
                                       if len(_gated_kl_ke_history) >= _med_W else float('nan'))

    if coef_CVaR != 0.0:                                               # joint_radial
        if cvar_gate == 'ramp':
            eta_P.assign(smoothed_quantile(radius(P), ramp_h_tf))      # smoothed quantile q_h
            dP_band, bcvar_current = calc_cvar_velocity_ramp(
                P, eta_P, bcvar_target, ramp_h_tf
            )
        else:
            eta_P.assign(empirical_var_bar(P, radius))
            dP_band, bcvar_current = calc_cvar_velocity_paper(
                P, eta_P, bcvar_target
            )
        vf_cvar = coef_CVaR * dP_band
        vf = vf + vf_cvar
        current_loss_CVaR = bcvar_current
        y = eta_P
        eta_history.append(float(eta_P.numpy()))
    else:
        current_loss_CVaR = tf.constant(float("nan"), tf.float32)
        vf_cvar = None
        eta_history.append(float("nan"))

    dPs.append(vf)
    P, dPs, dP = solve_ode(P, lr_P, dPs, param['ode_solver'], aux_params)

    if param['mobility'] == 'bounded':
        P.assign(bounded_relu(P))

    if param['unseen'] == True:
        unseen_dPs.append(coef_f * calc_vectorfield(phi, P_unseen, parameters, NN_par, loss_par, data_par))
        P_unseen, unseen_dPs, dP_unseen = solve_ode(P_unseen, lr_P, unseen_dPs, param['ode_solver'], aux_params)
        if param['mobility'] == 'bounded':
            P_unseen.assign(bounded_relu(P_unseen))

    lr_Ps.append(lr_P.numpy())

    bcvar_diff_sq = tf.square(current_loss_CVaR - bcvar_target)
    QoIs.append(float(bcvar_diff_sq.numpy()))
    cvar_signed_diff_history.append(float((current_loss_CVaR - bcvar_target).numpy()))

    divergences.append(current_loss)
    KE_P = calc_ke(dP, param['N_samples_P'])
    KE_Ps.append(KE_P)
    comp_times.append(time.time() - t0)

    # Optional KE-dead early stop (--stop_ke_tol; default off). If the
    # kinetic energy stays at/below the tolerance for --stop_ke_window
    # consecutive iterations, the particles are frozen and the remaining
    # iterations cannot change anything: stop and save.
    if param.get('stop_ke_tol', None) is not None:
        _ke_stop_streak = _ke_stop_streak + 1 if float(KE_P) <= param['stop_ke_tol'] else 0
        if _ke_stop_streak >= param.get('stop_ke_window', 500):
            print('[early-stop] kinetic energy <= %g for %d consecutive iters '
                  '(iter %d): algorithm died, stopping.'
                  % (param['stop_ke_tol'], param['stop_ke_window'], it))
            _save_snapshot()                       # final state -> trajectories[-1]
            param['stop_snapshot_iter'] = it
            break

    # Stillness stop = AND of the enabled conditions (both medians over the last W):
    #   --stop_tail_ke_median : median per-particle KL KE of the k pushed particles < tol
    #   --stop_ke_median      : median total KE_P < tol
    # Metrics cannot improve further on held-out data once both hold: stop and save.
    if _kem_enabled:
        _ke_median_history.append(float(np.median(KE_Ps[-_med_W:])) if len(KE_Ps) >= _med_W
                                  else float('nan'))
    if _any_stop:
        _ok_t = (not _tkm_enabled) or (np.isfinite(_tail_ke_median_history[-1])
                                       and _tail_ke_median_history[-1] < _tkm_tol)
        _ok_k = (not _kem_enabled) or (np.isfinite(_ke_median_history[-1])
                                       and _ke_median_history[-1] < _kem_tol)
        if _ok_t and _ok_k:
            _msg = []
            if _tkm_enabled:
                _msg.append('median KL KE of top-%d over last %d iters = %.3e < %g'
                            % (_gkl_k, _med_W, _tail_ke_median_history[-1], _tkm_tol))
            if _kem_enabled:
                _msg.append('median KE_P over last %d iters = %.3e < %g'
                            % (_med_W, _ke_median_history[-1], _kem_tol))
            print('[early-stop] ' + '; '.join(_msg) + ' (iter %d): particles still, stopping.' % it)
            param['stop_still_iter'] = it
            _save_snapshot()                       # final state -> trajectories[-1]
            param['stop_snapshot_iter'] = it
            break

    if param['calc_Wasserstein1'] == True:
        wasserstein1s.append(current_wass1)

    if param['epochs'] <= 100 or it % param['save_iter'] == 0:
        _save_snapshot()

    if it % (param['epochs'] / 10) == 0:
        display_msg = (
            'iter %6d / time %.2f: CVaR_Diff_Square = %.10f, proximal = %.10f, norm of dW = %.2f, '
            'kinetic energy of P = %.10f, average learning rate for P = %.6f, eta_bar=%.3f'
            % (it, comp_times[-1], bcvar_diff_sq.numpy(), current_loss, dW_norm, KE_P,
               tf.math.reduce_mean(lr_P).numpy(), y.numpy())
        )
        if len(FIDs) > 0:
            display_msg = display_msg + ', FID = %.3f' % FIDs[-1]
        print(display_msg)

        if param['plot_intermediate_result'] == True:
            data = {
                'trajectories': trajectories,
                'QoIs': QoIs,
                'divergences': divergences,
                'wasserstein1s': wasserstein1s,
                'KE_Ps': KE_Ps,
                'comp_times': comp_times,
                'FIDs': FIDs,
                'X_': X_,
                'Y_': Y_,
                'X_label': X_label,
                'Y_label': Y_label,
                'dt': lr_Ps,
                'dataset': param['dataset'],
                'r_param': r_param,
                'vectorfields': vectorfields,
                'save_iter': param['save_iter']
            }
            if param['N_dim'] == 2:
                data.update({'phi': phi, 'W': W, 'b': b, 'NN_par': NN_par})
            plot_result(filename, intermediate=True, epochs=it, iter_nos=None, data=data, show=False)

        if np.prod(param['N_dim']) == 1:
            zz = phi(xx, None, W, b, NN_par).numpy()
            zz = np.reshape(zz, -1)
            phis.append(zz)

total_time = time.time() - t0
print(f'total time {total_time:.3f}s')
print("FINAL_DF=", float(divergences[-1]) if len(divergences) > 0 else float("nan"))
print("FINAL_QoI=", float(QoIs[-1]) if len(QoIs) > 0 else float("nan"))

# CVaR report on generated distribution (final P) for selected (a, beta)
report_betas = param.get('report_beta_levels', None)
if report_betas is None:
    report_betas = [_alpha_val]
report_ab_weights = param.get('report_ab_weights', None)
if report_ab_weights is None:
    report_ab_weights = [_ab_weight_val]

risk_generated = radius(P).numpy().astype(np.float64)
risk_target = radius(Q).numpy().astype(np.float64)

cvar_report = []
for beta_report in report_betas:
    if not (0.0 < beta_report < 1.0):
        continue

    q_gen, cvar_gen = empirical_var_cvar_from_risk(risk_generated, beta_report)
    q_tgt_emp, cvar_tgt_emp = empirical_var_cvar_from_risk(risk_target, beta_report)

    for abw_report in report_ab_weights:
        if not (0.0 <= abw_report <= beta_report):
            continue
        a_report = abw_report / beta_report
        _, _, bcvar_gen = empirical_bcvar_from_risk(risk_generated, beta_report, a_report)
        _, _, bcvar_tgt_emp = empirical_bcvar_from_risk(risk_target, beta_report, a_report)

        row = {
            'beta': float(beta_report),
            'a': float(a_report),
            'ab_weight': float(abw_report),
            'gen_var_beta': float(q_gen),
            'gen_cvar_beta': float(cvar_gen),
            'gen_bcvar': float(bcvar_gen),
            'target_var_beta_empirical': float(q_tgt_emp),
            'target_cvar_beta_empirical': float(cvar_tgt_emp),
            # None since the analytical reference was retired with the
            # Refine_heavytail datasets; keys kept for pickle-schema stability
            'target_cvar_beta_true': None,
            'target_bcvar_empirical': float(bcvar_tgt_emp),
            'gen_cvar_minus_true': None
        }
        cvar_report.append(row)

if '1D' in p.dataset:
    import matplotlib.pyplot as plt
    for i, yy in enumerate(phis):
        color_gradient = (
            max(-25/4*(i/9)**2+0.85, 0),
            max(-25/4*(i/9-1/2)**2+0.85, 0),
            max(-25/4*(i/9-1)**2+0.85, 0)
        )
        plt.plot(xx, yy, label='t=%.2f' % ((i+1)*param['epochs']/10*param['lr_P']), color=color_gradient)
    plt.legend()
    plt.title(r'$\phi_t$')
    f = filename.split('.pickle')
    plt.savefig(f[0] + "-phis.png")
    plt.show()

# Save result ------------------------------------------------------
import pickle
if param['N_dim'] == 1:
    X_ = np.concatenate((X_, np.zeros(shape=X_.shape)), axis=1)
    Y_ = np.concatenate((Y_, np.zeros(shape=Y_.shape)), axis=1)

    trajectories = [np.concatenate((x, np.zeros(shape=x.shape)), axis=1) for x in trajectories]
    vectorfields = [np.concatenate((x, np.zeros(shape=x.shape)), axis=1) for x in vectorfields]

if param['L'] == None:
    param['L'] = 'inf'
param.update({'X_': X_, 'Y_': Y_, 'lr_Ps': lr_Ps})
param['paper_lambda'] = lam  # effective CVaR force coefficient (1/(1-alpha) absorbed)
param['alpha_paper']  = _alpha_val

result = {
    'trajectories': trajectories,
    'vectorfields': vectorfields,
    'QoIs': QoIs,
    'divergences': divergences,
    'KE_Ps': KE_Ps,
    'FIDs': FIDs,
    'wasserstein1s': wasserstein1s,
    'comp_times': comp_times,
    'cvar_report': cvar_report,
    'eta_history': eta_history,
    'kl_vf_snapshots': kl_vf_snapshots,
    'cvar_vf_snapshots': cvar_vf_snapshots,
    'cvar_signed_diff_history': cvar_signed_diff_history,
    'gated_kl_ke_history': _gated_kl_ke_history,         # per-iter per-particle KL KE of the k pushed (--stop_tail_ke_median)
    'tail_ke_median_history': _tail_ke_median_history,   # its median over the last W (nan for it<W)
    'ke_median_history': _ke_median_history,             # per-iter median KE_P over last W (--stop_ke_median; nan for it<W)
}
if param['unseen'] == True:
    result.update({'unseen_trajectories': unseen_trajectories})

if param['dataset'] in ['Labeled_disease']:
    np.savetxt(
        main_dir + "/data/gene_expression_example/GPL570/" + param['dataset'] + '/output_norm_dataset_dim_%d.csv'
        % param['N_dim'],
        trajectories[-1],
        delimiter=","
    )

with open(filename, "wb") as fw:
    pickle.dump([param, result], fw)
print("Results saved at:", filename)

if param['plot_result'] == True:
    try:
        plot_result(filename, intermediate=False, show=False)
    except Exception as _plot_err:  # plotting assumes the full epoch count;
        # an early-stopped run (--stop_ke_tol / the two-median stillness stop) has fewer snapshots. The
        # result pickle is already saved above, so a plot failure is benign.
        print('[warn] final plot_result failed (early stop?):', _plot_err)
