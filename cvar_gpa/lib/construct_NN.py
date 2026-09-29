import tensorflow as tf
from numpy import pi
from functools import partial

class UndefinedError(Exception):
    """Exception raised for errors in the undefined Neural network topology.
    
    Attributes:
        NN_model -- model with undefined network topology
    """
    
    def __init__(self, undefined_NN_model):
        self.undefined = undefined_NN_model
    
    def __str__(self):
        if 'fnn' in self.undefined:
            return('N_fnn_layers is undefined for the fnn.')
        elif 'cnn' in self.undefined:
            return('N_cnn_layers is undefined for the cnn.')

def check_nn_topology(NN_model, N_fnn_layers, N_cnn_layers, N_dim, activation_ftn):
    undefined = ''
    if 'cnn' in NN_model: # cnn or cnn_fnn
        if N_cnn_layers == None:
            undefined = undefined + 'cnn'
        else: 
            N_cnn_layers = [N_dim]+N_cnn_layers
            
        x, y = (N_dim[0], N_dim[1])
        for i in N_cnn_layers[1:]:
            x, y = (x//2+x%2, y//2+y%2)
        
        N_fnn_layers = [N_cnn_layers[-1]*x*y]+N_fnn_layers
        if 'leaky_relu' not in activation_ftn:
            activation_ftn = ['leaky_relu'] + activation_ftn
    
    if NN_model == 'fnn':
        if N_fnn_layers == None:
            undefined = undefined + 'fnn'  
        else:
            N_fnn_layers = [N_dim]+N_fnn_layers
            
    if undefined != '':
        raise UndefinedError(undefined)
        
    return N_fnn_layers, N_cnn_layers, activation_ftn
    
# ---------------------------------------------------------
def spectral_normalization(W, norm_=1):
    if tf.norm(W, 2) < 1e-6:
        print('WARNING: Norm of W is too small')
    W.assign(tf.math.scalar_mul(norm_/tf.norm(W, 2), W))
    
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1.0 / tf.sqrt(in_dim / 2.)
    return tf.random.normal(shape=size, stddev=xavier_stddev)

def initialize_W(layers, N_conditions):
    W_init, NN_W, NN_b=[], [], []
    
    if layers == None:
        return NN_W, NN_b
        
    num_layers = len(layers)
    if type(layers[0])==int: # fnn
        for l in range(0,num_layers-1): 
            b = tf.Variable(tf.zeros([1,layers[l+1]]), dtype=tf.float32)
            NN_b.append(b)
            W_init.append(xavier_init(size=[layers[l], layers[l+1]]))
        b = tf.Variable(tf.zeros([1,N_conditions]), dtype=tf.float32)
        NN_b.append(b)
        W_init.append(xavier_init(size=[layers[-1], N_conditions]))
    else: # cnn
        # filter shape: [filter_height, filter_width, in_channels, out_channels]
        NN_b = [tf.constant(1, dtype=tf.float32)]*(num_layers-1)
        #W_init.append(tf.random.normal(shape=[5,5, layers[0][-1], layers[1]],  stddev=0.01))
        W_init.append(xavier_init(size=[5,5, layers[0][-1], layers[1]]))
        for l in range(1,num_layers-1):
            #W_init.append(tf.random.normal(shape=[5,5, layers[l], layers[l+1]],  stddev=0.01))
            W_init.append(xavier_init(size=[5,5, layers[l], layers[l+1]]))
      
    for l, w in enumerate(W_init):
        W = tf.Variable(w, dtype=tf.float32)
        NN_W.append(W)
     
    return NN_W, NN_b

def initialize_NN(NN_par):
    [W_cnn,b_cnn]=initialize_W(NN_par['N_cnn_layers'], NN_par['N_conditions'])
    [W_fnn,b_fnn]=initialize_W(NN_par['N_fnn_layers'], NN_par['N_conditions'])
    
    W = W_cnn+W_fnn
    if NN_par['constraint'] == 'hard' and NN_par['L'] != None:
        for w in W:
            w = spectral_normalization(w, NN_par['L']**(1/len(W)))
    b = b_cnn+b_fnn
    
    return W, b

# ----------------------------------------------------------------
def bounded_relu(x):
    return tf.math.subtract(tf.nn.relu(x), tf.nn.relu(x-1))
    
def bounded_elu(x):
    return tf.math.subtract(tf.nn.elu(x), tf.nn.elu(x-1))
    
def mollified_relu_poly3(x):
    return tf.cast(x<0, tf.float32)*0 + tf.cast(x>=2, tf.float32)*(x-1) + tf.cast(tf.logical_and(x>=0, x<2), tf.float32) *tf.math.multiply(tf.math.pow(x,4), tf.math.pow(x,2)-6*x+10)/32
    
def mollified_relu_cos3(x, eps):
    if eps == None:
        raise NameError("Determine -eps [float]")
    return tf.cast(x<0, tf.float32)*0 + tf.cast(x>=2*eps, tf.float32)*(x-eps) + tf.cast(tf.logical_and(x>=0, x<2*eps), tf.float32) *(tf.math.pow(x,2)/(4*eps)+eps*(tf.math.cos(pi*x/eps)-1)/(2*pi**2))
    
def mollified_relu_cos3_shift(x, eps):
    if eps == None:
        raise NameError("Determine -eps [float]")
    shift = eps/2
    return mollified_relu3_2(x+shift)
    
def determine_activation_ftn(ftn_name, eps):
    # activation function that is comatible with Spectral normalization to enforce Lipschitz continuity
    
    # unbounded
    if ftn_name == 'relu':
        activation_ftn = tf.nn.relu
    elif ftn_name == 'leaky_relu':
        activation_ftn = tf.nn.leaky_relu                       # tf default slope 0.2
    elif ftn_name == 'leaky_relu_001':
        # slope 0.01: keeps units alive (no exact-zero gradient, so no dying-ReLU
        # collapse) while restoring most of ReLU's shielding of far-tail particles from KL pull.
        activation_ftn = partial(tf.nn.leaky_relu, alpha=0.01)
    elif ftn_name == 'mollified_relu_poly3':
        activation_ftn = mollified_relu_poly3
    elif ftn_name == 'mollified_relu_cos3':
        activation_ftn = partial(mollified_relu_cos3, eps = eps)
    elif ftn_name == 'mollified_relu_cos3_shift':
        activation_ftn = partial(mollified_relu_cos3_shift, eps = eps)
    elif ftn_name == 'softplus':
        activation_ftn = tf.math.softplus
    elif ftn_name == 'elu':
        activation_ftn = tf.nn.elu
    elif ftn_name == 'abs':
        activation_ftn = tf.math.abs
    elif ftn_name == 'linear':
        activation_ftn = tf.identity
        
    # bounded
    elif ftn_name == 'bounded_relu':
        activation_ftn = bounded_relu
    elif ftn_name == 'bounded_elu':
        activation_ftn = bounded_elu
        
    elif ftn_name == 'sigmoid': # not compatible with SN
        activation_ftn = tf.nn.sigmoid
        
    return activation_ftn


# --------------------------------------------------------------- 
# NN structure for phi

def fnn(x, x_label, W, b, NN_par):
    num_layers = len(W)
    activation_ftn = determine_activation_ftn(NN_par['activation_ftn'][0], NN_par['eps'])
    try:
        activation_ftn2 = determine_activation_ftn(NN_par['activation_ftn'][1], NN_par['eps'])
    except:
        activation_ftn2 = determine_activation_ftn(NN_par['activation_ftn'][0], NN_par['eps'])
        
    h = x
    for l in range(0,num_layers-2):
        h = activation_ftn(tf.add(tf.matmul(h, W[l]), b[l]))
    h = activation_ftn2(tf.add(tf.matmul(h, W[num_layers-2]), b[num_layers-2]))
    out=tf.add(tf.matmul(h, W[-1]), b[-1])
    
    if NN_par['N_conditions'] > 1:
        out = tf.math.reduce_sum(tf.math.multiply(out, x_label), axis=1, keepdims=True)
    
    return out


def cnn_fnn(x, x_label, W, b, NN_par):
    num_layers = len(NN_par['N_cnn_layers'])-1
    activation_ftn = determine_activation_ftn(NN_par['activation_ftn'][0], NN_par['eps'])
    N_samples = x.shape[0]
    
    # test
    '''
    if NN_par['constraint'] == 'hard' and NN_par['L'] != None: # spectral normalization
        for w in W:
            spectral_normalization(w, NN_par['L']**(1/len(W)))
    '''
        
    h = x
    for l in range(num_layers):
        h = activation_ftn( tf.nn.conv2d(input=h,filters=W[l],strides=(2,2),padding='SAME'))
        #h = tf.nn.dropout(h, rate=0.3)
        
    # flatten and dense layers
    num_layers2 = len(NN_par['N_fnn_layers'])
    activation_ftn2 = determine_activation_ftn(NN_par['activation_ftn'][1], NN_par['eps'])
    try:
        activation_ftn3 = determine_activation_ftn(NN_par['activation_ftn'][2], NN_par['eps'])
    except:
        activation_ftn3 = determine_activation_ftn(NN_par['activation_ftn'][1], NN_par['eps'])

    h = tf.transpose(h, perm=[3, 2,1,0])
    h = tf.transpose(tf.reshape(h, [-1,N_samples]))

    for l in range(num_layers,num_layers+num_layers2-2):
        h = activation_ftn2(tf.add(tf.matmul(h, W[l]), b[l]))
    h = activation_ftn2(tf.add(tf.matmul(h, W[num_layers+num_layers2-2]), b[num_layers+num_layers2-2]))
            
    out=tf.add(tf.matmul(h, W[-1]), b[-1])
    if NN_par['N_conditions'] > 1:
        out = tf.math.reduce_sum(tf.math.multiply(out, x_label), axis=1, keepdims=True)
        
    out = tf.expand_dims(tf.expand_dims(out, axis=-1), axis=-1)
    return out
    

# ---------------------------------------------------------------- 
def model(NN_par):
    if NN_par['NN_model'] == 'fnn':
        return fnn
    elif 'cnn' in NN_par['NN_model']:
        return cnn_fnn
