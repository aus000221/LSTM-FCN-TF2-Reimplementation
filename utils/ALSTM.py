import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import activations
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import RNN
#from tensorflow.keras import backend as K


# wirtten by zz3128 following the structure given in the code base https://github.com/houshd/LSTM-FCN
class AttentionLSTMCell(Layer):
    '''
    Methods: 
    - __init__: initialize the model
    - build   : build the parameters
    - call    : implement the forward pass
    - get_config: for saving and loading model purpose
    '''

    def __init__(self, units,
             activation='tanh',
             recurrent_activation='hard_sigmoid',
             attention_activation='tanh',
             kernel_initializer='glorot_uniform',
             recurrent_initializer='orthogonal',
             attention_initializer='orthogonal',
             bias_initializer= 'zeros',
             **kwargs):
        ''' Initialize the model '''
        
        super(AttentionLSTMCell, self).__init__(**kwargs) #initialized as Layer class

        # Number of units (dimensions) for ALSTM
        self.units = units
        
        # Activation functions initializer
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.attention_activation = activations.get(attention_activation)
        
        # Weight initializers
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.attention_initializer = initializers.get(attention_initializer)
        self.bias_initializer =initializers.get(bias_initializer)

        # For RNN layer
        self.state_size = (units, units)


    def build(self, input_shape):
        '''
        Build the parameters

        :param input_shape: the shape of "inputs" of "call" method [N, D] ,N = batch size
        '''
        
        self.D = input_shape[-1]
        
        ## add LSTM kernels
        self.kernel = self.add_weight(shape=(self.D, self.units * 4),
                            name='kernel', #W = [Wi,Wf,Wcell,Wo]
                            initializer=self.kernel_initializer)
        
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units * 4),
                                   name='recurrent_kernel', #Wh
                                   initializer=self.recurrent_initializer)
        
        ## add attention kernel
        self.attention_kernel = self.add_weight(shape=(self.D, self.units * 4),
                                   name='attention_kernel', #Wa = [Wai,Waf,Wacell,Wao]
                                   initializer=self.attention_initializer)

        ## add attention weights
        ## weights for attention model
        self.attention_weights = self.add_weight(shape=(self.D, self.units),
                                    name='attention_U', #Ua
                                    initializer=self.attention_initializer)

        self.attention_recurrent_weights = self.add_weight(shape=(self.units, self.units),
                                           name='attention_recurrent', #Wah
                                           initializer=self.recurrent_initializer)
        
        ## add bias
        ## set the initialized forget gate bias = 1 to remember more early stage information
        def _bias_initializer(shape, dtype=float,*args, **kwargs):
            return tf.concat([self.bias_initializer((self.units,), dtype=dtype,*args, **kwargs),
                         initializers.Ones()((self.units,), dtype=dtype,*args, **kwargs),
                         self.bias_initializer((self.units * 2,), dtype=dtype,*args, **kwargs)], 
                         axis=0)
         
        self.bias = self.add_weight(shape=(self.units * 4,),
                                name='bias', # b = [bi,bf,bcell,bo]
                                initializer=_bias_initializer)
        
        self.attention_bias = self.add_weight(shape=(self.units,),
                                  name='attention_bias', #ba
                                  initializer=self.bias_initializer)

        self.attention_recurrent_bias = self.add_weight(shape=(self.units, 1),
                                        name='attention_v', #bar
                                        initializer=self.bias_initializer)

        # Set build flag to true
        self.built = True


    def call(self, inputs, states):
        '''
        Forward pass for LSTM cell. 

        :param inputs: cell inputs of one time step, 
            a tf.Tensor of shape [batch_size,D]
        :param states: cell states from last time step, 
            a tuple of (h_[t-1], c_[t-1])

        Return
        : hidden states & a tuple of new hidden states and cell states
        '''
        
        batch_size = inputs.shape[0]
        
        h = states[0] #h_[t-1]
        c = states[1] #c_[t-1]
        
        ## alignment model
        ## etj = align(h_[t-1],x_t) = ACT_FUNC(hWa + xUa +ba)bar
        ## the following calculate et = [et1,et2,...,etT]
        h_att = tf.tile(h[:, tf.newaxis, :], multiples=[1, self.T, 1]) # N*T*K
        
        x_att = inputs
        x_att = tf.nn.bias_add(x_att@self.attention_weights, self.attention_bias) # N*K
        x_att = tf.reshape(x_att, [-1, self.T, self.units]) #N*T*K
        x_att.set_shape([None, None, self.units]) #N*T*K
    
        et = self.attention_activation(h_att@self.attention_recurrent_weights + x_att) # N*T*K
        et = et@self.attention_recurrent_bias # N*T*1
        et = tf.squeeze(et,axis = 2) # N*T
        
        ## calculate αt = [αt1,αt2,...,αtT]
        alpha_t = tf.exp(et) #N*T
        alpha_t /= tf.reduce_sum(alpha_t, axis=1, keepdims=True) #N*T
        alpha_r = tf.tile(alpha_t[:,:,tf.newaxis], multiples=[1, 1, self.D])#N*T*D
        
        ## calculate context vector
        context_sequence = tf.multiply(inputs, alpha_r) # N*T*D
        context = tf.reduce_sum(context_sequence, axis=1) #N*D
        
        ## LSTM calculate
        Z = inputs@self.kernel + h@self.recurrent_kernel + context@self.attention_kernel 
        Z = tf.nn.bias_add(Z,self.bias)
        
        f = self.recurrent_activation(Z[:,0:self.units])
        i = self.recurrent_activation(Z[:,self.units:self.units*2])
        c_ = self.activation(Z[:,self.units*2:self.units*3])
        o = self.recurrent_activation(Z[:,self.units*3:self.units*4])
        
        c = tf.multiply(f,c) + tf.multiply(i,c_)
        h = tf.multiply(o,self.activation(c))

        return h, [h, c]
    
    def get_config(self):
        base_config = super(AttentionLSTMCell, self).get_config()
        return base_config


class AttentionLSTM(RNN):

    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 attention_activation='tanh',
                 #use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 attention_initializer='orthogonal',
                 bias_initializer='zeros',
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):

        '''
        Initialize the model. 
        '''
        ## init for get_config purpose
        self.units = units
        
        # Activation functions initializer
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.attention_activation = activations.get(attention_activation)
        
        # Weight initializers
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.attention_initializer = initializers.get(attention_initializer)
        self.bias_initializer =initializers.get(bias_initializer)
        
        
        cell = AttentionLSTMCell(units,
                         activation=activation,
                         recurrent_activation=recurrent_activation,
                         attention_activation=attention_activation,
                         kernel_initializer=kernel_initializer,
                         recurrent_initializer=recurrent_initializer,
                         attention_initializer=attention_initializer,
                         bias_initializer=bias_initializer,
                         )
        
        
        super(AttentionLSTM, self).__init__(cell,
                                            return_sequences=return_sequences,
                                            return_state=return_state,
                                            go_backwards=go_backwards,
                                            stateful=stateful,
                                            unroll=unroll,
                                             **kwargs)

    def build(self, input_shape):
        
        self.cell.T = input_shape[1] #time step
        self.cell.build(input_shape)  

    def call(self, inputs, mask=None, training=None, initial_state=None):
        return super(AttentionLSTM, self).call(inputs,
                                   mask=mask,
                                   training=training,
                                   initial_state=initial_state)
    
    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'attention_activation': activations.serialize(self.attention_activation),
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'attention_initializer': initializers.serialize(self.attention_initializer)}
        base_config = super(AttentionLSTM, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))
    

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config and config['implementation'] == 0:
            config['implementation'] = 1
        return cls(**config)


