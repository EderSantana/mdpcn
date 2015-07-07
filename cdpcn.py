__authors__ = "Eder Santana"
__copyright__ = "Copyright 2014-2015, University of Florida"
__credits__ = "Eder Santana"
__license__ = "3-clause BSD"
__maintainer__ = "Eder Santana"

import logging
logger = logging.getLogger(__name__)

from pylearn2.space import VectorSpace, Conv2DSpace
from pylearn2.models.mlp import Layer, Linear, ConvElemwise, MLP
from theano.tensor.signal.downsample import max_pool_2d as max_pool
from pylearn2.utils import sharedX
from pylearn2.utils import wraps
from pylearn2.space import Space, CompositeSpace
from pylearn2.utils import wraps
from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.linear import conv2d

import functools
import theano
from theano import tensor as T
from theano.printing import Print
from theano.compat.python2x import OrderedDict
from theano.gof.op import get_debug_values
import top
import numpy as np
from pylearn2.models.mlp import IdentityConvNonlinearity, RectifierConvNonlinearity

class SparseCodingLayer(Linear):
    
    def __init__(self, batch_size, fprop_code=True, lr=.01, n_steps=10, lbda=0, top_most=False, 
            nonlinearity=RectifierConvNonlinearity(),*args, **kwargs):
        '''
        Compiled version: the sparse code is calulated using 'top' and is not just simbolic.
        Parameters for the optimization/feedforward operation:
        lr      : learning rate
        n_steps : number of steps or uptades of the hidden code
        truncate: truncate the gradient after this number (default -1 which means do not truncate)
        '''
        super(SparseCodingLayer, self).__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.fprop_code = fprop_code
        self.n_steps = n_steps
        self.lr = lr
        self.lbda = lbda
        self.top_most = top_most
        self.nonlin = nonlinearity

    @wraps(Linear.set_input_space)
    def set_input_space(self, space):
        
        self.input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)

        if self.fprop_code==True:
            self.output_space = VectorSpace(self.dim)
        else:
            self.output_space = VectorSpace(self.input_dim)

        rng = self.mlp.rng
        W = rng.randn(self.input_dim, self.dim)
        self.W = sharedX(W.T, self.layer_name + '_W')
        self.transformer = MatrixMul(self.W)
        self.W, = self.transformer.get_params()
        b = np.zeros((self.input_dim,))
        self.b = sharedX(b, self.layer_name + '_b') # We need both to pass input_dim valid
        X = .001 * rng.randn(self.batch_size, self.dim)
        self.X = sharedX(X, self.layer_name + '_X')
        S = rng.normal(0, .001, size=(self.batch_size, self.input_dim))
        self.S = sharedX(S, self.layer_name + '_S')
        self._params = [self.W, self.b]
        #self.state_below = T.zeros((self.batch_size, self.input_dim))
        
        cost = self.get_local_cost()
        self.opt = top.Optimizer(self.X, cost,  
                                 method='rmsprop', 
                                 learning_rate=self.lr, momentum=.9)

        self._reconstruction = theano.function([], T.dot(self.X, self.W))
    
    def get_local_cost(self):
        er = T.sqr(self.S - T.dot(self.X, self.W)).sum()
        l1 = T.sqrt(T.sqr(self.X) + 1e-6).sum()
        top_down = self.get_top_down_flow()
        return er + .1 * l1 + top_down
    
    def update_top_state(self, state_above=None):
        if self.lbda is not 0:
            assert state_above is not None
            self.top_flow.set_value(state_above)     
    
    def get_nonlin_output(self):
        return self.nonlin(self.X)

    def get_top_down_flow(self):
        if self.lbda == 0:
            rval = 0.
        elif self.top_flow == True:
            rval = (self.lbda * (self.top_flow - self.X)**2).sum()
        else:
            out = self.get_nonlin_output()
            rval = (self.lbda * (self.top_flow - out)**2).sum()

        return rval

    def _renormW(self):
        A = self.W.get_value(borrow=True)
        A = np.dot(A.T, np.diag(1./np.sqrt(np.sum(A**2, axis=1)))).T
        self.W.set_value( A )
  
    def get_reconstruction(self):
        return self._reconstruction()

    def get_sparse_code(self, state_below):

        # Renorm W
        self._renormW()

        if hasattr(state_below, 'get_value'):
            #print '!!!! state_below does have get_value'
            self.S.set_value(state_below.get_value(borrow=True))
            self.opt.run(self.n_steps) 
                         

        if isinstance(state_below, np.ndarray):
            self.S.set_value(state_below.astype('float32'))
            self.opt.run(self.n_steps) #, 
            #np.arange(self.batch_size))


        return self.X

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        
        self._renormW()
        rval = self.get_sparse_code(state_below)
        if self.fprop_code == True:
            #rval = T.switch(rval > 0., rval, 0.)
            rval = self.nonlin.apply(rval)
        else:
            # Fprops the filtered input instead
            rval = T.dot(rval, self.W)
        
        return rval
    
    @wraps(Layer.get_params)
    def get_params(self):
        return self.W

    @functools.wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                      state=None, targets=None):

        #sc = abs(self.Xout).sum() #Get last local_error get_local_error()
        #le = self.local_reconstruction_error 
        W, = self.transformer.get_params()

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        row_norms_min = row_norms.min()
        row_norms_min.__doc__ = ("The smallest norm of any row of the "
                                 "weight matrix W. This is a measure of the "
                                 "least influence any visible unit has.")
        '''
        rval = OrderedDict([('row_norms_min',  row_norms_min),
                            ('row_norms_mean', row_norms.mean()),
                            ('row_norms_max',  row_norms.max()),
                            ('col_norms_min',  col_norms.min()),
                            ('col_norms_mean', col_norms.mean()),
                            ('col_norms_max',  col_norms.max())])#,
                            #('sparse_code_l1_norm', sc.mean())])
        '''
        rval = OrderedDict()

        
        if False:
            #(state is not None) or (state_below is not None):
            if state is None:
                state = self.fprop(state_below)

            P = state
            #if self.pool_size == 1:
            vars_and_prefixes = [(P, '')]
            #else:
            #    vars_and_prefixes = [(P, 'p_')]

            for var, prefix in vars_and_prefixes:
                v_max = var.max(axis=0)
                v_min = var.min(axis=0)
                v_mean = var.mean(axis=0)
                v_range = v_max - v_min

                # max_x.mean_u is "the mean over *u*nits of the max over
                # e*x*amples" The x and u are included in the name because
                # otherwise its hard to remember which axis is which when
                # reading the monitor I use inner.outer
                # rather than outer_of_inner or
                # something like that because I want mean_x.* to appear next to
                # each other in the alphabetical list, as these are commonly
                # plotted together
                for key, val in [('max_x.max_u', v_max.max()),
                                 ('max_x.mean_u', v_max.mean()),
                                 ('max_x.min_u', v_max.min()),
                                 ('min_x.max_u', v_min.max()),
                                 ('min_x.mean_u', v_min.mean()),
                                 ('min_x.min_u', v_min.min()),
                                 ('range_x.max_u', v_range.max()),
                                 ('range_x.mean_u', v_range.mean()),
                                 ('range_x.min_u', v_range.min()),
                                 ('mean_x.max_u', v_mean.max()),
                                 ('mean_x.mean_u', v_mean.mean()),
                                 ('mean_x.min_u', v_mean.min())]:
                    rval[prefix+key] = val
       
       
        return rval    

class ConvSparseCoding(ConvElemwise):
    '''
        Parameters for the optimization/feedforward operation:
        lr      : learning rate
        n_steps : number of steps or uptades of the hidden code
        truncate: truncate the gradient after this number (default -1 which 
                  means do not truncate)
    '''
    
    def __init__(self, batch_size, input_channels=1, x_axes=['b', 'c', 0, 1], 
                 fprop_code=True, lr=.01, n_steps=10, lbda=0, top_most = False,
                  **kwargs):
        
        super(ConvSparseCoding, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.fprop_code = fprop_code
        self.n_steps = n_steps
        self.lr = lr
        self.input_channels = input_channels
        self.lbda = lbda
        self.top_most = top_most
    
    def initialize_x_space(self,rng):
        """
        This function initializes the coding space and dimmensions
        
        X is how I generally call the sparse code variables. 
        Thus, X_space has its dimmensions

        """
        dummy_batch_size = self.mlp.batch_size

        if dummy_batch_size is None:
            dummy_batch_size = self.batch_size
        dummy_detector =\
                sharedX(self.detector_space.get_origin_batch(dummy_batch_size))
        
        if self.pool_type is not None:
            assert self.pool_type in ['max', 'mean']
            if self.pool_type == 'max':
                dummy_p = max_pool(dummy_detector,
                                   self.pool_shape)
                ''',
                                   pool_stride=self.pool_stride,
                                   image_shape=self.detector_space.shape)
                '''
            elif self.pool_type == 'mean':
                dummy_p = mean_pool(dummy_detector,
                                    self.pool_shape)
                ''',
                                    pool_stride=self.pool_stride,
                                    image_shape=self.detector_shape.shape)
                '''
            dummy_p = dummy_p.eval()
            self.x_space = Conv2DSpace(shape=[dummy_p.shape[2],
                                              dummy_p.shape[3]],
                                            num_channels=self.output_channels,
                                            axes=('b', 'c', 0, 1))
        else:
            dummy_detector = dummy_detector.eval()
            self.x_space = Conv2DSpace(shape=[dummy_detector.shape[2],
                                            dummy_detector.shape[3]],
                                            num_channels=self.output_channels,
                                            axes=('b', 'c', 0, 1))
        
        X = rng.normal(0, .001, size=(dummy_batch_size,
                                     self.output_channels,
                                     self.detector_space.shape[0],
                                     self.detector_space.shape[1]))
        
        self.X = sharedX(X, self.layer_name+'_X')
        
        S = rng.normal(0, .001, size=(dummy_batch_size,
                                      self.input_channels,
                                      self.input_space.shape[0],
                                      self.input_space.shape[1]))
        
        self.S = sharedX(S, self.layer_name+'_S')
        
        # This is the statistic that comes from the layer above
        top_flow = rng.binomial(1, .1, size=(dummy_batch_size,
                                            self.output_channels,
                                            self.x_space.shape[0],
                                            self.x_space.shape[0]))

        self.top_flow = sharedX(top_flow, self.layer_name+'_top_flow')
                                      
        logger.info('Code space: {0}'.format(self.x_space.shape))

    @wraps(ConvElemwise.initialize_transformer)
    def initialize_transformer(self, rng):
        """
        This function initializes the transformer of the class. Re-running
        this function will reset the transformer.

        X is how I generally call the sparse code variables. 
        Thus, X_space has its dimmensions

        Parameters
        ----------
        rng : object
            random number generator object.
        """

        if self.irange is not None:
            assert self.sparse_init is None
            self.transformer = conv2d.make_random_conv2D(
                    irange=self.irange,
                    input_space=self.x_space,
                    output_space=self.input_space,
                    kernel_shape=self.kernel_shape,
                    subsample=self.kernel_stride,
                    border_mode=self.border_mode,
                    rng=rng)
        elif self.sparse_init is not None:
            self.transformer = conv2d.make_sparse_random_conv2D(
                    num_nonzero=self.sparse_init,
                    input_space=self.X_space,
                    output_space=self.detector_space,
                    kernel_shape=self.kernel_shape,
                    subsample=self.kernel_stride,
                    border_mode=self.border_mode,
                    rng=rng)
        
        self._reconstruction = theano.function([], self.transformer.lmul(self.X))

    
    @wraps(ConvElemwise.initialize_output_space)
    def initialize_output_space(self):
        
        if self.fprop_code is True:
            self.output_space = self.x_space
            '''
            if self.pool_shape is not None:
                self.output_space.shape = [self.output_space.shape[0] / self.pool_stride[0],
                                           self.output_space.shape[1] / self.pool_stride[1]]
            '''
        else:
            self.output_space = self.input_space

        logger.info('Output space: {0}'.format(self.output_space.shape))
    
    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        """ Note: this function will reset the parameters! """

        self.input_space = space

        if not isinstance(space, Conv2DSpace):
            raise BadInputSpaceError(self.__class__.__name__ +
                                     ".set_input_space "
                                     "expected a Conv2DSpace, got " +
                                     str(space) + " of type " +
                                     str(type(space)))

        rng = self.mlp.rng

        output_shape = [(self.input_space.shape[0] + self.kernel_shape[0])
                            / self.kernel_stride[0] - 1,
                            (self.input_space.shape[1] + self.kernel_shape[1])
                            / self.kernel_stride[1] - 1]

        self.detector_space = Conv2DSpace(shape=output_shape,
                                          num_channels=self.output_channels,
                                          axes=('b', 'c', 0, 1))

        self.initialize_x_space(rng)
        self.initialize_transformer(rng)

        W, = self.transformer.get_params()
        W.name = self.layer_name + '_W'

        if self.tied_b:
            self.b = sharedX(np.zeros((self.detector_space.num_channels)) +
                             self.init_bias)
        else:
            self.b = sharedX(self.detector_space.get_origin() + self.init_bias)

        self.b.name = self.layer_name + '_b'

        logger.info('Input shape: {0}'.format(self.input_space.shape))
        logger.info('Detector space: {0}'.format(self.detector_space.shape))

        self.initialize_output_space()

        cost = self.get_local_cost()
        self.opt = top.Optimizer(self.X, cost, method='rmsprop', 
                                 learning_rate=self.lr, momentum=.9)
        
    def get_reconstruction(self):
        return self._reconstruction()

    def get_local_cost(self):
        er = T.sqr(self.S - self.transformer.lmul(self.X)).sum()
        l1 = T.sqrt( T.sqr(self.X) + 1e-6).sum()
        top_down = self.get_top_down_flow()
        return er + .1 * l1 + top_down
    
    def update_top_state(self, state_above=None):
        if self.lbda is not 0:
            assert state_above is not None
            self.top_flow.set_value(state_above)
           
    def get_nonlin_output(self):
        rval = max_pool(self.X, self.pool_shape) 
        ''', 
        self.pool_stride, 
        [self.X.shape[2], self.X.shape[3]])
        '''
        #rval = T.switch(rval > 0., rval, 0.)
        #rval = T.maximum(rval, 0.)
        rval = self.nonlin.apply(rval)
        return rval


    def get_top_down_flow(self):
        if self.lbda == 0:
            rval = 0.
        elif self.top_flow == True:
            rval = (self.lbda * (self.top_flow - self.X)**2).sum()
        else:
            out = self.get_nonlin_output()
            rval = (self.lbda * (self.top_flow - out)**2).sum()

        return rval

    def _renormW(self):
        A = self.transformer.get_params()[0].get_value(borrow=True)
        Ashape = A.shape
        A = A.reshape((Ashape[0]*Ashape[1],Ashape[2]*Ashape[3]))
        A = np.dot(A.T, np.diag(1./np.sqrt(np.sum(A**2, axis=1)))).T
        A = A.reshape(Ashape)
        self.transformer.get_params()[0].set_value( A )
    
    def get_sparse_code(self, state_below):
        
        # Define code optimizer
                
        # Renorm W
        self._renormW()

        if hasattr(state_below, 'get_value'):
            #print '!!!! state_below does have get_value'
            assert state_below.get_value().shape == self.S.get_value().shape
            s_below = state_below.get_value(borrow=True)
            s_below = s_below[:,:self.input_channels,:,:]
            self.S.set_value(s_below)
            self.opt.run(self.n_steps)#, 
            #np.arange(self.batch_size))
        elif isinstance(state_below, np.ndarray):
            #print '!!! state_below is np.ndarray'
            s_below = state_below[:,:self.input_channels,:,:].astype('float32')
            self.S.set_value(s_below)
            self.opt.run(self.n_steps)#, 
            #np.arange(self.batch_size))
        #else:
        #    state_below = state_below[:,0,:,:].dimshuffle(0,'x',1,2)

        #self.state_below = state_below
        #self.local_reconstruction_error = \
        #        ((state_below - T.dot(self.Xout, self.W) - 0*self.b) ** 2).sum() + \
        #                   .1 * T.sqrt(self.Xout**2 + 1e-6).sum()
        
        return self.X

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        
        
        self.input_space.validate(state_below)
        rval = self.get_sparse_code(state_below)

        if self.fprop_code == True:
            '''
            rval = max_pool(rval, self.pool_shape, 
                            self.pool_stride, 
                            self.x_space.shape)
            rval = T.switch(rval > 0., rval, 0.)
            '''
            rval = self.get_nonlin_output()
        else:
            # Fprops the filtered input instead
            #rval = self.transformer.lmul(rval)
            rval = self.transformer.lmul(self.X)
        self.output_space.validate(rval)
        
        return rval

    #@wraps(Layer.get_params)
    #def get_params(self):
    #    return [self.transformer.get_params()[0]]
    
class CompositeSparseCoding(Linear):
    
    def __init__(self, batch_size, fprop_code=True, lr=.01, n_steps=10, lbda=0, top_most=False, 
            nonlinearity=RectifierConvNonlinearity(),*args, **kwargs):
        '''
        Compiled version: the sparse code is calulated using 'top' and is not just simbolic.
        Parameters for the optimization/feedforward operation:
        lr      : learning rate
        n_steps : number of steps or uptades of the hidden code
        truncate: truncate the gradient after this number (default -1 which means do not truncate)
        '''
        super(CompositeSparseCoding, self).__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.fprop_code = fprop_code
        self.n_steps = n_steps
        self.lr = lr
        self.lbda = lbda
        self.top_most = top_most
        self.nonlin = nonlinearity

    @wraps(Linear.set_input_space)
    def set_input_space(self, space):
        
        self.input_space = space
        assert isinstance(space, CompositeSpace)
        self.input_dim = []
        self.desired_space = []
        for sp in space.components:
            if isinstance(sp, VectorSpace):
                self.requires_reformat = False
                self.input_dim.append(sp.dim)
            else:
                self.requires_reformat = True
                self.input_dim.append(sp.get_total_dimension())
                self.desired_space.append( VectorSpace(self.input_dim[-1]) )

        if self.fprop_code==True:
            self.output_space = VectorSpace(self.dim)
        else:
            #self.output_space = VectorSpace(self.input_dim)
            # TODO: return composite space
            raise NotImplementedError

        rng = self.mlp.rng
        self.W = []
        self.S = []
        self.b = []
        self.transformer = []
        self._params = []
        
        X = .001 * rng.randn(self.batch_size, self.dim)
        self.X = sharedX(X, self.layer_name + '_X')
        
        for c in range(len(self.input_space.components)):
            W = rng.randn(self.input_dim[c], self.dim)
            self.W += [ sharedX(W.T, self.layer_name + '_W' + str(c)) ]
            self.transformer += [ MatrixMul(self.W[c]) ]
            self.W[-1], = self.transformer[-1].get_params()
            b = np.zeros((self.input_dim[c],))
            self.b += [ sharedX(b, self.layer_name + '_b' + str(c)) ] # We need both to pass input_dim valid
            S = rng.normal(0, .001, size=(self.batch_size, self.input_dim[c]))
            self.S += [ sharedX(S, self.layer_name + '_S' + str(c)) ]
            self._params += [self.W[-1], self.b[-1]]
            #self.state_below = T.zeros((self.batch_size, self.input_dim))
            
        cost = self.get_local_cost()
        self.opt = top.Optimizer(self.X, cost,  
                                 method='rmsprop', 
                                 learning_rate=self.lr, momentum=.9)
    
    def get_nonlin_ouput(self):
        return self.nonlin(self.X)
    
    def get_local_cost(self):
        er = 0.
        tflow = self.get_top_down_flow()
        flag = 0
        for s,w in zip(self.S, self.W):
            if flag==0:
                er += T.sqr(s - T.dot(self.X, w)).sum()
                flag = 1
        l1 = T.sqrt(T.sqr(self.X) + 1e-6).sum()
        return er + .2 * l1 + tflow
    
    def update_top_state(self, state_above=None):
        if self.lbda is not 0:
            assert state_above is not None
            self.top_flow.set_value(state_above)     

    def get_top_down_flow(self):
        if self.lbda == 0:
            rval = 0.
        elif self.top_flow == True:
            rval = (self.lbda * (self.top_flow - self.X)**2).sum()
        else:
            out = self.get_nonlin_output()
            rval = (self.lbda * (self.top_flow - out)**2).sum()

        return rval

    def _renormW(self):
        for w in self.W:
            A = w.get_value(borrow=True)
            A = np.dot(A.T, np.diag(1./np.sqrt(np.sum(A**2, axis=1)))).T
            w.set_value( A )
  
    def get_reconstruction(self):
        raise NotImplementedError
    
    def get_sparse_code(self, state_below):

        # Renorm W
        flag = False
        self._renormW()
        for sbelow, s in zip(state_below, self.S):
            if hasattr(sbelow, 'get_value'):
                #print '!!!! state_below does have get_value'
                s.set_value(sbelow.get_value(borrow=True))
                flag = True
            if isinstance(state_below, np.ndarray):
                s.set_value(sbelow.astype('float32'))
                flag = True
                #np.arange(self.batch_size))
        
        if flag is True:
            self.opt.run(self.n_steps) 

        return self.X

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        
        self._renormW()
        rval = self.get_sparse_code(state_below)
        if self.fprop_code == True:
            #rval = T.switch(rval > 0., rval, 0.)
            rval = self.nonlin.apply(rval)
        else:
            # Fprops the filtered input instead
            rval = T.dot(rval, self.W)
        
        return rval
    
    @wraps(Layer.get_params)
    def get_params(self):
        return self.W

    @functools.wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                      state=None, targets=None):

        rval = OrderedDict()       
       
        return rval  


class CompositeConvLin(ConvElemwise):
    '''
        Parameters for the optimization/feedforward operation:
        lr      : learning rate
        n_steps : number of steps or uptades of the hidden code
        truncate: truncate the gradient after this number (default -1 which 
                  means do not truncate)
    '''
    
    def __init__(self, batch_size, dim, input_channels=1, x_axes=['b', 'c', 0, 1], 
                 fprop_code=True, lr=.01, n_steps=10, lbda=0, top_most = False,
                  **kwargs):
        
        super(CompositeConvLin, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.fprop_code = fprop_code
        self.n_steps = n_steps
        self.lr = lr
        self.input_channels = input_channels
        self.lbda = lbda
        self.top_most = top_most
        self.dim = dim
    
    def initialize_x_space(self,rng):
        """
        This function initializes the coding space and dimmensions
        
        X is how I generally call the sparse code variables. 
        Thus, X_space has its dimmensions

        """
        dummy_batch_size = self.mlp.batch_size

        if dummy_batch_size is None:
            dummy_batch_size = self.batch_size
        dummy_detector =\
                sharedX(self.detector_space.get_origin_batch(dummy_batch_size))
        
        if self.pool_type is not None:
            assert self.pool_type in ['max', 'mean']
            if self.pool_type == 'max':
                dummy_p = max_pool(dummy_detector,
                                   self.pool_shape)
                ''',
                                   pool_stride=self.pool_stride,
                                   image_shape=self.detector_space.shape)
                '''
            elif self.pool_type == 'mean':
                dummy_p = mean_pool(dummy_detector,
                                    self.pool_shape)
                ''',
                                    pool_stride=self.pool_stride,
                                    image_shape=self.detector_shape.shape)
                '''
            dummy_p = dummy_p.eval()
            self.x_space = Conv2DSpace(shape=[dummy_p.shape[2],
                                              dummy_p.shape[3]],
                                            num_channels=self.output_channels,
                                            axes=('b', 'c', 0, 1))
        else:
            dummy_detector = dummy_detector.eval()
            self.x_space = Conv2DSpace(shape=[dummy_detector.shape[2],
                                            dummy_detector.shape[3]],
                                            num_channels=self.output_channels,
                                            axes=('b', 'c', 0, 1))
        
        X = rng.normal(0, .001, size=(dummy_batch_size,
                                     self.output_channels,
                                     self.detector_space.shape[0],
                                     self.detector_space.shape[1]))
        
        self.dim_x = self.output_channels * self.detector_space.shape[0] \
                      * self.detector_space.shape[1]

        self.X = sharedX(X, self.layer_name+'_X')
        
        S0 = rng.normal(0, .001, size=(dummy_batch_size,
                                      self.input_channels,
                                      self.input_space.components[0].shape[0],
                                      self.input_space.components[0].shape[1]))
        
        self.S0 = sharedX(S0, self.layer_name+'_S0')

        S1 = rng.normal(0, .001, 
                size=(dummy_batch_size, self.dim))

        self.S1 = sharedX(S1, self.layer_name+'_S1')
        
        # This is the statistic that comes from the layer above
        top_flow = rng.binomial(1, .1, size=(dummy_batch_size,
                                            self.output_channels,
                                            self.x_space.shape[0],
                                            self.x_space.shape[0]))

        self.top_flow = sharedX(top_flow, self.layer_name+'_top_flow')
                                      
        logger.info('Code space: {0}'.format(self.x_space.shape))

    @wraps(ConvElemwise.initialize_transformer)
    def initialize_transformer(self, rng):
        """
        This function initializes the transformer of the class. Re-running
        this function will reset the transformer.

        X is how I generally call the sparse code variables. 
        Thus, X_space has its dimmensions

        Parameters
        ----------
        rng : object
            random number generator object.
        """

        if self.irange is not None:
            assert self.sparse_init is None
            self.transformer = conv2d.make_random_conv2D(
                    irange=self.irange,
                    input_space=self.x_space,
                    output_space=self.input_space.components[0],
                    kernel_shape=self.kernel_shape,
                    subsample=self.kernel_stride,
                    border_mode=self.border_mode,
                    rng=rng)
        elif self.sparse_init is not None:
            self.transformer = conv2d.make_sparse_random_conv2D(
                    num_nonzero=self.sparse_init,
                    input_space=self.X_space,
                    output_space=self.detector_space,
                    kernel_shape=self.kernel_shape,
                    subsample=self.kernel_stride,
                    border_mode=self.border_mode,
                    rng=rng)

    
    @wraps(ConvElemwise.initialize_output_space)
    def initialize_output_space(self):
        
        if self.fprop_code is True:
            self.output_space = self.x_space
            '''
            if self.pool_shape is not None:
                self.output_space.shape = [self.output_space.shape[0] / self.pool_stride[0],
                                           self.output_space.shape[1] / self.pool_stride[1]]
            '''
        else:
            #self.output_space = self.input_space
            raise NotImplementedError

        logger.info('Output space: {0}'.format(self.output_space.shape))
    
    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        """ Note: this function will reset the parameters! """

        self.input_space = space

        if not isinstance(space, CompositeSpace):
            raise BadInputSpaceError(self.__class__.__name__ +
                                     ".set_input_space "
                                     "expected a CompositeSpace, got " +
                                     str(space) + " of type " +
                                     str(type(space)))

        rng = self.mlp.rng

        output_shape = [(self.input_space.components[0].shape[0] + self.kernel_shape[0])
                            / self.kernel_stride[0] - 1,
                            (self.input_space.components[0].shape[1] + self.kernel_shape[1])
                            / self.kernel_stride[1] - 1]

        self.detector_space = Conv2DSpace(shape=output_shape,
                                          num_channels=self.output_channels,
                                          axes=('b', 'c', 0, 1))

        self.initialize_x_space(rng)
        self.initialize_transformer(rng)

        W, = self.transformer.get_params()
        W.name = self.layer_name + '_W'

        W1 = rng.normal(0, .001, size=(self.dim_x, self.dim))
        self.W1 = sharedX(W1, name=self.layer_name + '_W1')

        if self.tied_b:
            self.b = sharedX(np.zeros((self.detector_space.num_channels)) +
                             self.init_bias)
        else:
            self.b = sharedX(self.detector_space.get_origin() + self.init_bias)

        self.b.name = self.layer_name + '_b'

        logger.info('Input 0 shape: {0}'.format(self.input_space.components[0].shape))
        logger.info('Input 1 shape: {0}'.format(self.input_space.components[1].shape))
        logger.info('Detector space: {0}'.format(self.detector_space.shape))

        self.initialize_output_space()

        cost = self.get_local_cost()
        self.opt = top.Optimizer(self.X, cost, method='rmsprop', 
                                 learning_rate=self.lr, momentum=.9)
        
    def get_reconstruction(self):
        raise NotImplementedError
    
    def get_local_cost(self):
        er = T.sqr(self.S0 - self.transformer.lmul(self.X)).sum()
        flatX = self.X.reshape((self.mlp.batch_size, self.dim_x))
        er1 = T.sqr(self.S1 - T.dot(flatX, self.W1)).sum()
        l1 = T.sqrt( T.sqr(self.X) + 1e-6).sum()
        top_down = self.get_top_down_flow()
        return er + er1 + .1 * l1 + top_down
    
    def update_top_state(self, state_above=None):
        if self.lbda is not 0:
            assert state_above is not None
            self.top_flow.set_value(state_above)
           
    def get_nonlin_output(self):
        rval = max_pool(self.X, self.pool_shape) 
        ''', 
        self.pool_stride, 
        [self.X.shape[2], self.X.shape[3]])
        '''
        #rval = T.switch(rval > 0., rval, 0.)
        #rval = T.maximum(rval, 0.)
        rval = self.nonlin.apply(rval)
        return rval


    def get_top_down_flow(self):
        if self.lbda == 0:
            rval = 0.
        elif self.top_flow == True:
            rval = (self.lbda * (self.top_flow - self.X)**2).sum()
        else:
            out = self.get_nonlin_output()
            rval = (self.lbda * (self.top_flow - out)**2).sum()

        return rval
    
    def get_params(self):
        params = []
        params += self.transformer.get_params()
        params += [self.W1]
        return params

    def _renormW(self):
        A = self.transformer.get_params()[0].get_value(borrow=True)
        Ashape = A.shape
        A = A.reshape((Ashape[0]*Ashape[1],Ashape[2]*Ashape[3]))
        A = np.dot(A.T, np.diag(1./np.sqrt(np.sum(A**2, axis=1)))).T
        A = A.reshape(Ashape)
        self.transformer.get_params()[0].set_value( A )
    
    def get_sparse_code(self, state_below):
        
        # Define code optimizer
                
        # Renorm W
        self._renormW()

        if isinstance(state_below, tuple):
            #print state_below[0].dtype
            if hasattr(state_below[0], 'get_value'):
                #print '!!!! state_below does have get_value'
                assert state_below[0].get_value().shape == self.S0.get_value().shape
                s_below0 = state_below[0].get_value(borrow=True)
                #s_below0 = s_below[:,:self.input_channels,:,:]
                self.S0.set_value(s_below0)
                s_below1 = state_below[1].get_value(borrow=True).reshape((self.mlp.batch_size,self.dim))
                self.S1.set_value(s_below1)
                self.opt.run(self.n_steps)#, 
                #np.arange(self.batch_size))
            elif isinstance(state_below[0], np.ndarray):
                #print '!!! state_below is np.ndarray'
                #s_below = state_below[:,:self.input_channels,:,:].astype('float32')
                self.S0.set_value(state_below[0])
                self.S1.set_value(state_below[1].reshape((self.mlp.batch_size,self.dim)))
                self.opt.run(self.n_steps)#,         
        return self.X

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        
        
        self.input_space.validate(state_below)
        rval = self.get_sparse_code(state_below)

        if self.fprop_code == True:
            '''
            rval = max_pool(rval, self.pool_shape, 
                            self.pool_stride, 
                            self.x_space.shape)
            rval = T.switch(rval > 0., rval, 0.)
            '''
            rval = self.get_nonlin_output()
        else:
            # Fprops the filtered input instead
            #rval = self.transformer.lmul(rval)
            raise NotImplementedError

        self.output_space.validate(rval)
        
        return rval

    #@wraps(Layer.get_params)
    #def get_params(self):
    #    return [self.transformer.get_params()[0]]


class DPCN(MLP):
    '''
    A Deep Predictive Coding Network

    Since pylearn2 MLPs can be considered a single layer, 
    we do so for DPCNs.
    DPCN has a characteristic top-down flow despite of the regular
    botton-up that is common to all Neural Networks.
    Here, we make the top-down flow to be help by the DPCN class
    instead of handling it to the trainer algorithm. 
    
    Parameters
    ----------
    layers: list
        A list of layer objects. Valid layers are those defined above like
        "ConvSparseCoding" and "CompositeConvLin". I still have TODO something
        for SparseCodingLayer.
        batch_size : int, optional
    layer_name : name of the MLP layer. Should be None if the MLP is
        not part of another MLP.
    time_range : int
        When pylearn2 starts supporting sequential datasets, this should be changed.
        Right now, the time range is encoded as part of the channels dimension.
        We use this extra time_range variable to extract the right temporal structure
        of the dataset. 
        TODO: read time_range from the dataset object
    '''

    def __init__(self, layers, time_range, batch_size=None, input_space=None,
                 input_source='features', nvis=None, seed=None,
                 layer_name=None, monitor_targets=True, **kwargs):
        
        super(DPCN, self).__init__(layers=layers, batch_size=batch_size, 
                input_space=input_space, input_source=input_source, 
                nvis=nvis, seed=seed, layer_name=layer_name)

        self.time_range = time_range
        
    def _update_layer_input_spaces(self):
        """
        Tells each layer what its input space should be.
        Notes
        -----
        This usually resets the layer's parameters!
        """
        layers = self.layers
        first_input_space = copy(self.get_input_space())
        first_input_space.num_channels /= self.time_range 

        try:
            layers[0].set_input_space( first_input_space )
        except BadInputSpaceError, e:
            raise TypeError("Layer 0 (" + str(layers[0]) + " of type " +
                            str(type(layers[0])) +
                            ") does not support the MLP's "
                            + "specified input space (" +
                            str(self.get_input_space()) +
                            " of type " + str(type(self.get_input_space())) +
                            "). Original exception: " + str(e))
        for i in xrange(1, len(layers)):
            layers[i].set_input_space(layers[i-1].get_output_space()) 

        self.compiled_fprop = []
        for L in layers:
            X = L.get_input_space().make_theano_batch()
            if isinstance(X, tuple):
                self.compiled_fprop += [ theano.function(X, L.fprop(X), on_unused_input='ignore') ]
            else:
                self.compiled_fprop += [ theano.function([X], L.fprop(X), on_unused_input='ignore') ]

    @wraps(Layer.get_input_space)
    def get_input_space(self):

        return self.input_space
    
    @wraps(Layer.fprop)
    def fprop(self, state_below, return_all=False):

        if not hasattr(self, "input_space"):
            raise AttributeError("Input space has not been provided.")

        
        for t in range(self.time_range):
            self.update_hidden_codes(state_below, t)
            
        rval = []
        if return_all:
            for layer in self.layers:
                rval += layer.get_nonlin_output()
        else:
            rval = self.layers[-1].get_nonlin_output()

        return rval

    def update_hidden_codes(self, batch, t):
        # Run top optmizer of all the generative models.
        flag = 1
        t1 = t * self.num_channels
        t2 = (t+1) * self.num_channels
        if isinstance(batch, tuple):
            nbatch = []
            for c in range( len(batch)-1 ):
                nbatch += [batch[c][:,t1:t2,:,:]]
            batch = tuple(nbatch)
            del nbatch
        else:
            batch = batch[:,t1:t2,:,:]

        for l in range(len(self.layers)):
            L = self.model.layers[l]
            if isinstance(L, ConvSparseCoding) or \
                    isinstance(L, SparseCodingLayer) or \
                    isinstance(L, CompositeSparseCoding) or \
                    isinstance(L, CompositeConvLin):
                # Update upper state
                if L.top_most is False:
                    try:
                        L_above = self.model.layers[l+1]
                        L.update_top_state(L_above.get_reconstruction())
                    except:
                        L.lbda = 0
                else:
                    L_above = L.X.get_value()
                    L.update_top_state(L_above)
                # Update layer state
                if flag:
                    X = batch; flag = 0
                else:
                    X = Y
                sc = L.get_sparse_code(X)
                if isinstance(X,tuple):
                    Y  = self.compiled_fprop[l](*X)
                else:
                    Y = self.compiled_fprop[l](X)

