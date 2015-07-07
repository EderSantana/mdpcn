__authors__ = "Eder Santana"
__copyright__ = "Copyright 2014-2015, University of Florida"
__credits__ = "Eder Santana"
__license__ = "3-clause BSD"
__maintainer__ = "Eder Santana"

import logging
logger = logging.getLogger(__name__)

from pylearn2.space import VectorSpace, Conv2DSpace
from pylearn2.models.mlp import Layer, Linear, ConvElemwise
from pylearn2.utils import sharedX
from pylearn2.utils import wraps
from pylearn2.space import Space
from pylearn2.utils import wraps
from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.linear import conv2d
from theano.tensor.signal.downsample import max_pool_2d as max_pool

import functools
import theano
from theano import tensor as T
from theano.printing import Print
from theano.compat.python2x import OrderedDict
from theano.gof.op import get_debug_values
import top
import numpy as np

class SparseCodingLayer(Linear):
    
    def __init__(self, batch_size, fprop_code=True, lr=.01, n_steps=10, truncate=-1, *args, **kwargs):
        '''
        Parameters for the optimization/feedforward operation:
        lr      : learning rate
        n_steps : number of steps or uptades of the hidden code
        truncate: truncate the gradient after this number (default -1 which means do not truncate)
        '''
        super(SparseCodingLayer, self).__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.fprop_code = fprop_code
        self.n_steps = n_steps
        self.truncate = truncate
        self.lr = lr
        self._scan_updates = OrderedDict()

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
        self._params = [self.W, self.b, self.X]
        self.state_below = T.zeros((self.batch_size, self.input_dim))

    def _renormW(self):
        A = self.W.get_value(borrow=True)
        A = np.dot(A.T, np.diag(1./np.sqrt(np.sum(A**2, axis=1)))).T
        self.W.set_value( A )
    
    def get_local_cost(self,state_below):
        er = T.sqr(state_below - T.dot(self.X, self.W)).sum()
        l1 = T.sqrt(T.sqr(self.X) + 1e-6).sum()
        return er + .1 * l1
        
    def get_sparse_code(self, state_below):

        def _optimization_step(Xt, accum, vt, S):
                
            '''
            Note that this is the RMSprop update. 
            Thus, we running gradient updates inside scan (the dream)
            
            TODO: put this a better place.
            I tried to make if a method of self, but I'm not sure how to tell 
            theano.scan that the first argument of the function is a non_sequence
            '''
            
            rho = .9
            momentum = .9
            lr = self.lr
            Y = T.dot(Xt, self.W) #+ self.b
            err = (S - Y) ** 2
            l1 = T.sqrt(Xt**2 + 1e-6)
            cost = err.sum() + .1 * l1.sum()
            gX = T.grad(cost, Xt)
            new_accum = rho * accum + (1-rho) * gX**2
            v = momentum * vt  - lr * gX / T.sqrt(new_accum + 1e-8)
            X = Xt + momentum * v - lr * gX / T.sqrt(new_accum + 1e-8)
            return [X, new_accum, v]

        # Renorm W
        self._renormW()
        
        rng = self.mlp.rng
        #X = rng.randn(self.batch_size, self.dim)
        #self.X = sharedX(X, 'SparseCodingLinear_X')
        '''
        accum = T.zeros_like(self.X)
        vt = T.zeros_like(self.X)
        [Xfinal,_,_], updates = theano.scan(fn=_optimization_step,
                     outputs_info=[self.X, accum, vt], 
                     non_sequences=[state_below], 
                     n_steps=self.n_steps, truncate_gradient=self.truncate)
        
        self._scan_updates.update(updates)

        self.Xout = Xfinal[-1]
        #self.Xout = (2*T.ge(self.Xout, 0.)-1) * T.maximum(abs(self.Xout) - .01, 0.)
        self.state_below = state_below
        #self.local_reconstruction_error = \
        #        ((state_below - T.dot(self.Xout, self.W) - 0*self.b) ** 2).sum() + \
        #                   .1 * T.sqrt(self.Xout**2 + 1e-6).sum()
        '''
        return self.X #out
    
    @wraps(Layer._modify_updates)
    def _modify_updates(self, updates):
        updates.update(self._scan_updates)

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        
        if self.fprop_code == True:
            rval = self.get_sparse_code(state_below)
            rval = T.switch(rval > 0., rval, 0.)
        else:
            # Fprops the filtered input instead
            rval = T.dot(self.get_sparse_code(state_below), self.W)
        
        return rval
        
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
    
    def __init__(self, batch_size, x_axes=['b', 'c', 0, 1], 
                 fprop_code=True, lr=.01, n_steps=10, 
                 truncate=-1, *args, **kwargs):
        
        super(ConvSparseCoding, self).__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.fprop_code = fprop_code
        self.n_steps = n_steps
        self.truncate = truncate
        self.lr = lr
        self._scan_updates = OrderedDict()
    
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
                '''
                                   pool_stride=self.pool_stride,
                                   image_shape=self.detector_space)
                '''
            elif self.pool_type == 'mean':
                dummy_p = mean_pool(dummy_detector,
                                    self.pool_shape)
                '''
                                    pool_stride=self.pool_stride,
                                    image_shape=self.detector_shape)
                '''
            dummy_p = dummy_p.eval()
            self.x_space = Conv2DSpace(shape=[dummy_p.shape[2],
                                              dummy_p.shape[3]],
                                            num_channels=
                                                self.output_channels,
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

            
    def get_local_cost(self, state_below):
        er = T.sqr(state_below - self.transformer.lmul(self.X)).sum()
        l1 = T.sqrt( T.sqr(self.X) + 1e-6).sum()
        return er + .1 * l1   


    @wraps(ConvElemwise.initialize_output_space)
    def initialize_output_space(self):
        
        if self.fprop_code is True:
            self.output_space = self.x_space
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


    def _renormW(self):
        A = self.transformer.get_params()[0].get_value(borrow=True)
        Ashape = A.shape
        A = A.reshape((Ashape[0]*Ashape[1],Ashape[2]*Ashape[3]))
        A = np.dot(A.T, np.diag(1./np.sqrt(np.sum(A**2, axis=1)))).T
        A = A.reshape(Ashape)
        self.transformer.get_params()[0].set_value( A )
    
    def get_sparse_code(self, state_below):

        def _optimization_step(Xt, accum, vt, S):
                
            '''
            Note that this is the RMSprop update. 
            Thus, we running gradient updates inside scan (the dream)
            
            TODO: put this a better place.
            I tried to make if a method of self, but I'm not sure how to tell 
            theano.scan that the first argument of the function is a non_sequence
            '''
            
            rho = .9
            momentum = .9
            lr = self.lr
            Y = self.transformer.lmul(Xt) #T.dot(Xt, self.W) #+ self.b
            err = (S - Y) ** 2
            l1 = T.sqrt(Xt**2 + 1e-6)
            cost = err.sum() + .1 * l1.sum()
            #cost = self.get_local_cost(S)
            gX = T.grad(cost, Xt)
            new_accum = rho * accum + (1-rho) * gX**2
            v = momentum * vt  - lr * gX / T.sqrt(new_accum + 1e-8)
            X = Xt + momentum * v - lr * gX / T.sqrt(new_accum + 1e-8)
            return [X, new_accum, v]

        # Renorm W
        self._renormW()
        
        rng = self.mlp.rng
        #X = rng.randn(self.batch_size, self.dim)
        #self.X = sharedX(X, 'SparseCodingLinear_X')
        accum = T.zeros_like(self.X)
        vt = T.zeros_like(self.X)
        [Xfinal,_,_], updates = theano.scan(fn=_optimization_step,
                     outputs_info=[self.X, accum, vt], 
                     non_sequences=[state_below], 
                     n_steps=self.n_steps, truncate_gradient=self.truncate)
            
        self._scan_updates.update(updates)
        
        self.Xout = Xfinal[-1]
        #self.Xout = (2*T.ge(self.Xout, 0.)-1) * T.maximum(abs(self.Xout) - .01, 0.)
        self.state_below = state_below
        #self.local_reconstruction_error = \
        #        ((state_below - T.dot(self.Xout, self.W) - 0*self.b) ** 2).sum() + \
        #                   .1 * T.sqrt(self.Xout**2 + 1e-6).sum()
        
        return self.Xout
    
    @wraps(Layer._modify_updates)
    def _modify_updates(self, updates):
        updates.update(self._scan_updates)

    def get_nonlin_output(self, state_below):
        rval = max_pool(self.X, self.pool_shape)
        rval = self.nonlin.apply(rval)
        return rval


    @wraps(Layer.fprop)
    def fprop(self, state_below):

        self.input_space.validate(state_below)
        rval = self.get_sparse_code(state_below)

        if self.fprop_code == True:
            #rval = T.switch(rval > 0., rval, 0.)
            rval = self.get_nonlin_output(state_below)
        else:
            # Fprops the filtered input instead
            rval = self.transformer.lmul(rval)

        self.output_space.validate(rval)
        
        return rval
       
