from theano import tensor as T
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError
from cdpcn import SparseCodingLayer, ConvSparseCoding, CompositeSparseCoding, CompositeConvLin, DPCN
from theano.tensor.signal.downsample import max_pool_2d
import dpcn

class SparseReconstructionError(MeanSquaredReconstructionError):
    '''
    Similar to MeanSquaredReconstructionError with extra l1 penalty.
    This is intended to be used with conjunction to LinearSparseCoder models. There,
    input reconstruction is part of the fprop method. Thus, we don't have to use a 
    specific Autoencoder Model here. On the other hand, we have to overwrite the 
    class 'expr' method.
    '''

    def expr(self, model, data, *args, **kwargs):
        '''
        .. todo::

        WRITEME
        '''
        self.get_data_specs(model)[0].validate(data)
        #X = data
        #print data
        '''
        TODO: I have intentions of making this cost function looks through 
        all the local reconstruction at every layer.
        '''
        rval = 0.
        flag = 1
        for L in model.layers: 
            if isinstance(L, DPCN):
                for M in L.layers:
                    if isinstance(M, SparseCodingLayer) or \
                            isinstance(M, ConvSparseCoding) or \
                            isinstance(M, CompositeSparseCoding) or \
                            isinstance(M, CompositeConvLin):
                        if flag:
                            X = data
                            flag = 0
                        else:
                            X = M.get_nonlin_output()
                            sc = M.get_sparse_code(X)
                    rval += M.get_local_cost()

            if isinstance(L, SparseCodingLayer) or \
                    isinstance(L, ConvSparseCoding) or \
                    isinstance(L, CompositeSparseCoding) or \
                    isinstance(L, CompositeConvLin):
                if flag:
                    X = data
                    flag = 0
                else:
                    X = L.get_nonlin_output()
                sc = L.get_sparse_code(X)
                rval += L.get_local_cost()

        return rval

class ConvSparseReconstructionError(MeanSquaredReconstructionError):
    '''
    Similar to MeanSquaredReconstructionError with extra l1 penalty.
    This is intended to be used with conjunction to LinearSparseCoder models. There,
    input reconstruction is part of the fprop method. Thus, we don't have to use a 
    specific Autoencoder Model here. On the other hand, we have to overwrite the 
    class 'expr' method.
    '''

    def expr(self, model, data, *args, **kwargs):
        '''
        .. todo::

        WRITEME
        '''
        self.get_data_specs(model)[0].validate(data)
        #X = data
        '''
        TODO: I have intentions of making this cost function looks through all the local 
        reconstruction at every layer.
        '''

        rval = 0.
        flag = 1
        for L in model.layers: 
            if isinstance(L, ConvSparseCoding): 
                #L = model.layers[-1]
                #print L
                #X = data
                #X = L.state_below #Yay!, I may delete state_below from model and save memory
                # Actually I think I'm keeping this... Think training several layers at once.
                if flag:
                    X = data
                    flag = 0
                else:
                    X = L.get_nonlin_output() #max_pool_2d(sc, [2,2])
                sc = L.get_sparse_code(X)
                #Y  = L.fprop(X)
                #l1 = T.sqrt(sc**2 + 1e-6).sum()
                #reconstruction = L.transformer.lmul(sc) #+ L.b
                #rval += T.sqr(X - reconstruction).sum() + 0.1 * l1
                rval += L.get_local_cost()
        return rval

class DPCNError(MeanSquaredReconstructionError):
    '''
    Similar to MeanSquaredReconstructionError with extra l1 penalty.
    This is intended to be used with conjunction to LinearSparseCoder models. There,
    input reconstruction is part of the fprop method. Thus, we don't have to use a 
    specific Autoencoder Model here. On the other hand, we have to overwrite the 
    class 'expr' method.
    '''

    def expr(self, model, data, *args, **kwargs):
        '''
        .. todo::

        WRITEME
        '''
        self.get_data_specs(model)[0].validate(data)
        #X = data
        '''
        TODO: I have intentions of making this cost function looks through all the local 
        reconstruction at every layer.
        '''

        rval = 0.
        flag = 1
        for L in model.layers: 
            if isinstance(L, dpcn.ConvSparseCoding) or isinstance(L, dpcn.SparseCodingLayer): 
                #L = model.layers[-1]
                #print L
                #X = data
                #X = L.state_below #Yay!, I may delete state_below from model and save memory
                # Actually I think I'm keeping this... Think training several layers at once.
                if flag:
                    X = data
                    flag = 0
                else:
                    X = L.get_nonlin_output() #max_pool_2d(sc, [2,2])
                sc = L.get_sparse_code(X)
                #Y  = L.fprop(X)
                #l1 = T.sqrt(sc**2 + 1e-6).sum()
                #reconstruction = L.transformer.lmul(sc) #+ L.b
                #rval += T.sqr(X - reconstruction).sum() + 0.1 * l1
                rval += L.get_local_cost(X)
        return rval

