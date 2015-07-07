from cdpcn import SparseCodingLayer, ConvSparseCoding, CompositeConvLin, CompositeSparseCoding
from theano import tensor as T
import theano
import numpy as np
from scipy.misc import imresize

cdpcn_classes = [SparseCodingLayer, ConvSparseCoding,
                  CompositeConvLin, CompositeSparseCoding]

class TRANSFORMER():
    '''
    This class get a DPCN model.
    TRANSFORMER has method for calculating the dynamic codes of a DPCN compatible dataset.
    Also, it calculates reconstructed images from noisy observations.
    The dataset should be organized as [batches x time x row x col]
    So far, only gray scale images are investigated.
    '''
    def __init__(self, model, time_range):
        self.model = model
        self.compiled_fprop = []
        for L in self.model.layers:
            X = L.get_input_space().make_theano_batch()
            if isinstance(X, tuple):
                self.compiled_fprop.append( theano.function(X, L.fprop(X), on_unused_input='ignore') )
            else:
                self.compiled_fprop.append( theano.function([X], L.fprop(X), on_unused_input='ignore') )
        
    def transform(self, dataset):
        
        if not isinstance(dataset, tuple):
            dataset = (dataset)
        batch_size = self.model.batch_size
        data_size = dataset[0].shape[0]
        time_range = dataset[0].shape[1]
        rval = []
        for L in self.model.layers:
            if isinstance(L, ConvSparseCoding) or \
                    isinstance(L, SparseCodingLayer) or \
                    isinstance(L, CompositeSparseCoding) or \
                    isinstance(L, CompositeConvLin):
                a,b,c,d = L.X.get_value().shape
                rval += [np.zeros((data_size, time_range, b, c, d))]

        iternum = 0
        for i in range(data_size / batch_size):
            data_batch = []        
            start = i*batch_size
            end   = (i+1)*batch_size  
            # Iterate through time
            #print iternum
            iternum += 1
            print iternum
            for t in range(time_range):
                #for d in dataset:
                #    data_batch +=  [ d[start:end,:,:,:] ]
                data_batch = [ dataset[0][start:end], dataset[1][start:end] ]
                self.update_hidden_codes(tuple(data_batch), t)
                cc = 0
                for L in self.model.layers:
                    if isinstance(L, ConvSparseCoding) or \
                       isinstance(L, SparseCodingLayer) or \
                       isinstance(L, CompositeSparseCoding) or \
                       isinstance(L, CompositeConvLin):
                        rval[cc][start:end,t,:,:,:] = L.X.get_value()
                        cc += 1
            
        return rval

    def transform2(self, dataset):
        
        if not isinstance(dataset, tuple):
            dataset = (dataset)
        batch_size = self.model.batch_size
        data_size = dataset[0].shape[0]
        time_range = dataset[0].shape[1]
        rval = []
        for L in self.model.layers:
            if isinstance(L, ConvSparseCoding) or \
                    isinstance(L, SparseCodingLayer) or \
                    isinstance(L, CompositeSparseCoding) or \
                    isinstance(L, CompositeConvLin):
                a,b,c,d = L.X.get_value().shape
                rval += [np.zeros((data_size, time_range, b, c, d))]

        iternum = 0
        for i in range(data_size / batch_size):
            data_batch = []        
            start = i*batch_size
            end   = (i+1)*batch_size  
            # Iterate through time
            #print iternum
            iternum += 1
            print iternum
            for t in range(time_range):
                #for d in dataset:
                #    data_batch +=  [ d[start:end,:,:,:] ]
                data_batch = [ dataset[0][start:end], dataset[1][start:end] ]
                self.update_hidden_codes2(tuple(data_batch), t)
                cc = 0
                for L in self.model.layers:
                    if isinstance(L, ConvSparseCoding) or \
                       isinstance(L, SparseCodingLayer) or \
                       isinstance(L, CompositeSparseCoding) or \
                       isinstance(L, CompositeConvLin):
                        rval[cc][start:end,t,:,:,:] = L.X.get_value()
                        cc += 1
            
        return rval
    
    def update_hidden_codes2(self, batch, t):
        # Run top optmizer of all the generative models.
        flag = 1

        nbatch = []
        for c in range( len(batch) ):
            nbatch += [ batch[c][:,t:t+1,:,:] ]
        batch = tuple(nbatch)
        del nbatch

        for l in range(len(self.model.layers)):
            L = self.model.layers[l]
            if isinstance(L, ConvSparseCoding) or \
                    isinstance(L, SparseCodingLayer) or \
                    isinstance(L, CompositeSparseCoding) or \
                    isinstance(L, CompositeConvLin):
                # Update upper state
                if L.top_most is False:
                    try:
                        L_above = self.model.layers[l+1]
                        L.update_top_state(L_above.X.get_value(borrow=True))
                    except:
                        L.lbda = 0
                else:
                    L_above = L.X
                    L.update_top_state(L_above.get_value(borrow=True))
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

    def update_hidden_codes(self, batch, t):
        # Run top optmizer of all the generative models.
        flag = 1
        
        nbatch = []
        for c in range( min(2, len(batch)) ):
            nbatch += [ batch[c][:,t:t+1,:,:] ]
        batch = tuple(nbatch)
        del nbatch

        for l in range(len(self.model.layers)):
            L = self.model.layers[l]
            if isinstance(L, ConvSparseCoding) or \
                    isinstance(L, SparseCodingLayer) or \
                    isinstance(L, CompositeSparseCoding) or \
                    isinstance(L, CompositeConvLin):
                # Update upper state
                if L.top_most is False:
                    try:
                        L_above = self.model.layers[l+1]
                        L.update_top_state(L_above.get_reconstruction() * (t!=0))
                    except:
                        L.lbda = 0
                else:
                    L_above = L.X
                    L.update_top_state(L_above.get_value(borrow=True) * (t!=0))
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

    def reconstruct_from_L(self, inp, L):
        '''
        Reconstruct the input from the code in rval. 
        rval can be calculated with the transform method of this class.
        L is the layer index that will govern the reconstruction
        '''
        rval = np.zeros_like(inp)
        compiled_rec = []
        img_size = []
        for i in range(L):
            #X = self.model.layers[0].get_input_space().make_theano_batch()
            if isinstance(L, ConvSparseCoding):
                X = T.tensor4()
                rec = L.transformer.lmul(X)
                compiled_rec += [ theano.function([X], rec) ]
            
            if isinstance(L, ConvSparseCoding):
                img_size += [ L.x_space.shape ]
        
        for i in range(L-1,-1,-1):
            for j in range(data_size / batch_size):
                start = j*batch_size
                end = (j+1)*batch_size
                batch = inp[start:end]
                Y = compiled_rec[i](batch)
                if i is not 0:
                    Y = Y.reshape((a,b,c,d))
                rval[start:end] = Y
        return rval
            
             
            



            

        
