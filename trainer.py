import top
from cdpcn import ConvSparseCoding, SparseCodingLayer, CompositeSparseCoding, CompositeConvLin
import cdpcn
import theano

class Trainer():
    
    def __init__(self, cost, model, dataset, monitoring_dataset, lr, save_path, epochs):
        self.model = model
        self.dataset = dataset
        self.monitoring_dataset = monitoring_dataset
        self.lr = lr
        self.input = model.get_input_space().make_theano_batch()
        cost=cost.expr(model, self.input)
        params = self.model.get_params()
        self.opt = top.Optimizer(params[0], cost, method='sgd', learning_rate=self.lr)
        
        self.compiled_fprop = []
        for L in self.model.layers:
            X = L.get_input_space().make_theano_batch()
            if isinstance(X, tuple):
                self.compiled_fprop.append( theano.function(X, L.fprop(X), on_unused_input='ignore') )
            else:
                self.compiled_fprop.append( theano.function([X], L.fprop(X), on_unused_input='ignore') )
        
        self.save_path = save_path
        self.epochs = epochs
        self.valid_best = 10e20
        self.test_best = 0

    def main_loop(self):
        
        for e in xrange(self.epochs):
            print "\n============ \nEpoch: %d" % e
            self.train_epoch()
            if 'valid' in self.monitoring_dataset.keys():
                self.validate_and_save()
            self.print_monitor()
            #do any update
    
    def train_epoch(self):
        data_size = self.dataset.get_data()[0].shape[0]
        # Get the full training batch
        iternum = 0
        for data_batch in self.dataset.iterator('shuffled_sequential', self.model.batch_size):
            if not isinstance(data_batch, tuple):
                data_batch = self.dataset.view_converter.design_mat_to_topo_view(data_batch)
            # Iterate through time
            #print iternum
            iternum += 1
            #print iternum
            self._train_err = 0.
            for t in range(data_batch.shape[1]):
                self.update_hidden_codes(data_batch[:,t:t+1,:,:])
                _, err = self.opt.run(1)
                self._train_err += err
            #print self._train_err
    
    def validate_and_save(self):
        data_size = self.monitoring_dataset['valid'].X.shape[0]
        iternum = 0
        for data_batch in self.monitoring_dataset['valid'].iterator('sequential', self.model.batch_size):
            data_batch = self.monitoring_dataset['valid'].view_converter.design_mat_to_topo_view(data_batch)
            self._valid_err = 0.
            for t in range(data_batch.shape[1]):
                self.update_hidden_codes(data_batch[:,t:t+1,:,:])
                err = self.opt.g()
                self._valid_err += err
        
        if 'test' in self.monitoring_dataset.keys():
            data_size = self.monitoring_dataset['test'].X.shape[0]
            for data_batch in self.monitoring_dataset['test'].iterator('sequential', self.model.batch_size):
                data_batch = self.monitoring_dataset['test'].view_converter.design_mat_to_topo_view(data_batch)
                self._test_err = 0.
                for t in range(data_batch.shape[1]):
                    self.update_hidden_codes(data_batch[:,t:t+1,:,:])
                    err = self.opt.g()
                    self._test_err += err
                
        if self.valid_best < self._valid_err:
            self.valid_best = self._valid_err
            self.test_best  = self._test_err
            print "Saving best results..."
            cPickle.dump(self.model, file(self.save_path, 'w'), -1)        
    
    def print_monitor(self):
        print "Training obj: %f" % self._train_err
        try:
            print 'Valid obj: %f' % self._valid_err
        except:
            pass
        try:
            print 'Test obj: %f' % self._test_err
        except:
            pass
        
    
    def update_hidden_codes(self, batch):
        # Run top optmizer of all the generative models.
        flag = 1
        for l in range(len(self.model.layers)):
            L = self.model.layers[l]
            if isinstance(L, ConvSparseCoding) or isinstance(L, SparseCodingLayer):
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
                Y  = self.compiled_fprop[l](X)


class CompositeTrainer():
    
    def __init__(self, cost, model, dataset, monitoring_dataset, lr, save_path, epochs):
        self.model = model
        self.dataset = dataset
        self.monitoring_dataset = monitoring_dataset
        self.lr = lr
        self.input = model.get_input_space().make_theano_batch()
        cost=cost.expr(model, self.input)
        params = self.model.get_params()
        self.opt = top.Optimizer(params[0], cost, method='sgd', learning_rate=self.lr)
        
        self.compiled_fprop = []
        for L in self.model.layers:
            X = L.get_input_space().make_theano_batch()
            if isinstance(X, tuple):
                self.compiled_fprop.append( theano.function(X, L.fprop(X), on_unused_input='ignore') )
            else:
                self.compiled_fprop.append( theano.function([X], L.fprop(X), on_unused_input='ignore') )
        
        self.save_path = save_path
        self.epochs = epochs
        self.valid_best = 10e20
        self.test_best = 0

    def main_loop(self):
        
        for e in xrange(self.epochs):
            print "\n============ \nEpoch: %d" % e
            self.train_epoch()
            if 'valid' in self.monitoring_dataset.keys():
                self.validate_and_save()
            self.print_monitor()
            #do any update
    
    def train_epoch(self):
        data_size = self.dataset.get_data()[0].shape[0]
        data_time = self.dataset.get_data()[0].shape[1]
        # Get the full training batch
        iternum = 0
        for data_batch in self.dataset.iterator('shuffled_sequential', self.model.batch_size):
            try:
                data_batch = self.dataset.view_converter.design_mat_to_topo_view(data_batch)
            except:
                pass
            # Iterate through time
            #print iternum
            iternum += 1
            #print iternum
            self._train_err = 0.
            for t in range(data_time):
                self.update_hidden_codes(data_batch , t)
                _, err = self.opt.run(1)
                self._train_err += err
            #print self._train_err
    
    def validate_and_save(self):
        data_size = self.monitoring_dataset['valid'].get_data()[0].shape[0]
        data_time = self.monitoring_dataset['valid'].get_data()[0].shape[1]

        iternum = 0
        for data_batch in self.monitoring_dataset['valid'].iterator('sequential', 
                                                          self.model.batch_size):
            try:
                data_batch = self.monitoring_dataset['valid'].view_converter.design_mat_to_topo_view(data_batch)
            except:
                pass
            self._valid_err = 0.
            for t in range(data_time):
                self.update_hidden_codes(data_batch[:,t:t+1,:,:])
                err = self.opt.g()
                self._valid_err += err
        
        if 'test' in self.monitoring_dataset.keys():
            data_size = self.monitoring_dataset['test'].get_data()[0].shape[0]
            for data_batch in self.monitoring_dataset['test'].iterator('sequential', self.model.batch_size):
                try:
                    data_batch = self.monitoring_dataset['test'].view_converter.design_mat_to_topo_view(data_batch)
                except:
                    pass
                self._test_err = 0.
                for t in range(data_time):
                    self.update_hidden_codes(data_batch[:,t:t+1,:,:])
                    err = self.opt.g()
                    self._test_err += err
                
        if self.valid_best < self._valid_err:
            self.valid_best = self._valid_err
            self.test_best  = self._test_err
            print "Saving best results..."
            cPickle.dump(self.model, file(self.save_path, 'w'), -1)        
    
    def print_monitor(self):
        print "Training obj: %f" % self._train_err
        try:
            print 'Valid obj: %f' % self._valid_err
        except:
            pass
        try:
            print 'Test obj: %f' % self._test_err
        except:
            pass
        
    
    def update_hidden_codes(self, batch, t):
        # Run top optmizer of all the generative models.
        flag = 1

        if isinstance(batch, tuple):
            nbatch = []
            for c in range(len(batch)-1):
                nbatch += [batch[c][:,t:t+1,:,:]]
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


