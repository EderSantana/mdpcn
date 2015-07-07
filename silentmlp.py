from pylearn2.models.mlp import MLP, Layer
from pylearn2.utils import wraps
from theano.compat.python2x import OrderedDict
from pylearn2.monitor import get_monitor_doc

class SilentMLP(MLP):
	'''
	This is the regular Pylearn2 MLP, go there for docs.
	Here we just include and extra option of excluding the extra
	elements at the 'get_layer_monitoring_channels' OrderedDict
        '''

	def __init__(self, verbose=False, **kwargs):
            super(SilentMLP, self).__init__(**kwargs)
            self.verbose = verbose

	@wraps(Layer.get_layer_monitoring_channels)
        def get_layer_monitoring_channels(self, state_below=None,
                                        state=None, targets=None):

            rval = OrderedDict()
            state = state_below

            for layer in self.layers:
                # We don't go through all the inner layers recursively
                state_below = state
                state = layer.fprop(state)
                args = [state_below, state]
                if layer is self.layers[-1] and targets is not None:
                    args.append(targets)
                ch = layer.get_layer_monitoring_channels(*args)
                if not isinstance(ch, OrderedDict):
                    raise TypeError(str((type(ch), layer.layer_name)))
                for key in ch:
                    value = ch[key]
                    doc = get_monitor_doc(value)
                    if doc is None:
                        doc = str(type(layer)) + \
                            ".get_monitoring_channels_from_state did" + \
                            " not provide any further documentation for" + \
                            " this channel."
                    doc = 'This channel came from a layer called "' + \
                            layer.layer_name + '" of an MLP.\n' + doc
                    value.__doc__ = doc
                    rval[layer.layer_name+'_'+key] = value
            
            return rval

        @wraps(Layer.get_monitoring_channels)
        def get_monitoring_channels(self, data):
            # if the MLP is the outer MLP \
            # (ie MLP is not contained in another structure)

            X, Y = data
            state = X
            rval = self.get_layer_monitoring_channels(state_below=X,
                                                        targets=Y)
            # Here is the contribution
            #if self.verbose:
            try:
                rval = OrderedDict([('y_misclass', rval['y_misclass']), 
                                  ('y_nll', rval['y_nll'])
                                  ])
            except:
                rval = OrderedDict()
            return rval
