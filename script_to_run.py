import numpy as np
from pylab import *
import theano
import pylearn2.models.mlp
import cdpcn
import silentmlp
#import vidtimit
from fullvidtimit import GOKVIDTIMIT
#reload(vidtimit)
import trainer
from trainer import Trainer, CompositeTrainer
#reload(trainer)
from pylearn2.space import CompositeSpace, Conv2DSpace, IndexSpace
#reload(cdpcn)
import cPickle
dataset = GOKVIDTIMIT('train', axes=('b','c',0,1), center=True)
monitoring_dataset = {'nada': GOKVIDTIMIT('test', axes=('b','c',0,1), center=True)}
from pylearn2.space import Conv2DSpace
from pylearn2.models.mlp import IdentityConvNonlinearity, RectifierConvNonlinearity
dim = 4
chan = 16
model = silentmlp.SilentMLP(
      batch_size=8,
      layers= [
               cdpcn.CompositeConvLin(
                   fprop_code = True,
                   top_most = True,
                   batch_size = 8,
                   lr = .001,
                   n_steps = 100,
                   output_channels = chan,
                   dim = dim,
                   kernel_shape= [5, 5],
                   pool_shape= [1, 1],
                   pool_stride= [1, 1],
                   pool_type = 'max',
                   irange = .0001,
                   layer_name = 'h0',
                   nonlinearity = IdentityConvNonlinearity()
               ),
               pylearn2.models.mlp.Softmax (
                   max_col_norm= 1.9365,
                   layer_name= 'y',
                   n_classes= 10,
                   irange= .005
              )
              ],
      input_space=CompositeSpace([Conv2DSpace(shape=[32,32], num_channels=1, axes=['b','c',0,1]),
            Conv2DSpace(shape=[dim,1], num_channels=1, axes=['b','c',0,1])]),
      input_source=('video', 'audio')
)
from sparse_costs import SparseReconstructionError
cost = SparseReconstructionError()
trainer_obj = trainer.CompositeTrainer(cost, model, dataset, monitoring_dataset, .06, './results.pkl',100)
trainer_obj.main_loop()
import cPickle
cPickle.dump(model, file('goktug_composite_model.pkl', 'w'), -1)
