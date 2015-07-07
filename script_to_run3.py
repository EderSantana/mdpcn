import numpy as np
from pylab import *
import theano
import pylearn2.models.mlp
import cdpcn
import silentmlp
#import vidtimit
from fullvidtimit import FULLVIDTIMIT
#reload(vidtimit)
import trainer
from trainer import Trainer, CompositeTrainer
#reload(trainer)
from pylearn2.space import CompositeSpace, Conv2DSpace, IndexSpace
#reload(cdpcn)
import cPickle
dataset = FULLVIDTIMIT('train', axes=('b','c',0,1), center=True)
monitoring_dataset = {'nada': FULLVIDTIMIT('test', axes=('b','c',0,1), center=True)}
from pylearn2.space import Conv2DSpace
from pylearn2.models.mlp import IdentityConvNonlinearity, RectifierConvNonlinearity
dim = 1000
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
cPickle.dump(model, file('composite_model.pkl', 'w'), -1)
