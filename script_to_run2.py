import numpy as np
from pylab import *
import theano
import pylearn2.models.mlp
import cdpcn
import silentmlp
#import vidtimit
from vidtimit import VIDTIMIT
#reload(vidtimit)
from trainer import Trainer

import cPickle
dataset = VIDTIMIT('train', axes=('b','c',0,1), center=True)
monitoring_dataset = {'nada': VIDTIMIT('test', axes=('b','c',0,1))}

from pylearn2.space import Conv2DSpace
from pylearn2.models.mlp import IdentityConvNonlinearity, RectifierConvNonlinearity
chan = 16
model = silentmlp.SilentMLP(
      batch_size=8,
      layers= [
               cdpcn.ConvSparseCoding(
                   fprop_code = True,
                   top_most = False,
                   batch_size = 8,
                   lr = .001,
                   n_steps = 100,
                   output_channels = chan,
                   kernel_shape= [5, 5],
                   pool_shape= [2, 2],
                   pool_stride= [2, 2],
                   pool_type = 'max',
                   irange = .0001,
                   layer_name = 'h0',
                   nonlinearity = RectifierConvNonlinearity()
               ),
               cdpcn.ConvSparseCoding(
                   fprop_code = True,
                   top_most = True,
                   batch_size = 8,
                   lr = .001,
                   n_steps = 100,
                   input_channels = chan,
                   output_channels = chan,
                   kernel_shape= [5, 5],
                   pool_shape= [1, 1],
                   pool_stride= [1, 1],
                   pool_type = 'max',
                   irange = .0001,
                   layer_name = 'h1',
                   nonlinearity = IdentityConvNonlinearity()
               ),
               pylearn2.models.mlp.Softmax (
                   max_col_norm= 1.9365,
                   layer_name= 'y',
                   n_classes= 10,
                   irange= .005
              )
              ],
      input_space=Conv2DSpace(
            shape= [32, 32],
            num_channels= 1,
            axes=('b','c',0,1)
      )
)

from sparse_costs import ConvSparseReconstructionError
cost = ConvSparseReconstructionError()
trainer = Trainer(cost, model, dataset, monitoring_dataset, .06, './results.pkl',100)

trainer.main_loop()

cPickle.dump(model, file('model.pkl','w'), -1)
