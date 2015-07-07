"""
.. todo::

    WRITEME
"""
import os
import cPickle
import logging
_logger = logging.getLogger(__name__)

import numpy as np
import warnings
N = np
from pylearn2.datasets import cache, dense_design_matrix
from pylearn2.expr.preprocessing import global_contrast_normalize
from pylearn2.utils import contains_nan
from pylearn2.utils import serial


class VIDTIMIT(dense_design_matrix.DenseDesignMatrix):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    which_set : str
        One of 'train', 'test'
    center : WRITEME
    rescale : WRITEME
    gcn : float, optional
        Multiplicative constant to use for global contrast normalization.
        No global contrast normalization is applied, if None
    one_hot : WRITEME
    start : WRITEME
    stop : WRITEME
    axes : WRITEME
    toronto_prepro : WRITEME
    preprocessor : WRITEME
    """

    def __init__(self, which_set, center=False, rescale=False, gcn=None,
                 one_hot=None, start=None, stop=None, axes=('b', 'c', 0, 1),
                 preprocessor = None, noise_v=0.):
        # note: there is no such thing as the cifar10 validation set;
        # pylearn1 defined one but really it should be user-configurable
        # (as it is here)
        self.noise_v = noise_v
        self.axes = axes

        # we define here:
        dtype = 'float32'
        ntrain = 288
        nvalid = 0  # artefact, we won't use it
        ntest = 72

        # we also expose the following details:
        self.img_shape = (1, 32, 32)
        self.img_size = N.prod(self.img_shape)
        self.n_classes = 36
        self.label_names = ['fadg0', 'fcft0', 'bird', 'cat', 'deer',
                            'dog', 'frog', 'horse', 'ship', 'truck']

        # prepare loading
        '''
        fnames = ['data_batch_%i' % i for i in range(1, 6)]
        lenx = N.ceil((ntrain + nvalid) / 10000.)*10000
        x = N.zeros((lenx, self.img_size), dtype=dtype)
        y = N.zeros((lenx, 1), dtype=dtype)

        # load train data
        nloaded = 0
        for i, fname in enumerate(fnames):
            data = CIFAR10._unpickle(fname)
            x[i*10000:(i+1)*10000, :] = data['data']
            y[i*10000:(i+1)*10000, 0] = data['labels']
            nloaded += 10000
            if nloaded >= ntrain + nvalid + ntest:
                break
        '''
        path = os.path.join(serial.preprocess('${VIDTIMIT}'), 'data', 'cut_dataset.pkl')
        dataset = cPickle.load(file(path))

        # process this data
        train_idx = dataset['train_idx']==1
        test_idx = dataset['test_idx']==1

        if noise_v==0.:
            v_noise_tr = 0.
            v_noise_te = 0.
        else:
            v_noise_tr = np.random.normal(0., noise_v, size=dataset['data'][train_idx].reshape((-1,54*32*32)).shape)
            v_noise_te = np.random.normal(0., noise_v, size=dataset['data'][test_idx].reshape((-1,54*32*32)).shape)

        Xs = {'train': dataset['data'][train_idx].reshape((-1,54*32*32)) + v_noise_tr,
              'test': dataset['data'][test_idx].reshape((-1,54*32*32)) + v_noise_te}

        Ys = {'train': dataset['labels'][train_idx],
              'test': dataset['labels'][test_idx]}

        X = N.cast['float32'](Xs[which_set])
        y = Ys[which_set]

        if isinstance(y, list):
            y = np.asarray(y).astype(dtype)

        if which_set == 'test':
            assert y.shape[0] == 72
            y = y.reshape((y.shape[0], 1))

        max_labels = 36
        if one_hot is not None:
            ynew = np.zeros((y.shape[0], 36))
            for i in range(y.shape[0]):
                ynew[y[i]] = 1
            warnings.warn("the `one_hot` parameter is deprecated. To get "
                          "one-hot encoded targets, request that they "
                          "live in `VectorSpace` through the `data_specs` "
                          "parameter of MNIST's iterator method. "
                          "`one_hot` will be removed on or after "
                          "September 20, 2014.", stacklevel=2)
            y = ynew.astype('int32')

        if center:
            X -= 0.2845
        self.center = center

        if rescale:
            X /= .1644
        self.rescale = rescale

        if start is not None:
            # This needs to come after the prepro so that it doesn't
            # change the pixel means computed above for toronto_prepro
            assert start >= 0
            assert stop > start
            assert stop <= X.shape[0]
            X = X[start:stop, :]
            y = y[start:stop, :]
            assert X.shape[0] == y.shape[0]

        if which_set == 'test':
            assert X.shape[0] == 72

        view_converter = dense_design_matrix.DefaultViewConverter((32, 32, 54),
                                                                  axes)

        super(VIDTIMIT, self).__init__(X=X, y=y, view_converter=view_converter,
                                      y_labels=self.n_classes)

        assert not contains_nan(self.X)

        if preprocessor:
            preprocessor.apply(self)

    def adjust_for_viewer(self, X):
        """
        .. todo::

            WRITEME
        """
        # assumes no preprocessing. need to make preprocessors mark the
        # new ranges
        rval = X.copy()

        # patch old pkl files
        if not hasattr(self, 'center'):
            self.center = False
        if not hasattr(self, 'rescale'):
            self.rescale = False

        if not self.center:
            rval -= .2845

        if not self.rescale:
            rval /= .1644

        rval = np.clip(rval, -1., 1.)

        return rval

    def adjust_to_be_viewed_with(self, X, orig, per_example=False):
        """
        .. todo::

            WRITEME
        """
        # if the scale is set based on the data, display X oring the
        # scale determined by orig
        # assumes no preprocessing. need to make preprocessors mark
        # the new ranges
        rval = X.copy()

        # patch old pkl files
        if not hasattr(self, 'center'):
            self.center = False
        if not hasattr(self, 'rescale'):
            self.rescale = False
        if not self.center:
            rval -= .2845
        if not self.rescale:
            rval /= .1644

        rval = np.clip(rval, -1., 1.)

        return rval

    def get_test_set(self):
        """
        .. todo::

            WRITEME
        """
        return VIDTIMIT(which_set='test', center=self.center,
                       rescale=self.rescale,
                       axes=self.axes)
