__authors__ = "Eder Santana"
__credits__ = ["Eder Santana"]
__license__ = "3-clause BSD"
__maintainer__ = "Eder Santana"
__email__ = "edersantana@ufl"

import numpy as N
np = N
from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets import control
from pylearn2.utils import serial
from pylearn2.utils.mnist_ubyte import read_mnist_images
from pylearn2.utils.mnist_ubyte import read_mnist_labels
from pylearn2.utils.rng import make_np_rng


class VidTIMIT(dense_design_matrix.DenseDesignMatrix):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    which_set : WRITEME
    center : WRITEME
    shuffle : WRITEME
    one_hot : WRITEME
    binarize : WRITEME
    start : WRITEME
    stop : WRITEME
    axes : WRITEME
    preprocessor : WRITEME
    fit_preprocessor : WRITEME
    fit_test_preprocessor : WRITEME
    """
    def __init__(self, which_set, center=False, shuffle=False,
                 one_hot=False, binarize=False, start=None,
                 stop=None, axes=['b', 0, 1, 'c'],
                 preprocessor=None,
                 fit_preprocessor=False,
                 fit_test_preprocessor=False):

        self.args = locals()

        if which_set not in ['train', 'test']:
            if which_set == 'valid':
                raise ValueError(
                    "There is no such thing as the MNIST validation set. MNIST"
                    "consists of 60,000 train examples and 10,000 test"
                    "examples. If you wish to use a validation set you should"
                    "divide the train set yourself. The pylearn2 dataset"
                    "implements and will only ever implement the standard"
                    "train / test split used in the literature.")
            raise ValueError(
                'Unrecognized which_set value "%s".' % (which_set,) +
                '". Valid values are ["train","test"].')

        def dimshuffle(b01c):
            default = ('b', 0, 1, 'c')
            return b01c.transpose(*[default.index(axis) for axis in axes])

        if control.get_load_data():
            path = "${VIDTIMIT}/data/"
            if which_set == 'train':
                im_path = path + 'train.npy'
                label_path = path + 'train-labels.npy'
            else:
                assert which_set == 'test'
                im_path = path + 'test.npy'
                label_path = path + 'test-labels.npy'
            # Path substitution done here in order to make the lower-level
            # mnist_ubyte.py as stand-alone as possible (for reuse in, e.g.,
            # the Deep Learning Tutorials, or in another package).
            im_path = serial.preprocess(im_path)
            label_path = serial.preprocess(label_path)
            topo_view = np.load(im_path)
            y = np.load(label_path)

            if binarize:
                topo_view = (topo_view > 0.5).astype('float32')

            self.one_hot = one_hot
            if one_hot:
                one_hot = N.zeros((y.shape[0], 36), dtype='float32')
                for i in xrange(y.shape[0]):
                    one_hot[i, y[i]] = 1.
                y = one_hot
                max_labels = None
            else:
                max_labels = 36

            m, r, c = topo_view.shape
            assert r == 32
            assert c == 32
            topo_view = topo_view.reshape(m, r, c, 1)

            if which_set == 'train':
                assert m == 27280
            elif which_set == 'test':
                assert m == 10929
            else:
                assert False

            if center:
                topo_view -= topo_view.mean(axis=0)

            if shuffle:
                self.shuffle_rng = make_np_rng(None, [1, 2, 3], which_method="shuffle")
                for i in xrange(topo_view.shape[0]):
                    j = self.shuffle_rng.randint(m)
                    # Copy ensures that memory is not aliased.
                    tmp = topo_view[i, :, :, :].copy()
                    topo_view[i, :, :, :] = topo_view[j, :, :, :]
                    topo_view[j, :, :, :] = tmp
                    # Note: slicing with i:i+1 works for one_hot=True/False
                    tmp = y[i:i+1].copy()
                    y[i] = y[j]
                    y[j] = tmp

            super(VidTIMIT, self).__init__(topo_view=dimshuffle(topo_view), y=y,
                                        axes=axes, max_labels=max_labels)

            assert not N.any(N.isnan(self.X))

            if start is not None:
                assert start >= 0
                if stop > self.X.shape[0]:
                    raise ValueError('stop=' + str(stop) + '>' +
                                     'm=' + str(self.X.shape[0]))
                assert stop > start
                self.X = self.X[start:stop, :]
                if self.X.shape[0] != stop - start:
                    raise ValueError("X.shape[0]: %d. start: %d stop: %d"
                                     % (self.X.shape[0], start, stop))
                if len(self.y.shape) > 1:
                    self.y = self.y[start:stop, :]
                else:
                    self.y = self.y[start:stop]
                assert self.y.shape[0] == stop - start
        else:
            # data loading is disabled, just make something that defines the
            # right topology
            topo = dimshuffle(np.zeros((1, 32, 32, 1)))
            super(VidTIMIT, self).__init__(topo_view=topo, axes=axes)
            self.X = None

        if which_set == 'test':
            assert fit_test_preprocessor is None or \
                (fit_preprocessor == fit_test_preprocessor)

        if self.X is not None and preprocessor:
            preprocessor.apply(self, fit_preprocessor)

    def adjust_for_viewer(self, X):
        return N.clip(X * 2. - 1., -1., 1.)

    def adjust_to_be_viewed_with(self, X, other, per_example=False):
        return self.adjust_for_viewer(X)

    def get_test_set(self):
        args = {}
        args.update(self.args)
        del args['self']
        args['which_set'] = 'test'
        args['start'] = None
        args['stop'] = None
        args['fit_preprocessor'] = args['fit_test_preprocessor']
        args['fit_test_preprocessor'] = None
        return VidTIMIT(**args)
