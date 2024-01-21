
import numpy as np

from abc import ABC, abstractmethod
from skimage.util.shape import view_as_windows

from autoencoder.vae import VariationalAutoencoderImage
from autoencoder.vae import VariationalAutoencoderTimeSeries
from autoencoder.cvae import ConditionalVariationalAutoencoderImage
from autoencoder.cvae import ConditionalVariationalAutoencoderTimeSeries


class Pooler(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def transform(self, X):
        pass

    @abstractmethod
    def inverse_transform(self, X):
        pass


class ImageAutoencoder(Pooler):

    def __init__(self, input_shape, latent_dim, n_classes=2, ae_type='cvae', path_ae='./', name='cvae'):
        super().__init__()

        kernel_size = (3, 3)
        strides = (1, 1)
        filters = 16
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        num_conv_layers = 2
        hidden_dim = 16
        use_mse = True
        if not use_mse:
            optimizer = 'adam'
        else:
            optimizer = 'rmsprop'

        self.ae_type = ae_type
        if self.ae_type == 'vae':
            self.ae = VariationalAutoencoderImage(input_shape, latent_dim=latent_dim, num_conv_layers=num_conv_layers,
                                                  filters=filters, kernel_size=kernel_size, strides=strides,
                                                  hidden_dim=hidden_dim, use_mse=use_mse, optimizer=optimizer,
                                                  store_intermediate=False, save_graph=False, path=path_ae,
                                                  name=name, verbose=0)
        else:
            self.ae = ConditionalVariationalAutoencoderImage(input_shape, n_classes, latent_dim=latent_dim,
                                                             num_conv_layers=num_conv_layers, filters=filters,
                                                             kernel_size=kernel_size, strides=strides,
                                                             hidden_dim=hidden_dim, use_mse=use_mse,
                                                             optimizer=optimizer, store_intermediate=False,
                                                             save_graph=False, path=path_ae, name=name, verbose=0)
        self.ae.load_model()

    def transform(self, X, y=None):
        if self.ae_type == 'vae':
            X_p = self.ae.encode(X)[2]
        else:
            y = y if y is not None else np.random.choice(self.n_classes, len(X))
            y_oh = self.ae.oh_encoder.transform(y.reshape(-1, 1)).toarray()
            X_p = self.ae.encode([X, y_oh])[2]
        return X_p

    def inverse_transform(self, X, y=None):
        if self.ae_type == 'vae':
            X_tilde = self.ae.decode(X)
        else:
            y = y if y is not None else np.random.choice(self.n_classes, len(X))
            y_oh = self.ae.oh_encoder.transform(y.reshape(-1, 1)).toarray()
            X_tilde = self.ae.decode([X, y_oh])
        return X_tilde


class ImagePooler(Pooler):

    def __init__(self, dims, window):
        super().__init__()
        self.w, self.h = dims[0], dims[1]
        self.w1, self.h1 = window[0], window[1]
        self.div_w = (self.w - self.w1) // self.w1 + 1
        self.div_h = (self.h - self.h1) // self.h1 + 1

    def transform(self, X, agg=np.mean):
        X_p = list()
        if self.w1 == 1 and self.h1 == 1:
            for x in X:
                X_p.append(x.flatten())
        else:
            for x in X:
                x_v = view_as_windows(x.reshape(self.w, self.h),
                                      window_shape=(self.w1, self.h1), step=(self.w1, self.h1))
                length = x_v.shape[0] * x_v.shape[1]
                x_v = x_v.reshape(length, x_v.shape[2], x_v.shape[3])
                x_p = np.zeros(len(x_v))
                for i, v in enumerate(x_v):
                    x_p[i] = agg(v)
                X_p.append(x_p)
        X_p = np.array(X_p)
        return X_p

    def inverse_transform(self, X):
        X_tilde = list()
        for x in X:
            x_tilde = np.zeros((self.w, self.h))
            for idx, v in enumerate(x):
                j = (idx // self.div_w) * self.w1
                i = (idx % self.div_h) * self.h1
                upv = np.zeros((self.w1, self.h1)) + v
                x_tilde[j:j + self.h1, i:i + self.w1] = upv
            X_tilde.append(x_tilde)
        X_tilde = np.array(X_tilde)
        X_tilde = np.expand_dims(X_tilde, -1)
        return X_tilde


class ImageIdentity(Pooler):

    def __init__(self, dims):
        self.w, self.h = dims[0], dims[1]
        super().__init__()

    def transform(self, X):
        X_p = list()
        for x in X:
            X_p.append(x.flatten())
        X_p = np.array(X_p)
        return X_p

    def inverse_transform(self, X):
        X_tilde = list()
        for x in X:
            X_tilde.append(np.reshape(x, (self.w, self.h)))
        X_tilde = np.array(X_tilde)
        X_tilde = np.expand_dims(X_tilde, -1)
        return X_tilde


class TsAutoencoder(Pooler):

    def __init__(self, input_shape, latent_dim, n_classes=2, ae_type='cvae', path_ae='./', name='cvae'):
        super().__init__()

        kernel_size = 4
        strides = 1
        filters = 16
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        num_conv_layers = 2
        hidden_dim = 16
        use_mse = False
        if not use_mse:
            optimizer = 'adam'
        else:
            optimizer = 'rmsprop'

        self.ae_type = ae_type
        if self.ae_type == 'vae':
            self.ae = VariationalAutoencoderTimeSeries(input_shape, latent_dim=latent_dim,
                                                       num_conv_layers=num_conv_layers,
                                                       filters=filters, kernel_size=kernel_size, strides=strides,
                                                       hidden_dim=hidden_dim, use_mse=use_mse, optimizer=optimizer,
                                                       store_intermediate=False, save_graph=False, path=path_ae,
                                                       name=name, verbose=0)
        else:
            self.ae = ConditionalVariationalAutoencoderTimeSeries(input_shape, n_classes, latent_dim=latent_dim,
                                                             num_conv_layers=num_conv_layers,
                                                             filters=filters, kernel_size=kernel_size, strides=strides,
                                                             hidden_dim=hidden_dim, use_mse=use_mse, optimizer=optimizer,
                                                             store_intermediate=False, save_graph=False,
                                                             path=path_ae, name=name, verbose=0)
        self.ae.load_model()

    def transform(self, X, y=None):
        if self.ae_type == 'vae':
            X_p = self.ae.encode(X)[2]
        else:
            y = y if y is not None else np.random.choice(self.n_classes, len(X))
            y_oh = self.ae.oh_encoder.transform(y.reshape(-1, 1)).toarray()
            X_p = self.ae.encode([X, y_oh])[2]
        return X_p

    def inverse_transform(self, X, y=None):
        if self.ae_type == 'vae':
            X_tilde = self.ae.decode(X)
        else:
            y = y if y is not None else np.random.choice(self.n_classes, len(X))
            y_oh = self.ae.oh_encoder.transform(y.reshape(-1, 1)).toarray()
            X_tilde = self.ae.decode([X, y_oh])
        return X_tilde


class TsPooler(Pooler):

    def __init__(self, dims, window):
        super().__init__()
        self.dims = dims
        self.window = window
        self.w = dims[0]
        self.w1 = window[0]
        self.div_w = (self.w - self.w1) // self.w1 + 1

    def transform(self, X, agg=np.mean):
        X_p = list()
        if self.w1 == 1:
            for x in X:
                X_p.append(x.flatten())
        else:
            for x in X:
                x_v = view_as_windows(x.reshape(self.w),
                                      window_shape=self.w1, step=self.w1)
                x_p = np.zeros(len(x_v))
                for i, v in enumerate(x_v):
                    x_p[i] = agg(v)
                X_p.append(x_p)
        X_p = np.array(X_p)
        return X_p

    def inverse_transform(self, X):
        X_tilde = list()
        for x in X:
            x_tilde = np.zeros(self.w)
            for idx, v in enumerate(x):
                i = (idx % self.div_w) * self.w1
                upv = np.zeros(self.w1) + v
                x_tilde[i:i + self.w1] = upv
            X_tilde.append(x_tilde)
        X_tilde = np.array(X_tilde)
        X_tilde = np.expand_dims(X_tilde, -1)
        return X_tilde


class TsIdentity(Pooler):

    def __init__(self, dims):
        self.w = dims[0]
        super().__init__()

    def transform(self, X):
        X_p = list()
        for x in X:
            X_p.append(x.flatten())
        X_p = np.array(X_p)
        return X_p

    def inverse_transform(self, X):
        X_tilde = list()
        for x in X:
            X_tilde.append(np.reshape(x, self.w))
        X_tilde = np.array(X_tilde)
        X_tilde = np.expand_dims(X_tilde, -1)
        return X_tilde
