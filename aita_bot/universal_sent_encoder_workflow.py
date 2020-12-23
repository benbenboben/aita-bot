import numpy as np
import functools
import tensorflow_hub as hub
import tensorflow as tf
import nltk
from nltk.tokenize import sent_tokenize


@functools.lru_cache(1)
def load_embed(a=1):
    """Load the universal sentence encoder model in memoized fashion.

    :param a: useless variable
    :return: USE model
    """
    return hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')


class EmbeddingAveragingGenerator(tf.keras.utils.Sequence):
    """
    Object for transforming text to vector sequences from a dataframe for batch learning with RNNs.
    """

    def __init__(self, df, x_col, y_col=None, batch_size=32, n_classes=None, shuffle=True):
        """Instantiate object.

        :param df: dataframe containing (at least) text column and labels column
        :param x_col: name of text column
        :param y_col: name of labels column
        :param batch_size: number of samples to return for training
        :param n_classes: total number of classes in dataset
        :param shuffle: shuffle between epochs
        """
        self.batch_size = batch_size
        self.df = df
        self.indices = self.df.index.tolist()
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.x_col = x_col
        self.y_col = y_col
        self.embed = load_embed()
        self.index = None
        self.on_epoch_end()

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]
        X, y = self.__get_data(batch)
        return X, y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        X, y = self.reshaper(
            self.df.loc[batch][self.x_col],
            self.df.loc[batch][self.y_col]
        )

        return X, y

    def reshaper(self, X, y=None):
        """This function essentially performs pre-padding zero vectors when the number of embedding vectors is less
        than the max seq length.  If it's larger there will be an error so buckle up.

        :param X: text documents as iterable
        :param y: labels (does not need to be one hot)
        :return: X of dim (-1, seq length, embed dim)
        """
        Xout = np.empty(shape=(len(X), 512))
        yout = np.zeros(shape=(len(X), self.n_classes))

        for idx, (xx, yy) in enumerate(zip(X, y)):
            arr = self.embed(sent_tokenize(xx))
            Xout[idx, :] = np.mean(arr, axis=0)

            if yy is not None:
                yout[idx, yy] = 1

        return Xout, yout


class VectorSequenceClassificationGenerator(tf.keras.utils.Sequence):
    """
    Object for transforming text to vector sequences from a dataframe for batch learning with RNNs.
    """

    def __init__(self, df, x_col, y_col=None, batch_size=32, n_classes=None,
                 shuffle=True, max_seq_len=100, embed_dim=512):
        """Instantiate object.

        :param df: dataframe containing (at least) text column and labels column
        :param x_col: name of text column
        :param y_col: name of labels column
        :param batch_size: number of samples to return for training
        :param n_classes: total number of classes in dataset
        :param shuffle: shuffle between epochs
        :param max_seq_len: maximum number of vectors for sequence
        :param embed_dim: size of embedding vector for each sentence (in case of USE=512).
        """
        self.batch_size = batch_size
        self.df = df
        self.indices = self.df.index.tolist()
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.x_col = x_col
        self.y_col = y_col
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.embed = load_embed()
        self.index = None
        self.on_epoch_end()

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]
        X, y = self.__get_data(batch)
        return X, y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        X, y = self.reshaper(
            self.df.loc[batch][self.x_col],
            self.df.loc[batch][self.y_col]
        )

        return X, y

    def reshaper(self, X, y=None):
        """This function essentially performs pre-padding zero vectors when the number of embedding vectors is less
        than the max seq length.  If it's larger there will be an error so buckle up.

        :param X: text documents as iterable
        :param y: labels (does not need to be one hot)
        :return: X of dim (-1, seq length, embed dim)
        """
        Xout = np.zeros(shape=(len(X), self.max_seq_len, self.embed_dim))
        yout = np.zeros(shape=(len(X), self.n_classes))

        for idx, (xx, yy) in enumerate(zip(X, y)):
            arr = self.embed(xx)

            Xout[idx, :arr.shape[0], :arr.shape[1]] = arr

            if yy is not None:
                yout[idx, yy] = 1

        return Xout, yout


