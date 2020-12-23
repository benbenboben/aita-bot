import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.pipeline import make_pipeline
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers





class KerasTokenizerTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, **tokenizer_params):
        self.tokenizer = Tokenizer(num_words=tokenizer_params.pop('num_words', None))
        # super().__init__(num_words=tokenizer_params.pop('num_words', None))

    def fit(self, X, y=None):
        self.tokenizer.fit_on_texts(X)
        return self

    def transform(self, X, y=None):
        X_transformed = self.tokenizer.texts_to_sequences(X)
        return X_transformed


class KerasPadSequencesTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, maxlen=500):
        self.maxlen = maxlen

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_padded = pad_sequences(X, maxlen=self.maxlen, padding='post')
        return X_padded


def read_glove(fname, vocab_size, tokenizer):
    embeddings_dictionary = dict()

    glove_file = open(fname, encoding="utf8")
    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions

    glove_file.close()

    embedding_matrix = np.zeros((vocab_size, 100))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

    return embedding_matrix


def create_model(embedding_dim, embedding_matrix, vocab_size, maxlen):
    model = Sequential()
    model.add(layers.Embedding(
        vocab_size,
        embedding_dim,
        input_length=maxlen,
        weights=[embedding_matrix],
        trainable=False)
    )
    model.add(layers.Conv1D(256, 5, activation='relu'))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Dropout(0.5))
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv1D(256, 5, activation='relu'))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Dropout(0.5))
    model.add(layers.Bidirectional(layers.LSTM(64)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])


