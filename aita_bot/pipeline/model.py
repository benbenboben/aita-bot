import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


class KerasTokenizeAndPadTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, num_words=50000, maxlen=500, document_count=0):
        self.num_words = num_words
        self.maxlen = maxlen
        self.vocab_size = None
        self.document_count = document_count
        self.tokenizer = Tokenizer(num_words=num_words, document_count=document_count)

    def fit(self, X, y=None):
        self.tokenizer.fit_on_texts(X)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        return self

    def transform(self, X, y=None):
        Xtransformed = self.tokenizer.texts_to_sequences(X)
        Xtransformed = pad_sequences(Xtransformed, padding='post', maxlen=self.maxlen)
        return Xtransformed


def read_glove(fname, vocab_size, tokenizer, embedding_dim):
    embeddings_dictionary = dict()

    glove_file = open(fname, encoding="utf8")
    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions

    glove_file.close()

    embedding_matrix = np.zeros((vocab_size, embedding_dim))
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
    model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv1D(256, 5, activation='relu'))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Dropout(0.5))
    model.add(layers.Bidirectional(layers.LSTM(128)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
