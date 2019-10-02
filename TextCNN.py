# -*- coding: utf-8 -*-
# @Time    : 2019-07-22 15:32
# @Author  : finupgroup
# @FileName: TextCNN.py
# @Software: PyCharm

# -*- coding: utf-8 -*-
# @Time    : 2019-07-19 18:22
# @Author  : finupgroup
# @FileName: Bi-LSTM-Attention.py
# @Software: PyCharm

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, Callback
import numpy as np
from sklearn.preprocessing import LabelEncoder
import keras
import jieba
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import initializers
import tensorflow.keras.backend as K


class Attention(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/pdf/1512.08756.pdf]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:

        """
        self.supports_masking = True
        # self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias

        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.step_dim = input_shape[1]
        assert len(input_shape) == 3  # batch ,timestep , num_features
        print(input_shape)
        self.W = self.add_weight((input_shape[-1],),  # num_features
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),  # timesteps
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        print(K.reshape(x, (-1, features_dim)))# n, d
        print(K.reshape(self.W, (features_dim, 1)))# w= dx1
        print(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))))#nx1

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))#batch,step
        print(eij)
        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        print(a)
        a = K.expand_dims(a)
        print("expand_dims:")
        print(a)
        print("x:")
        print(x)
        weighted_input = x * a
        print(weighted_input.shape)
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return input_shape[0], self.features_dim


class Tokenizer:
    def __init__(self, num_words=10000,
                 dict_path="/Users/finupgroup/Desktop/FF/深度学习/merge_sgns_bigram_char300.txt",
                 stopwords_path="stopwords.txt",
                 m=300):
        """

        :param num_words: int
        :param dict_path: dict path
        :param stopwords_path: stopwords path
        :param m: dict Dimension
        """
        self.num_words = num_words
        self.dict_path = dict_path
        self.stopwords_path = stopwords_path
        self.m = m
        self.dict_index = {}
        self.sentences_seg = []
        self.w2id = {}
        self.embedding_matrix = None
        self.value_cnt = pd.Series()
        self.stopwords = []

    def fit_on_texts(self, data):
        # 初始化词典
        # 选择m=300的预训练数据,将预训练的词与vector提取到dict_index中存储起来
        with open(self.dict_path) as f:
            for line in f:
                try:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype=np.float32)
                    self.dict_index[word] = coefs
                except:
                    pass
        # 初始化停用词
        with open(self.stopwords_path) as f:
            for line in f:
                self.stopwords.append(line.strip())

        for da in range(len(data)):
            sentences = data[da][0]
            # sen_str = "\n".join(sentences)
            res = jieba.lcut(sentences, HMM=True)
            # seg_str = " ".join(res)
            # sen_list = seg_str.split("\n")
            for el in res:
                self.sentences_seg.append(el)

        aa = pd.Series(self.sentences_seg).value_counts()
        self.value_cnt = aa[list(set(aa.index) - set(self.stopwords) - {' '})].sort_values(ascending=False)
        self.w2id = {self.value_cnt.index[k]: k for k in range(1, self.num_words)}  # 词语的索引，从1开始编号

        self.embedding_matrix = np.zeros(shape=(self.num_words, self.m))
        for word, i in self.w2id.items():
            if i < self.num_words:
                # 查预训练的词表
                embedding_vec = self.dict_index.get(word)
                if embedding_vec is not None:
                    self.embedding_matrix[i] = embedding_vec
                else:
                    pass
        return self

    def encode(self, text):
        # 文本转为索引数字模式
        sentences_array = []
        sen = jieba.lcut(text, HMM=True)
        new_sen = [self.w2id.get(word, 0) for word in sen]  # 单词转索引数字
        sentences_array.append(new_sen)
        return np.array(sentences_array)


def text_to_array(w2index, senlist):  # 文本转为索引数字模式
    sentences_array = []
    sen = jieba.lcut(senlist, HMM=True)
    new_sen = [w2index.get(word, 0) for word in sen]   # 单词转索引数字
    sentences_array.append(new_sen)
    return np.array(sentences_array)


class data_generator:
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X, Y = [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                x = tokenizer.encode(text)
                y = d[1]
                x = pad_sequences(x, maxlen=maxlen)[0].tolist()
                X.append(x)
                X_array = np.array(X)
                Y.append(keras.utils.to_categorical(y, num_classes=num_classes).tolist())
                Y_array = np.array(Y)

                if len(X) == self.batch_size or i == idxs[-1]:
                    yield X_array, Y_array
                    [X, Y] = [], []


# 数据预处理
df = pd.read_csv('data.csv')
df['modified_label'] = df['modified_label'].astype(str)
encoder = LabelEncoder()
df['modified_label_'] = encoder.fit_transform(df['modified_label'])
encoder_dict = {}
j = 0
for i in encoder.classes_:
    encoder_dict[i] = j
    j += 1

data = []
for x, y in zip(df['modified_content'], df['modified_label_']):
    data.append((x, y))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)

# 按照8:2的比例划分训练集和验证集
random_order = list(range(len(data)))
np.random.shuffle(random_order)
train_data = [data[j] for i, j in enumerate(random_order) if i % 5 != 0]
valid_data = [data[j] for i, j in enumerate(random_order) if i % 5 == 0]

# 模型参数
num_classes = len(encoder_dict)
maxlen = 50

# 模型结构：词嵌入-卷积池化*3-拼接-全连接-dropout-全连接
inputs = Input(shape=(maxlen, ), name="Input", dtype='float32')
# 词嵌入（使用预训练的词向量）
embedded_sequences = Embedding(input_dim=tokenizer.num_words, output_dim=tokenizer.m,
                               input_length=maxlen, weights=[tokenizer.embedding_matrix],
                               name="embedding")(inputs)
# 词窗大小分别为3,4,5
convs = []
for kernel_size in [3, 4, 5]:
    c = Conv1D(128, kernel_size, padding='same', activation='relu')(embedded_sequences)
    c = GlobalMaxPooling1D()(c)
    convs.append(c)
x = Concatenate()(convs)
drop = Dropout(0.1)(x)
main_output = Dense(num_classes, activation='softmax')(drop)
model = Model(inputs=inputs, outputs=main_output)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()


train_D = data_generator(train_data)
valid_D = data_generator(valid_data)

model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=len(train_D),
    epochs=5,
    validation_data=valid_D.__iter__(),
    validation_steps=len(valid_D)
)
