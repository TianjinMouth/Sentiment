# -*- coding: utf-8 -*-
# @Time    : 2019-08-05 11:29
# @Author  : ZHang
# @FileName: Entropy-based Semi-supervised.py
# @Software: PyCharm

import pandas as pd
import tensorflow as tf
from tensorflow import keras
import jieba
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences


def entropy_based_semi_supervised_loss(y_true, y_pred, lam=0.2, axis=-1):
    """My Categorical cross entropy between an output tensor and a target tensor.

    # Arguments
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
        lam:
        axis: Int specifying the channels axis. `axis=-1`
            corresponds to data format `channels_last`,
            and `axis=1` corresponds to data format
            `channels_first`.

    # Returns
        Output tensor.

    # Raises
        ValueError: if `axis` is neither -1 nor one of
            the axes of `output`.
    """
    output_dimensions = list(range(len(y_pred.get_shape())))
    if axis != -1 and axis not in output_dimensions:
        raise ValueError(
            '{}{}{}'.format(
                'Unexpected channels axis {}. '.format(axis),
                'Expected to be -1 or one of the axes of `output`, ',
                'which has {} dimensions.'.format(len(y_pred.get_shape()))))

    # scale preds so that the class probas of each sample sum to 1
    y_pred /= tf.reduce_sum(y_pred, axis, True)
    # manual computation of cross entropy
    _epsilon = tf.convert_to_tensor(1e-7, y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
    return - tf.reduce_sum((tf.reshape(tf.reduce_sum(y_true, axis), (-1, 1))) * (y_true * tf.log(y_pred+1e-8)), axis)\
           - tf.reduce_sum(lam * (1-tf.reshape(tf.reduce_sum(y_true, axis), (-1, 1))) * (y_pred * tf.log(y_pred+1e-8)), axis)


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


class data_generator:
    def __init__(self, data, batch_size=64):
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
                if np.isnan(y):
                    Y.append(np.array([0.] * num_classes))
                else:
                    Y.append(keras.utils.to_categorical(y, num_classes=num_classes).tolist())
                Y_array = np.array(Y)

                if len(X) == self.batch_size or i == idxs[-1]:
                    yield X_array, Y_array
                    [X, Y] = [], []


# 数据预处理
df = pd.read_csv('data_semi.csv', error_bad_lines=False)
df['modified_label'] = df['modified_label'].astype(str)
encoder_dict = {'不理解': 0, '不还': 1, '不需回应': 2, '介绍': 3, '信息丢失': 4, '司法途径': 5, '周边联系': 6, '在吗': 7, '多扣款': 8,
                '多方催收': 9, '客户操作问题': 10, '已知悉': 11, '已还款': 12, '征信': 13, '待人工处理': 14, '微信联系': 15, '投诉': 16, '抱怨': 17,
                '新联系': 18, '无关': 19, '有还款意向': 20, '有还款意向及时间': 21, '没钱': 22, '甲方操作问题': 23, '要求减免': 24, '诉苦': 25,
                '询问借款信息': 26, '询问还款方式': 27, '费用计算': 28}
df['modified_label_'] = df['modified_label'].apply(lambda x: encoder_dict[x] if x != 'nan' else None)

data = []
for x, y in zip(df['modified_content'], df['modified_label_']):
    data.append((x, y))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)

# 按照8:2的比例划分训练集和验证集
random_order = list(range(len(data)))
np.random.shuffle(random_order)
train_data = [data[j] for i, j in enumerate(random_order) if i % 5 != 0]
valid_data = [data[j] for i, j in enumerate(random_order) if (i % 5 == 0) and (not np.isnan(data[j][1]))]

# 模型参数
num_classes = len(encoder_dict)
maxlen = 50

inputs = Input(shape=(maxlen,), name="Input", dtype='float32')

embedded_sequences = Embedding(input_dim=tokenizer.num_words, output_dim=tokenizer.m,
                               input_length=maxlen, weights=[tokenizer.embedding_matrix],
                               name="embedding")(inputs)

x_context = Bidirectional(LSTM(128, return_sequences=True))(embedded_sequences)
x = Concatenate()([embedded_sequences, x_context])

convs = []
for kernel_size in range(1, 5):
    conv = Conv1D(128, kernel_size, activation='relu')(x)
    convs.append(conv)
poolings = [GlobalAveragePooling1D()(conv) for conv in convs] + [GlobalMaxPooling1D()(conv) for conv in convs]
x = Concatenate()(poolings)

output = Dense(num_classes, activation='softmax', name='softmax')(x)
model = Model(inputs=inputs, outputs=output)
model.compile(loss=entropy_based_semi_supervised_loss,
              optimizer='Adam',
              metrics=['categorical_accuracy'])
model.summary()


train_D = data_generator(train_data)
valid_D = data_generator(valid_data)

model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=len(train_D),
    epochs=10,
    validation_data=valid_D.__iter__(),
    validation_steps=len(valid_D)
)
