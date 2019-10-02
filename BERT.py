# -*- coding: utf-8 -*-
# @Time    : 2019-07-19 16:43
# @Author  : finupgroup
# @FileName: BERT.py
# @Software: PyCharm

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import TensorBoard
from keras_bert import load_trained_model_from_checkpoint, Tokenizer, get_custom_objects
import codecs
import keras
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
# tensorboard --logdir=logs


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


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
            _X1, _X2, _Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                _X1.append(x1)
                _X2.append(x2)
                _Y.append(keras.utils.to_categorical(y, num_classes=num_classes))
                if len(_X1) == self.batch_size or i == idxs[-1]:
                    _X1 = seq_padding(_X1)
                    _X2 = seq_padding(_X2)
                    _Y = seq_padding(_Y)
                    yield [_X1, _X2], _Y
                    [_X1, _X2, _Y] = [], [], []


# 数据预处理
df = pd.read_csv('data.csv')
df = df[~df['modified_label'].isnull()]
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

# 按照8:2的比例划分训练集和验证集
random_order = list(range(len(data)))
np.random.shuffle(random_order)
train_data = [data[j] for i, j in enumerate(random_order) if i % 5 != 0]
valid_data = [data[j] for i, j in enumerate(random_order) if i % 5 == 0]

# 模型参数
num_classes = len(encoder_dict)
maxlen = 100
config_path = '/Users/finupgroup/Desktop/FinupCredit/资易通-云电猫/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/Users/finupgroup/Desktop/FinupCredit/资易通-云电猫/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/Users/finupgroup/Desktop/FinupCredit/资易通-云电猫/chinese_L-12_H-768_A-12/vocab.txt'
token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
tokenizer = OurTokenizer(token_dict)

bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

for l in bert_model.layers:
    l.trainable = True

x1_in = Input(shape=(None,))
x2_in = Input(shape=(None,))

x = bert_model([x1_in, x2_in])
x = Lambda(lambda x: x[:, 0])(x)
p = Dense(num_classes, activation='softmax', name='softmax')(x)

model = Model([x1_in, x2_in], p)
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(1e-5),  # 用足够小的学习率
    metrics=['categorical_accuracy']
)
model.summary()

train_D = data_generator(train_data)
valid_D = data_generator(valid_data)

model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=len(train_D),
    epochs=5,
    validation_data=valid_D.__iter__(),
    validation_steps=len(valid_D),
    callbacks=[keras.callbacks.ModelCheckpoint("weights-{epoch:04d}--{val_categorical_accuracy:.4f}.h5",
                                               monitor='val_categorical_accuracy',
                                               save_best_only=True, verbose=1),
               TensorBoard(log_dir='./logs', write_graph=True, write_images=True)]
)

# Test
model = keras.models.load_model('weights-0002--0.8593.h5', custom_objects=get_custom_objects())
test = '马上就还了，这会钱还没到位，马上了'
test = test[:100]
x1, x2 = tokenizer.encode(first=test)


oot = pd.read_csv('oot5-8.csv')
oot = oot[~oot['content'].isnull()]
oot['label'] = oot['label'].astype(str)
oot['modified_label_'] = encoder.fit_transform(oot['label'])


oot_data = []
for x, y in zip(oot['content'], oot['modified_label_']):
    oot_data.append((x, y))


def ft_api_parse_result_new(encoder_dict, result):
    return list(encoder_dict.keys())[result.argmax()], result[0].max()


yuliao, true, predict_label, predict_prob = [], [], [], []
for i in range(len(oot_data)):
    test = oot_data[i][0]
    test = test[:100]
    x1, x2 = tokenizer.encode(first=test)
    label, prob = ft_api_parse_result_new(encoder_dict, model.predict([[x1], [x2]]))
    yuliao.append(test)
    true.append(oot_data[i][1])
    predict_label.append(label)
    predict_prob.append(prob)

result = pd.DataFrame({'yuliao': yuliao, 'true': true, 'predict_label': predict_label, 'predict_prob': predict_prob})


def get_true(ele):
    return list(encoder_dict)[ele]


def get_acc(ele1, ele2):
    if ele1 == ele2:
        return 1
    else:
        return 0


result['true_label'] = result['true'].apply(lambda x: get_true(x))
result['acc'] = result.apply(lambda row: get_acc(row['predict_label'], row['true_label']), axis=1)
result.groupby(['predict_label']).agg({'predict_label': 'count', 'acc': 'sum'})
