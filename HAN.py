# -*- coding: utf-8 -*-
# @Time    : 2019-08-09 12:23
# @Author  : finupgroup
# @FileName: HAN.py
# @Software: PyCharm
from .KerasSentiment import *


def han():
    """
    Refer to [Hierarchical Attention Networks for Document Classification]
    (https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf)
    """
    sentence_input = Input(shape=(maxlen,), dtype='int32')
    embedded_sequences = Embedding(input_dim=tokenizer.num_words, output_dim=tokenizer.m,
                                   input_length=maxlen)(sentence_input)
    l_lstm = Bidirectional(LSTM(100, return_sequences=True))(embedded_sequences)
    l_att = Attention()(l_lstm)
    sentEncoder = Model(sentence_input, l_att)
    review_input = Input(shape=(10, maxlen), dtype='int32')  # 10代表有几个句子
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(LSTM(100, return_sequences=True))(review_encoder)
    l_lstm_att = Attention()(l_lstm_sent)
    preds = Dense(1, activation='softmax')(l_lstm_att)
    model = Model(review_input, preds)


model.summary()
