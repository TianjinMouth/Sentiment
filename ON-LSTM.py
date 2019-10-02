# -*- coding: utf-8 -*-
# @Time    : 2019-07-30 15:51
# @Author  : finupgroup
# @FileName: ON-LSTM.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K


def cumsoftmax(x, mode='l2r'):
    """先softmax，然后cumsum，
    cumsum区分从左到右、从右到左两种模式
    """
    axis = K.ndim(x) - 1
    if mode == 'l2r':
        x = K.softmax(x, axis=axis)
        x = K.cumsum(x, axis=axis)
        return x
    elif mode == 'r2l':
        x = x[..., ::-1]
        x = K.softmax(x, axis=axis)
        x = K.cumsum(x, axis=axis)
        return x[..., ::-1]
    else:
        return x


class ONLSTM(Layer):
    """
    实现有序LSTM，来自论文
    Ordered Neurons: Integrating Tree Structures into Recurrent Neural Networks
    """
    def __init__(self,
                 units,
                 levels,
                 return_sequences=False,
                 dropconnect=None,
                 **kwargs):
        assert units % levels == 0
        self.units = units
        self.levels = levels
        self.chunk_size = units // levels
        self.return_sequences = return_sequences
        self.dropconnect = dropconnect
        super(ONLSTM, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4 + self.levels * 2),
            name='kernel',
            initializer='glorot_uniform')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4 + self.levels * 2),
            name='recurrent_kernel',
            initializer='orthogonal')
        self.bias = self.add_weight(
            shape=(self.units * 4 + self.levels * 2,),
            name='bias',
            initializer='zeros')
        self.built = True
        if self.dropconnect:
            self._kernel = K.dropout(self.kernel, self.dropconnect)
            self._kernel = K.in_train_phase(self._kernel, self.kernel)
            self._recurrent_kernel = K.dropout(self.recurrent_kernel, self.dropconnect)
            self._recurrent_kernel = K.in_train_phase(self._recurrent_kernel, self.recurrent_kernel)
        else:
            self._kernel = self.kernel
            self._recurrent_kernel = self.recurrent_kernel

    def one_step(self, inputs, states):
        x_in, (c_last, h_last) = inputs, states
        print('x_in:    ' + str(x_in))
        print('c_last:    ' + str(c_last))
        print('h_last:    ' + str(h_last))
        x_out = K.dot(x_in, self._kernel) + K.dot(h_last, self._recurrent_kernel)
        print('K.dot(x_in, self._kernel):    ' + str(K.dot(x_in, self._kernel)))
        print('K.dot(h_last, self._recurrent_kernel):    ' + str(K.dot(h_last, self._recurrent_kernel)))
        x_out = K.bias_add(x_out, self.bias)
        print('x_out:    ' + str(x_out))
        f_master_gate = cumsoftmax(x_out[:, :self.levels], 'l2r')
        print('x_out[:, :self.levels]:    ' + str(x_out[:, :self.levels]))
        print('f_master_gate:    ' + str(f_master_gate))
        f_master_gate = K.expand_dims(f_master_gate, 2)
        print('f_master_gate:    ' + str(f_master_gate))
        i_master_gate = cumsoftmax(x_out[:, self.levels: self.levels * 2], 'r2l')
        print('x_out[:, self.levels: self.levels * 2]:    ' + str(x_out[:, self.levels: self.levels * 2]))
        print('i_master_gate:    ' + str(i_master_gate))
        i_master_gate = K.expand_dims(i_master_gate, 2)
        print('i_master_gate:    ' + str(i_master_gate))
        x_out = x_out[:, self.levels * 2:]
        print('x_out:    ' + str(x_out))
        x_out = K.reshape(x_out, (-1, self.levels * 4, self.chunk_size))
        print('x_out:    ' + str(x_out))
        f_gate = K.sigmoid(x_out[:, :self.levels])
        print('x_out[:, :self.levels]    ' + str(x_out[:, :self.levels]))
        print(f_gate)
        i_gate = K.sigmoid(x_out[:, self.levels: self.levels * 2])
        print('x_out[:, self.levels: self.levels * 2]    ' + str(x_out[:, self.levels: self.levels * 2]))
        print(i_gate)
        o_gate = K.sigmoid(x_out[:, self.levels * 2: self.levels * 3])
        print('x_out[:, self.levels * 2: self.levels * 3]    ' + str(x_out[:, self.levels * 2: self.levels * 3]))
        print(o_gate)
        c_in = K.tanh(x_out[:, self.levels * 3:])
        c_last = K.reshape(c_last, (-1, self.levels, self.chunk_size))
        overlap = f_master_gate * i_master_gate
        print('overlap:    ' + str(overlap))
        c_out = overlap * (f_gate * c_last + i_gate * c_in) + \
                (f_master_gate - overlap) * c_last + \
                (i_master_gate - overlap) * c_in
        print('c_out:    ' + str(c_out))
        h_out = o_gate * K.tanh(c_out)
        print('h_out:    ' + str(h_out))
        c_out = K.reshape(c_out, (-1, self.units))
        print('c_out:    ' + str(c_out))
        h_out = K.reshape(h_out, (-1, self.units))
        print('h_out:    ' + str(h_out))
        out = K.concatenate([h_out, f_master_gate[..., 0], i_master_gate[..., 0]], 1)
        return out, [c_out, h_out]

    def call(self, inputs):
        initial_states = [
            K.zeros((K.shape(inputs)[0], self.units)),
            K.zeros((K.shape(inputs)[0], self.units))
        ]  # 定义初始态(全零)
        outputs = K.rnn(self.one_step, inputs, initial_states)
        self.distance = 1 - K.mean(outputs[1][..., self.units: self.units + self.levels], -1)
        self.distance_in = K.mean(outputs[1][..., self.units + self.levels:], -1)
        if self.return_sequences:
            return outputs[1][..., :self.units]
        else:
            return outputs[0][..., :self.units]

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.units)
        else:
            return (input_shape[0], self.units)
