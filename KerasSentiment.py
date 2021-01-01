# -*- coding: utf-8 -*-
# @Time    : 2019-07-19 18:22
# @Author  : finupgroup
# @FileName: KerasSentiment.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import jieba
import math

import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.callbacks import TensorBoard
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K


def lr_schedule(epoch):
    # 根据epoch返回不同的学习率
    """LearningRateScheduler(lr_schedule)"""
    if epoch < 50:
        lr = 1e-2
    elif epoch < 80:
        lr = 1e-3
    else:
        lr = 1e-4
    return lr


def gelu(x):
    """
    An approximation of gelu.
    See: https://arxiv.org/pdf/1606.08415.pdf
    """
    return 0.5 * x * (
      1.0 + K.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * K.pow(x, 3))))


def my_binary_loss(y_true, y_pred):
    margin = 0.6
    theta = lambda t: (K.sign(t) + 1.) / 2.
    return - (1 - theta(y_true - margin) * theta(y_pred - margin)
                - theta(1 - margin - y_true) * theta(1 - margin - y_pred)
             ) * (y_true * K.log(y_pred + 1e-8) + (1 - y_true) * K.log(1 - y_pred + 1e-8))


def my_categorical_crossentropy(y_true, y_pred, margin=0.6, from_logits=False, axis=-1):
    """My Categorical crossentropy between an output tensor and a target tensor.

    # Arguments
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
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
    theta = lambda t: (tf.sign(t) + 1.) / 2.
    if axis != -1 and axis not in output_dimensions:
        raise ValueError(
            '{}{}{}'.format(
                'Unexpected channels axis {}. '.format(axis),
                'Expected to be -1 or one of the axes of `output`, ',
                'which has {} dimensions.'.format(len(y_pred.get_shape()))))
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        y_pred /= tf.reduce_sum(y_pred, axis, True)
        # manual computation of crossentropy
        _epsilon = tf.convert_to_tensor(1e-7, y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
        return - tf.reduce_sum((1 - theta(y_true - margin) * theta(y_pred - margin)
             - theta(1 - margin - y_true) * theta(1 - margin - y_pred)
          ) * (y_true * K.log(y_pred + 1e-8)), axis)
    else:
        return tf.nn.softmax_cross_entropy_with_logits(labels=y_true,
                                                       logits=y_pred)


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss

    return binary_focal_loss_fixed


class Evaluate(Callback):
    """
    Warmup
    """
    def __init__(self):
        self.num_passed_batchs = 0
        self.warmup_epochs = 10

    def on_batch_begin(self, batch, logs=None):
        # params是模型自动传递给Callback的一些参数
        if self.params['steps'] == None:
            self.steps_per_epoch = np.ceil(1. * self.params['samples'] / self.params['batch_size'])
        else:
            self.steps_per_epoch = self.params['steps']
        if self.num_passed_batchs < self.steps_per_epoch * self.warmup_epochs:
            # 前10个epoch中，学习率线性地从零增加到0.001
            K.set_value(self.model.optimizer.lr,
                        0.001 * (self.num_passed_batchs + 1) / self.steps_per_epoch / self.warmup_epochs)
            self.num_passed_batchs += 1


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

        self.W_regularizer = keras.regularizers.get(W_regularizer)
        self.b_regularizer = keras.regularizers.get(b_regularizer)

        self.W_constraint = keras.constraints.get(W_constraint)
        self.b_constraint = keras.constraints.get(b_constraint)

        self.bias = bias

        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.step_dim = input_shape[1]
        assert len(input_shape) == 3  # batch, timestep, num_features
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
        # print(K.reshape(x, (-1, features_dim)))  # n, d
        # print(K.reshape(self.W, (features_dim, 1)))  # w= dx1
        # print(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))))  # nx1

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))  # batch, step
        # print(eij)
        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        # print(a)
        a = K.expand_dims(a)
        # print("expand_dims:")
        # print(a)
        # print("x:")
        # print(x)
        weighted_input = x * a
        # print(weighted_input.shape)
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return input_shape[0], self.features_dim


class Lookahead(object):
    """
    Add the [Lookahead Optimizer](https://arxiv.org/abs/1907.08610) functionality for [keras](https://keras.io/).
    model.compile(optimizer=Adam(1e-3), loss='mse') # 用你想用的优化器
    lookahead = Lookahead(k=5, alpha=0.5) # 初始化Lookahead
    lookahead.inject(model) # 插入到模型中
    """

    def __init__(self, k=5, alpha=0.5):
        self.k = k
        self.alpha = alpha
        self.count = 0

    def inject(self, model):
        """Inject the Lookahead algorithm for the given model.
        The following code is modified from keras's _make_train_function method.
        See: https://github.com/keras-team/keras/blob/master/keras/engine/training.py#L497
        """
        if not hasattr(model, 'train_function'):
            raise RuntimeError('You must compile your model before using it.')

        model._check_trainable_weights_consistency()

        if model.train_function is None:
            inputs = (model._feed_inputs +
                      model._feed_targets +
                      model._feed_sample_weights)
            if model._uses_dynamic_learning_phase():
                inputs += [K.learning_phase()]
            fast_params = model._collected_trainable_weights

            with K.name_scope('training'):
                with K.name_scope(model.optimizer.__class__.__name__):
                    training_updates = model.optimizer.get_updates(
                        params=fast_params,
                        loss=model.total_loss)
                    slow_params = [K.variable(p) for p in fast_params]
                fast_updates = (model.updates +
                                training_updates +
                                model.metrics_updates)

                slow_updates, copy_updates = [], []
                for p, q in zip(fast_params, slow_params):
                    slow_updates.append(K.update(q, q + self.alpha * (p - q)))
                    copy_updates.append(K.update(p, q))

                # Gets loss and metrics. Updates weights at each call.
                fast_train_function = K.function(
                    inputs,
                    [model.total_loss] + model.metrics_tensors,
                    updates=fast_updates,
                    name='fast_train_function',
                    **model._function_kwargs)

                def F(inputs):
                    self.count += 1
                    R = fast_train_function(inputs)
                    if self.count % self.k == 0:
                        K.batch_get_value(slow_updates)
                        K.batch_get_value(copy_updates)
                    return R

                model.train_function = F


class LazyOptimizer(Optimizer):
    """Inheriting Optimizer class, wrapping the original optimizer
    to achieve a new corresponding lazy optimizer.
    (Not only LazyAdam, but also LazySGD with momentum if you like.)
    # Arguments
        optimizer: an instance of keras optimizer (supporting
                    all keras optimizers currently available);
        embedding_layers: all Embedding layers you want to update sparsely.
    # Returns
        a new keras optimizer.
    继承Optimizer类，包装原有优化器，实现Lazy版优化器
    （不局限于LazyAdam，任何带动量的优化器都可以有对应的Lazy版）。
    # 参数
        optimizer：优化器实例，支持目前所有的keras优化器；
        embedding_layers：模型中所有你喜欢稀疏更新的Embedding层。
    # 返回
        一个新的keras优化器
    """
    def __init__(self, optimizer, embedding_layers=None, **kwargs):
        super(LazyOptimizer, self).__init__(**kwargs)
        self.optimizer = optimizer
        self.embeddings = []
        if embedding_layers is not None:
            for l in embedding_layers:
                self.embeddings.append(
                    model.get_layer(name=l).trainable_weights[0]
                )
        with K.name_scope(self.__class__.__name__):
            for attr in self.optimizer.get_config():
                if not hasattr(self, attr):
                    value = getattr(self.optimizer, attr)
                    setattr(self, attr, value)
        self.optimizer.get_gradients = self.get_gradients
        self._cache_grads = {}

    def get_gradients(self, loss, params):
        """Cache the gradients to avoiding recalculating.
        把梯度缓存起来，避免重复计算，提高效率。
        """
        _params = []
        for p in params:
            if (loss, p) not in self._cache_grads:
                _params.append(p)
        _grads = super(LazyOptimizer, self).get_gradients(loss, _params)
        for p, g in zip(_params, _grads):
            self._cache_grads[(loss, p)] = g
        return [self._cache_grads[(loss, p)] for p in params]

    def get_updates(self, loss, params):
        # Only for initialization (仅初始化)
        self.optimizer.get_updates(loss, params)
        # Common updates (常规更新)
        dense_params = [p for p in params if p not in self.embeddings]
        self.updates = self.optimizer.get_updates(loss, dense_params)
        # Sparse update (稀疏更新)
        sparse_params = self.embeddings
        sparse_grads = self.get_gradients(loss, sparse_params)
        sparse_flags = [
            K.all(K.not_equal(g, 0), axis=-1, keepdims=True)
            for g in sparse_grads
        ]
        original_lr = self.optimizer.lr
        for f, p in zip(sparse_flags, sparse_params):
            self.optimizer.lr = original_lr * K.cast(f, 'float32')
            # updates only when gradients are not equal to zeros.
            # (gradients are equal to zeros means these words are not sampled very likely.)
            # 仅更新梯度不为0的Embedding（梯度为0意味着这些词很可能是没被采样到的）
            self.updates.extend(
                self.optimizer.get_updates(loss, [p])
            )
        self.optimizer.lr = original_lr
        return self.updates

    def get_config(self):
        config = self.optimizer.get_config()
        return config


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
            _X, _Y = [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                x = tokenizer.encode(text)
                y = d[1]
                x = pad_sequences(x, maxlen=maxlen)[0].tolist()
                _X.append(x)
                _X_array = np.array(_X)
                _Y.append(keras.utils.to_categorical(y, num_classes=num_classes).tolist())
                _Y_array = np.array(_Y)

                if len(_X) == self.batch_size or i == idxs[-1]:
                    yield _X_array, _Y_array
                    [_X, _Y] = [], []


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
model_type = "RCNNVariant"
if model_type == "Bi-LSTM-Attention":
    inputs = Input(shape=(maxlen, ), name="Input", dtype='float32')
    embedded_sequences = Embedding(input_dim=tokenizer.num_words, output_dim=tokenizer.m,
                                   input_length=maxlen,
                                   # weights=[tokenizer.embedding_matrix],
                                   name="embedding")(inputs)
    l_lstm = Bidirectional(LSTM(128, return_sequences=True, name="Bi-LSTM"))(embedded_sequences)
    l_att = Attention(name="Attention")(l_lstm)
    output = Dense(num_classes, activation='softmax', name="softmax")(l_att)
    model = Model(inputs, output)


elif model_type == "TextCNN":
    # 模型结构：词嵌入-卷积池化*3-拼接-全连接-dropout-全连接
    inputs = Input(shape=(maxlen,), name="Input", dtype='float32')
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

elif model_type == "FastText":
    inputs = Input(shape=(maxlen,), name="Input", dtype='float32')

    embedded_sequences = Embedding(input_dim=tokenizer.num_words, output_dim=tokenizer.m,
                                   input_length=maxlen, weights=[tokenizer.embedding_matrix],
                                   name="embedding")(inputs)
    x = GlobalAveragePooling1D()(embedded_sequences)

    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=output)

elif model_type == "RCNN":
    input_current = Input(shape=(maxlen,), name="Input_current", dtype='float32')
    input_left = Input(shape=(maxlen,), name="Input_left", dtype='float32')
    input_right = Input(shape=(maxlen,), name="Input_right", dtype='float32')

    embedded_sequences = Embedding(input_dim=tokenizer.num_words, output_dim=tokenizer.m,
                                   input_length=maxlen, weights=[tokenizer.embedding_matrix],
                                   name="embedded_sequences")(input_current)

    embedded_left = Embedding(input_dim=tokenizer.num_words, output_dim=tokenizer.m,
                              input_length=maxlen, weights=[tokenizer.embedding_matrix],
                              name="embedded_left")(input_left)

    embedded_right = Embedding(input_dim=tokenizer.num_words, output_dim=tokenizer.m,
                               input_length=maxlen, weights=[tokenizer.embedding_matrix],
                               name="embedded_right")(input_right)

    x_left = SimpleRNN(128, return_sequences=True)(embedded_left)
    x_right = SimpleRNN(128, return_sequences=True, go_backwards=True)(embedded_right)
    x_right = Lambda(lambda x: K.reverse(x, axes=1))(x_right)
    x = Concatenate(axis=2)([x_left, embedded_sequences, x_right])

    x = Conv1D(64, kernel_size=1, activation='tanh')(x)
    x = GlobalMaxPooling1D()(x)

    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=[input_current, input_left, input_right], outputs=output)

elif model_type == "RCNNVariant":
    """Variant of RCNN.
            Base on structure of RCNN, we do some improvement:
            1. Ignore the shift for left/right context.
            2. Use Bidirectional LSTM/GRU to encode context.
            3. Use Multi-CNN to represent the semantic vectors.
            4. Use ReLU instead of Tanh.
            5. Use both AveragePooling and MaxPooling.
    """
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

    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=output)

model.compile(loss=my_categorical_crossentropy,
              # optimizer=LazyOptimizer(keras.optimizers.Adam(), ["embedding"]),
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
model.summary()


train_D = data_generator(train_data)
valid_D = data_generator(valid_data)

model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=len(train_D),
    epochs=5,
    validation_data=valid_D.__iter__(),
    validation_steps=len(valid_D),
    callbacks=[keras.callbacks.ModelCheckpoint("weights-{epoch:04d}--{val_acc:.4f}.h5", monitor='val_acc',
                                               save_best_only=True, verbose=1),
               TensorBoard(log_dir='./logs', write_graph=True, write_images=True)]
)
