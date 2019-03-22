'''
这个模块包括了工程中所使用的模型的定义
1. Decomposable Attention
2. ESIM网络
3. Siamese网络
4. DSSM网络
'''
import logging

from keras.activations import softmax
from keras.layers import *
from keras.models import Model
from keras.regularizers import L1L2

from constant import *

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


###################################################
# Common Methods
###################################################

def unchanged_shape(input_shape):
    "Function for Lambda layer"
    return input_shape


def time_distributed(input_, layers):
    "Apply a list of layers in TimeDistributed mode"
    out_ = []
    node_ = input_
    for layer_ in layers:
        node_ = TimeDistributed(layer_)(node_)
    out_ = node_
    return out_


def soft_attention_alignment(input_1, input_2):
    "Align text representation with neural soft attention"
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                     output_shape=unchanged_shape)(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2),
                                     output_shape=unchanged_shape)(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned


def apply_multiple(input_, layers):
    "Apply layers to input then concatenate result"
    if not len(layers) > 1:
        raise ValueError('Layers list should contain more than 1 layer')
    else:
        agg_ = []
        for layer in layers:
            agg_.append(layer(input_))
        out_ = Concatenate()(agg_)
    return out_


def substract(input_1, input_2):
    "Substract element-wise"
    neg_input_2 = Lambda(lambda x: -x, output_shape=unchanged_shape)(input_2)
    out_ = Add()([input_1, neg_input_2])
    return out_


def submult(input_1, input_2):
    "Get multiplication and subtraction then concatenate results"
    mult = Multiply()([input_1, input_2])
    sub = substract(input_1, input_2)
    out_ = Concatenate()([sub, mult])
    return out_


###################################################
# Soft Attention Alignment
###################################################

class SoftAttention(Layer):
    '''
    使用Keras的backend的函数,实现soft attention alignment
    '''

    def __init__(self, **kwargs):
        super(SoftAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 检查参数是否合格,参数是list,并且list的长度不能小于2
        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError('A `Concatenate` layer should be called '
                             'on a list of at least 2 inputs')
        # input_shape不能有为None的项
        if all([shape is None for shape in input_shape]):
            return
        # Be sure to call this somewhere!
        super(SoftAttention, self).build(input_shape)

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('A merge layer should be called '
                             'on a list of inputs.')
        q1_encoded = inputs[0]
        q2_encoded = inputs[1]
        attention = K.batch_dot(q1_encoded, q2_encoded, axes=[-1 % K.ndim(q1_encoded), -1 % K.ndim(q2_encoded)])
        w_att_1 = softmax(attention, axis=1)
        w_att_2 = softmax(attention, axis=2)
        w_att_2 = K.permute_dimensions(w_att_2, (0, 2, 1))
        q1_aligned = K.batch_dot(w_att_1, q1_encoded, axes=[1, 1])
        q2_aligned = K.batch_dot(w_att_2, q2_encoded, axes=[1, 1])
        # Compare
        q1_combined = K.concatenate([q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned)])
        q2_combined = K.concatenate([q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned)])

        return [q1_combined, q2_combined]

    def compute_output_shape(self, input_shape):
        shape1 = input_shape[0]
        embedding_length = shape1[1]
        vec_length = shape1[2] * 4

        return [(None, embedding_length, vec_length), (None, embedding_length, vec_length)]


class SoftAttentionAlignment(Layer):
    '''
    使用Keras自带的层,实现soft attention alignment
    '''

    def __init__(self, **kwargs):
        super(SoftAttentionAlignment, self).__init__(**kwargs)

    def build(self, input_shape):
        # 检查参数是否合格,参数是list,并且list的长度不能小于2
        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError('A `Concatenate` layer should be called '
                             'on a list of at least 2 inputs')
        # input_shape不能有为None的项
        if all([shape is None for shape in input_shape]):
            return
        # Be sure to call this somewhere!
        super(SoftAttentionAlignment, self).build(input_shape)

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('A merge layer should be called '
                             'on a list of inputs.')
        input_1 = inputs[0]
        input_2 = inputs[1]
        attention = Dot(axes=-1)([input_1, input_2])
        w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                         output_shape=unchanged_shape)(attention)
        w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2),
                                         output_shape=unchanged_shape)(attention))
        in1_aligned = Dot(axes=1)([w_att_1, input_1])
        in2_aligned = Dot(axes=1)([w_att_2, input_2])

        # Compose
        q1_combined = Concatenate()([input_1, in2_aligned, submult(input_1, in2_aligned)])
        q2_combined = Concatenate()([input_2, in1_aligned, submult(input_2, in1_aligned)])

        return [q1_combined, q2_combined]

    def compute_output_shape(self, input_shape):
        shape1 = input_shape[0]
        embedding_length = shape1[1]
        vec_length = shape1[2] * 4

        return [(None, embedding_length, vec_length), (None, embedding_length, vec_length)]


###################################################
#  siamese 网络中用到的参数共享共享部分
###################################################

class SharedLSTM(Layer):
    '''
    siamese 网络中用到的参数共享共享部分,包括LSTM,卷积,池化部分
    '''

    def __init__(self, n_hidden, **kwargs):
        self.n_hidden = n_hidden
        self.filters = 64
        self.dropout_rate = 0.5
        super(SharedLSTM, self).__init__(**kwargs)

    def build(self, input_shape):
        # 检查参数是否合格
        if input_shape is None:
            raise ValueError('Shape can not be None')
        # Be sure to call this somewhere!
        super(SharedLSTM, self).build(input_shape)

    def call(self, inputs):
        x = Dropout(self.dropout_rate, )(inputs)
        for i, hidden_length in enumerate(self.n_hidden):
            x = Bidirectional(LSTM(hidden_length, return_sequences=True, kernel_regularizer=L1L2(l1=0.01, l2=0.01)))(x)

        # 卷及网络特征层
        connved = Conv1D(64, kernel_size=2, strides=1, padding="valid", kernel_initializer="he_uniform")(x)
        ave_pooling = GlobalAveragePooling1D()(connved)
        max_pooling = GlobalMaxPooling1D()(connved)
        concatenate = Concatenate()([ave_pooling, max_pooling])

        return concatenate

    def compute_output_shape(self, input_shape):
        return (None, self.filters * 2)


###################################################
#  siamese 网络中用到计算曼哈顿距离的层
###################################################

class ManhattanDistance(Layer):
    '''
    计算曼哈顿距离
    '''

    def __init__(self, **kwargs):
        super(ManhattanDistance, self).__init__(**kwargs)

    def build(self, input_shape):
        # 检查参数是否合格,参数是list,并且list的长度不能小于2
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('参数必须是一个长度为2的list')
        # input_shape不能有为None的项
        if all([shape is None for shape in input_shape]):
            return
        # Be sure to call this somewhere!
        super(ManhattanDistance, self).build(input_shape)

    def call(self, inputs):
        left_output = inputs[0]
        right_output = inputs[1]
        distance = K.exp(-K.sum(K.abs(left_output - right_output), axis=1, keepdims=True))

        return distance

    def compute_output_shape(self, input_shape):
        output_shape = (None, 1)
        return output_shape


class DenseBatchnormDropout(Layer):
    '''
    Dense-BatchNorm-Dropout
    '''

    def __init__(self, dense_dim, dense_dropout, activation, **kwargs):
        self.dense_dim = dense_dim
        self.dense_dropout = dense_dropout
        self.activation = activation
        super(DenseBatchnormDropout, self).__init__(**kwargs)

    def build(self, input_shape):
        super(DenseBatchnormDropout, self).build(input_shape)

    def call(self, inputs):
        dense = Dense(self.dense_dim, activation=self.activation)(inputs)
        dense = BatchNormalization()(dense)
        dense = Dropout(self.dense_dropout)(dense)

        return dense

    def compute_output_shape(self, input_shape):
        output_shape = (None, self.dense_dim)
        return output_shape


class AveMaxPooling(Layer):
    '''
    计算平均池化和最大池化,并将结果拼接起来
    '''

    def __init__(self, **kwargs):
        super(AveMaxPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AveMaxPooling, self).build(input_shape)

    def call(self, inputs):
        ave_pooling = GlobalAvgPool1D()(inputs)
        max_pooling = GlobalMaxPool1D()(inputs)
        concat = Concatenate()([ave_pooling, max_pooling])

        return concat

    def compute_output_shape(self, input_shape):
        output_shape = (None, input_shape[2] * 2)
        return output_shape


###################################################
# siamese
###################################################

def siamese(pretrained_embedding=None,
            input_length=MAX_LEN,
            w2v_length=300,
            n_hidden=[64, 64, 64]):
    # 输入层
    left_input = Input(shape=(input_length,), dtype='int32')
    right_input = Input(shape=(input_length,), dtype='int32')

    # 对句子embedding
    encoded_left = pretrained_embedding(left_input)
    encoded_right = pretrained_embedding(right_input)

    # 使用自定义的LSTM(包括LSTM,卷积,池化)
    shared_lstm = SharedLSTM(n_hidden)
    # 共享自定义LSTM的参数
    left_output = shared_lstm(encoded_left)
    right_output = shared_lstm(encoded_right)

    # 计算曼哈顿距离
    malstm_distance = ManhattanDistance()([left_output, right_output])

    model = Model([left_input, right_input], [malstm_distance])

    return model


###################################################
# Model ESIM
###################################################

def esim(pretrained_embedding,
         maxlen=MAX_LEN,
         lstm_dim=300,
         dense_dim=300,
         dense_dropout=0.5):
    '''
    # Based on arXiv:1609.06038
    :param pretrained_embedding:
    :param maxlen:
    :param lstm_dim:
    :param dense_dim:
    :param dense_dropout:
    :return:
    '''

    q1 = Input(name='q1', shape=(maxlen,))
    q2 = Input(name='q2', shape=(maxlen,))

    # Embedding
    # embedding = create_pretrained_embedding(pretrained_embedding, mask_zero=False)
    embedding = pretrained_embedding
    bn = BatchNormalization(axis=2)
    q1_embed = bn(embedding(q1))
    q2_embed = bn(embedding(q2))

    # Encode
    encode = Bidirectional(LSTM(lstm_dim, return_sequences=True))
    q1_encoded = encode(q1_embed)
    q2_encoded = encode(q2_embed)

    # Attention
    # q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)

    # Compose
    # q1_combined = Concatenate()([q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned)])
    # q2_combined = Concatenate()([q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned)])
    q1_combined, q2_combined = SoftAttentionAlignment()([q1_encoded, q2_encoded])
    compose = Bidirectional(LSTM(lstm_dim, return_sequences=True))
    q1_compare = compose(q1_combined)
    q2_compare = compose(q2_combined)

    # Aggregate
    q1_rep = AveMaxPooling()(q1_compare)
    q2_rep = AveMaxPooling()(q2_compare)

    # Classifier
    merged = Concatenate()([q1_rep, q2_rep])

    dense = BatchNormalization()(merged)
    dense = DenseBatchnormDropout(dense_dim, dense_dropout, 'relu')(dense)
    dense = DenseBatchnormDropout(dense_dim, dense_dropout, 'relu')(dense)
    out_ = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[q1, q2], outputs=out_)
    return model


###################################################
# Model decomposable attention
###################################################

def decomposable_attention(pretrained_embedding,
                           projection_dim=300, projection_hidden=0, projection_dropout=0.2,
                           compare_dim=500, compare_dropout=0.2,
                           dense_dim=300, dense_dropout=0.2,
                           lr=1e-3, activation='elu', maxlen=MAX_LEN):
    '''Based on: https://arxiv.org/abs/1606.01933
    关于 Decomposable attention的中文简单解释查看博客: https://blog.csdn.net/u013398398/article/details/81024021
    :param pretrained_embedding:
    :param projection_dim:
    :param projection_hidden:
    :param projection_dropout:
    :param compare_dim:
    :param compare_dropout:
    :param dense_dim:
    :param dense_dropout:
    :param lr:
    :param activation:
    :param maxlen:
    :return:
    '''

    q1 = Input(name='q1', shape=(maxlen,))
    q2 = Input(name='q2', shape=(maxlen,))

    # Embedding
    # embedding = create_pretrained_embedding(pretrained_embedding,
    #                                         mask_zero=False)
    embedding = pretrained_embedding
    q1_embed = embedding(q1)
    q2_embed = embedding(q2)

    # Projection
    projection_layers = []
    if projection_hidden > 0:
        projection_layers.extend([
            Dense(projection_hidden, activation=activation),
            Dropout(rate=projection_dropout),
        ])
    projection_layers.extend([
        Dense(projection_dim, activation=None),
        Dropout(rate=projection_dropout),
    ])
    q1_encoded = time_distributed(q1_embed, projection_layers)
    q2_encoded = time_distributed(q2_embed, projection_layers)

    # Attention
    # q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)

    # Compare
    # q1_combined = Concatenate()([q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned)])
    # q2_combined = Concatenate()([q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned)])
    q1_combined, q2_combined = SoftAttentionAlignment()([q1_encoded, q2_encoded])
    compare_layers = [
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
    ]
    q1_compare = time_distributed(q1_combined, compare_layers)
    q2_compare = time_distributed(q2_combined, compare_layers)

    # Aggregate
    q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])

    # Classifier
    merged = Concatenate()([q1_rep, q2_rep])
    dense = BatchNormalization()(merged)
    dense = Dense(dense_dim, activation=activation)(dense)
    dense = Dropout(dense_dropout)(dense)
    dense = BatchNormalization()(dense)
    dense = Dense(dense_dim, activation=activation)(dense)
    dense = Dropout(dense_dropout)(dense)
    out_ = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[q1, q2], outputs=out_)
    return model
