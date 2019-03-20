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
# decomposable attention
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
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)

    # Compare
    q1_combined = Concatenate()([q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned)])
    q2_combined = Concatenate()([q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned)])
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


###################################################
# ESIM
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
    q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])

    # Classifier
    merged = Concatenate()([q1_rep, q2_rep])

    dense = BatchNormalization()(merged)
    dense = Dense(dense_dim, activation='elu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(dense_dropout)(dense)
    dense = Dense(dense_dim, activation='elu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(dense_dropout)(dense)
    out_ = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[q1, q2], outputs=out_)
    return model


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

    # 两个LSTM共享参数
    # # v1 一层lstm
    # shared_lstm = LSTM(n_hidden)

    # # v2 带drop和正则化的多层lstm
    ipt = Input(shape=(input_length, w2v_length))
    dropout_rate = 0.5
    x = Dropout(dropout_rate, )(ipt)
    for i, hidden_length in enumerate(n_hidden):
        # x = Bidirectional(LSTM(hidden_length, return_sequences=(i!=len(n_hidden)-1), kernel_regularizer=L1L2(l1=0.01, l2=0.01)))(x)
        x = Bidirectional(LSTM(hidden_length, return_sequences=True, kernel_regularizer=L1L2(l1=0.01, l2=0.01)))(x)

    # v3 卷及网络特征层
    x = Conv1D(64, kernel_size=2, strides=1, padding="valid", kernel_initializer="he_uniform")(x)
    x_p1 = GlobalAveragePooling1D()(x)
    x_p2 = GlobalMaxPooling1D()(x)
    x = Concatenate()([x_p1, x_p2])
    shared_lstm = Model(inputs=ipt, outputs=x)

    left_output = shared_lstm(encoded_left)
    right_output = shared_lstm(encoded_right)

    # 距离函数 exponent_neg_manhattan_distance
    malstm_distance = Lambda(lambda x: K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True)),
                             output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

    model = Model([left_input, right_input], [malstm_distance])

    return model


###################################################
# Attention Layer
###################################################

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        # self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='%s_W' % self.name,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='%s_b' % self.name,
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)
        a = K.exp(eij)
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        # print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return input_shape[0], self.features_dim


###################################################
# DSSM
###################################################

def DSSM(pretrained_embedding, input_length, lstmsize=90):
    '''
    Deep Structured Semantic Models
    :param pretrained_embedding:
    :param input_length:
    :param lstmsize:
    :return:
    '''
    word_embedding, char_embedding = pretrained_embedding
    wordlen, charlen = input_length

    input1 = Input(shape=(wordlen,))
    input2 = Input(shape=(wordlen,))
    lstm0 = LSTM(lstmsize, return_sequences=True)
    lstm1 = Bidirectional(LSTM(lstmsize))
    lstm2 = LSTM(lstmsize)
    att1 = Attention(wordlen)
    den = Dense(64, activation='tanh')

    # att1 = Lambda(lambda x: K.max(x,axis = 1))

    v1 = word_embedding(input1)
    v2 = word_embedding(input2)
    v11 = lstm1(v1)
    v22 = lstm1(v2)
    v1ls = lstm2(lstm0(v1))
    v2ls = lstm2(lstm0(v2))
    v1 = Concatenate(axis=1)([att1(v1), v11])
    v2 = Concatenate(axis=1)([att1(v2), v22])

    input1c = Input(shape=(charlen,))
    input2c = Input(shape=(charlen,))
    lstm1c = Bidirectional(LSTM(lstmsize))
    att1c = Attention(charlen)
    v1c = char_embedding(input1c)
    v2c = char_embedding(input2c)
    v11c = lstm1c(v1c)
    v22c = lstm1c(v2c)
    v1c = Concatenate(axis=1)([att1c(v1c), v11c])
    v2c = Concatenate(axis=1)([att1c(v2c), v22c])

    mul = Multiply()([v1, v2])
    sub = Lambda(lambda x: K.abs(x))(Subtract()([v1, v2]))
    maximum = Maximum()([Multiply()([v1, v1]), Multiply()([v2, v2])])
    mulc = Multiply()([v1c, v2c])
    subc = Lambda(lambda x: K.abs(x))(Subtract()([v1c, v2c]))
    maximumc = Maximum()([Multiply()([v1c, v1c]), Multiply()([v2c, v2c])])
    sub2 = Lambda(lambda x: K.abs(x))(Subtract()([v1ls, v2ls]))
    matchlist = Concatenate(axis=1)([mul, sub, mulc, subc, maximum, maximumc, sub2])
    matchlist = Dropout(0.05)(matchlist)

    matchlist = Concatenate(axis=1)(
        [Dense(32, activation='relu')(matchlist), Dense(48, activation='sigmoid')(matchlist)])
    res = Dense(1, activation='sigmoid')(matchlist)

    model = Model(inputs=[input1, input2, input1c, input2c], outputs=res)

    return model
