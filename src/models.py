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
from keras.preprocessing import sequence

from attention import PositionEmbedding, Attention

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

def esim_base(pretrained_embedding,
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
    encode = Bidirectional(CuDNNLSTM(lstm_dim, return_sequences=True))
    q1_encoded = encode(q1_embed)
    q2_encoded = encode(q2_embed)

    # Attention
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)

    # Compose
    q1_combined = Concatenate()([q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned)])
    q2_combined = Concatenate()([q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned)])

    compose = Bidirectional(CuDNNLSTM(lstm_dim, return_sequences=True))
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

    # position embedding
    q1_embed = PositionEmbedding()(q1_embed)
    q2_embed = PositionEmbedding()(q2_embed)

    # self attention
    self_attetention = Attention(4, 128)
    q1_encoded = self_attetention([q1_embed, q1_embed, q1_embed])
    q2_encoded = self_attetention([q2_embed, q2_embed, q2_embed])
    #O_seq1 = time_distributed(O_seq1, layers)

    # Encode
    # encode = Bidirectional(CuDNNLSTM(lstm_dim, return_sequences=True))
    # q1_encoded = encode(q1_embed)
    # q2_encoded = encode(q2_embed)

    # Attention
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)

    # Compose
    q1_combined = Concatenate()([q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned)])
    q2_combined = Concatenate()([q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned)])

    compose = Bidirectional(CuDNNLSTM(lstm_dim, return_sequences=True))
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

def siamese_base(pretrained_embedding=None,
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
    # shared_lstm = CuDNNLSTM(n_hidden)

    # # v2 带drop和正则化的多层lstm
    ipt = Input(shape=(input_length, w2v_length))
    dropout_rate = 0.5
    x = Dropout(dropout_rate, )(ipt)
    for i, hidden_length in enumerate(n_hidden):
        # x = Bidirectional(CuDNNLSTM(hidden_length, return_sequences=(i!=len(n_hidden)-1), kernel_regularizer=L1L2(l1=0.01, l2=0.01)))(x)
        x = Bidirectional(CuDNNLSTM(hidden_length, return_sequences=True, kernel_regularizer=L1L2(l1=0.01, l2=0.01)))(x)

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


def siamese(pretrained_embedding=None,
            input_length=MAX_LEN,
            w2v_length=300,
            n_hidden=[64, 64, 64]):
    # 输入层
    left_input = Input(shape=(input_length,))
    right_input = Input(shape=(input_length,))

    # 两个LSTM共享参
    attention_input = Input(shape=(None,))
    embeddings = pretrained_embedding(attention_input)
    # 增加Position_Embedding能轻微提高准确率
    embeddings = PositionEmbedding()(embeddings)
    O_seq1 = Attention(8, 128)([embeddings, embeddings, embeddings])
    layers = [
        Dense(w2v_length),
        Dropout(0.2)
    ]
    O_seq1 = time_distributed(O_seq1, layers)
    O_seq1 = GlobalAveragePooling1D()(O_seq1)
    # O_seq1 = Dropout(0.5)(O_seq1)
    # O_seq1 = BatchNormalization()(O_seq1)

    # O_seq2 = Attention(2, 128)([O_seq1, O_seq1, O_seq1])
    #O_seq2 = GlobalAveragePooling1D()(O_seq2)
    # O_seq2 = BatchNormalization()(O_seq2)

    self_attention = Model(inputs=attention_input, outputs=O_seq1)
    # print(model.summary())

    left_output = self_attention(left_input)
    right_output = self_attention(right_input)

    # merged = Concatenate()([left_output, right_output])

    # out_ = Dense(1, activation='sigmoid')(merged)

    # 距离函数 exponent_neg_manhattan_distance
    malstm_distance = Lambda(lambda x: K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True)),
                             output_shape=lambda x: (x[0][0], 1))([left_output, right_output])
    model = Model([left_input, right_input], [malstm_distance])

    return model
