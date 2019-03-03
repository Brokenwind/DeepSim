import time

start_time = time.time()
import multiprocessing
import os
import re
import json
import gensim
import jieba
import keras
import keras.backend as K
import numpy as np
import pandas as pd
from itertools import combinations
from keras.activations import softmax
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback, Callback, ReduceLROnPlateau, \
    LearningRateScheduler
from keras.layers import *
from keras.models import Model
from keras.optimizers import SGD, Adadelta, Adam, Nadam, RMSprop
from keras.regularizers import L1L2, l2
from keras.preprocessing.sequence import pad_sequences
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

from gensim.models.word2vec import LineSentence
from gensim.models.fasttext import FastText
import copy
from get_embedding import *
from models import *
from swa import SWA


class LR_Updater(Callback):
    '''
    Abstract class where all Learning Rate updaters inherit from. (e.g., CircularLR)
    Calculates and updates new learning rate and momentum at the end of each batch.
    Have to be extended.
    '''

    def __init__(self, init_lrs):
        self.init_lrs = init_lrs

    def on_train_begin(self, logs=None):
        self.update_lr()

    def on_batch_end(self, batch, logs=None):
        self.update_lr()

    def update_lr(self):
        # cur_lrs = K.get_value(self.model.optimizer.lr)
        new_lrs = self.calc_lr(self.init_lrs)
        K.set_value(self.model.optimizer.lr, new_lrs)

    def calc_lr(self, init_lrs): raise NotImplementedError


class CircularLR(LR_Updater):
    '''
    A learning rate updater that implements the CircularLearningRate (CLR) scheme.
    Learning rate is increased then decreased linearly.
    '''

    def __init__(self, init_lrs, nb, div=4, cut_div=8, on_cycle_end=None):
        self.nb, self.div, self.cut_div, self.on_cycle_end = nb, div, cut_div, on_cycle_end
        super().__init__(init_lrs)

    def on_train_begin(self, logs=None):
        self.cycle_iter, self.cycle_count = 0, 0
        super().on_train_begin()

    def calc_lr(self, init_lrs):
        cut_pt = self.nb // self.cut_div
        if self.cycle_iter > cut_pt:
            pct = 1 - (self.cycle_iter - cut_pt) / (self.nb - cut_pt)
        else:
            pct = self.cycle_iter / cut_pt
        res = init_lrs * (1 + pct * (self.div - 1)) / self.div
        self.cycle_iter += 1
        if self.cycle_iter == self.nb:
            self.cycle_iter = 0
            if self.on_cycle_end: self.on_cycle_end(self, self.cycle_count)
            self.cycle_count += 1
        return res


class CircularLR(LR_Updater):
    '''
    A learning rate updater that implements the CircularLearningRate (CLR) scheme.
    Learning rate is increased then decreased linearly.
    '''

    def __init__(self, init_lrs, nb, div=4, cut_div=8, on_cycle_end=None):
        self.nb, self.div, self.cut_div, self.on_cycle_end = nb, div, cut_div, on_cycle_end
        super().__init__(init_lrs)

    def on_train_begin(self, logs=None):
        self.cycle_iter, self.cycle_count = 0, 0
        super().on_train_begin()

    def calc_lr(self, init_lrs):
        cut_pt = self.nb // self.cut_div
        if self.cycle_iter > cut_pt:
            pct = 1 - (self.cycle_iter - cut_pt) / (self.nb - cut_pt)
        else:
            pct = self.cycle_iter / cut_pt
        res = init_lrs * (1 + pct * (self.div - 1)) / self.div
        self.cycle_iter += 1
        if self.cycle_iter == self.nb:
            self.cycle_iter = 0
            if self.on_cycle_end: self.on_cycle_end(self, self.cycle_count)
            self.cycle_count += 1
        return res


class TimerStop(Callback):
    """docstring for TimerStop"""

    def __init__(self, start_time, total_seconds):
        super(TimerStop, self).__init__()
        self.start_time = start_time
        self.total_seconds = total_seconds
        self.epoch_seconds = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_seconds.append(time.time() - self.epoch_start)

        mean_epoch_seconds = sum(self.epoch_seconds) / len(self.epoch_seconds)
        if time.time() + mean_epoch_seconds > self.start_time + self.total_seconds:
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        print('timer stopping')


def get_model(cfg, model_weights=None):
    print("=======   CONFIG: ", cfg)

    model_type, dtype, input_length, ebed_type, w2v_length, n_hidden, n_epoch, patience = cfg
    embedding = get_embedding_layers(dtype, input_length, w2v_length, with_weight=True)

    if model_type == "esim":
        model = esim(pretrained_embedding=embedding,
                     maxlen=input_length,
                     lstm_dim=300,
                     dense_dim=300,
                     dense_dropout=0.5)
    elif model_type == "decom":
        model = decomposable_attention(pretrained_embedding=embedding,
                                       projection_dim=300, projection_hidden=0, projection_dropout=0.2,
                                       compare_dim=500, compare_dropout=0.2,
                                       dense_dim=300, dense_dropout=0.2,
                                       lr=1e-3, activation='elu', maxlen=input_length)
    elif model_type == "siamese":
        model = siamese(pretrained_embedding=embedding, input_length=input_length, w2v_length=w2v_length,
                        n_hidden=n_hidden)
    elif model_type == "dssm":
        model = DSSM(pretrained_embedding=embedding, input_length=input_length, lstmsize=90)

    if model_weights is not None:
        model.load_weights(model_weights)

    # keras.utils.plot_model(model, to_file=MODEL_DIR+model_type+"_"+dtype+'.png', show_shapes=True, show_layer_names=True, rankdir='TB')
    return model


def save_config(filepath, cfg):
    '''
    保存配置
    :param filepath:
    :param cfg:
    :return:
    '''
    configs = {}
    if os.path.exists(CONFIG_PATH): configs = json.loads(open(CONFIG_PATH, "r", encoding="utf8").read())
    configs[filepath] = cfg
    open(CONFIG_PATH, "w", encoding="utf8").write(json.dumps(configs, indent=2, ensure_ascii=False))


def train_model(model, swa_model, cfg):
    model_type, dtype, input_length, ebed_type, w2v_length, n_hidden, n_epoch, patience = cfg

    data = load_data(dtype, input_length, w2v_length)
    train_x, train_y, test_x, test_y = split_data(data)
    # 每次运行的模型都进行保存，不覆盖之前的结果
    filepath = os.path.join(MODEL_DIR, model_type + "_" + dtype + time.strftime("_%m-%d %H-%M-%S") + ".h5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True,
                                 mode='auto')
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', verbose=0, factor=0.5, patience=2, min_lr=1e-6)
    swa_cbk = SWA(model, swa_model, swa_start=1)

    init_lrs = 0.001
    clr_div, cut_div = 10, 8
    batch_num = (train_x[0].shape[0] - 1) // train_batch_size + 1
    cycle_len = 1
    total_iterators = batch_num * cycle_len
    print("total iters per cycle(epoch):", total_iterators)
    circular_lr = CircularLR(init_lrs, total_iterators, on_cycle_end=None, div=clr_div, cut_div=cut_div)
    callbacks = [checkpoint, earlystop, swa_cbk, circular_lr]
    callbacks.append(TimerStop(start_time=start_time, total_seconds=7100))

    def fit(n_epoch=n_epoch):
        history = model.fit(x=train_x, y=train_y,
                            class_weight={0: 1 / np.mean(train_y), 1: 1 / (1 - np.mean(train_y))},
                            validation_data=((test_x, test_y)),
                            batch_size=train_batch_size,
                            callbacks=callbacks,
                            epochs=n_epoch, verbose=2)
        return history

    loss, metrics = 'binary_crossentropy', ['binary_crossentropy', "accuracy"]

    model.compile(optimizer=Adam(lr=init_lrs, beta_1=0.8), loss=loss, metrics=metrics)
    fit()

    filepath_swa = os.path.join(MODEL_DIR, filepath.split("/")[-1].split(".")[0] + "-swa.h5")
    swa_cbk.swa_model.save_weights(filepath_swa)

    # 保存配置，方便多模型集成
    save_config(filepath, cfg)
    save_config(filepath_swa, cfg)


def train_all_models(index):
    cfg = cfgs[index]
    K.clear_session()
    model = get_model(cfg, None)
    swa_model = get_model(cfg, None)
    train_model(model, swa_model, cfg)

if __name__ == '__main__':
    train_all_models(0)
