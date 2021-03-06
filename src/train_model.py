import time

start_time = time.time()
import json
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau
from keras.optimizers import Adam

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from get_embedding import *
from models import *


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
    '''
    根据constant.py模块中的cfgs配置,获取对应的模型
    :param cfg: constant.py模块中的cfgs配置项
    :param model_weights: 训练好的模型参数位置(训练阶段传None,评估和融合阶段出入模型的参数文件位置)
    :return:
    '''
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

    # 如果有参数位置,则加载相应的参数
    if model_weights is not None:
        model.load_weights(model_weights)
    # 保存模型的图
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


def train_model(model, cfg):
    model_type, dtype, input_length, ebed_type, w2v_length, n_hidden, n_epoch, patience = cfg

    data = load_data(dtype, input_length, w2v_length)
    train_x, train_y, test_x, test_y = split_data(data)
    # 每次运行的模型都进行保存，不覆盖之前的结果
    filepath = os.path.join(MODEL_DIR, model_type + "_" + dtype + time.strftime("_%m-%d %H-%M-%S") + ".h5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True,
                                 mode='auto')
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', verbose=0, factor=0.5, patience=2, min_lr=1e-6)

    init_lrs = 0.001
    clr_div, cut_div = 10, 8
    batch_num = (train_x[0].shape[0] - 1) // train_batch_size + 1
    cycle_len = 1
    total_iterators = batch_num * cycle_len
    print("total iters per cycle(epoch):", total_iterators)
    circular_lr = CircularLR(init_lrs, total_iterators, on_cycle_end=None, div=clr_div, cut_div=cut_div)
    callbacks = [checkpoint, earlystop, circular_lr]

    # 当训练达到一定的时间后就停止训练
    # callbacks.append(TimerStop(start_time=start_time, total_seconds=7100))

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

    # 保存配置，方便多模型集成
    save_config(filepath, cfg)


def train_all_models():
    count = 0
    for cfg in cfgs:
        count += 1
        print("start %d model train" % count)
        K.clear_session()
        model = get_model(cfg, None)
        train_model(model, cfg)


#####################################################################
#                         评估指标和最佳阈值
#####################################################################

def r_f1_thresh(y_pred, y_true, step=1000):
    e = np.zeros((len(y_true), 2))
    e[:, 0] = y_pred.reshape(-1)
    e[:, 1] = y_true
    f = pd.DataFrame(e)
    thrs = np.linspace(0, 1, step + 1)
    x = np.array([f1_score(y_pred=f.loc[:, 0] > thr, y_true=f.loc[:, 1]) for thr in thrs])
    f1_, thresh = max(x), thrs[x.argmax()]
    return f.corr()[0][1], f1_, thresh


#####################################################################
#                         模型评估、模型融合、模型测试
#####################################################################

evaluate_path = os.path.join(MODEL_DIR, "y_pred.pkl")


def evaluate_models():
    train_y_preds, test_y_preds = [], []
    all_cfgs = json.loads(open(CONFIG_PATH, 'r', encoding="utf8").read())
    num_clfs = len(all_cfgs)

    for weight, cfg in all_cfgs.items():
        K.clear_session()
        model_type, dtype, input_length, ebed_type, w2v_length, n_hidden, n_epoch, patience = cfg
        data = load_data(dtype, input_length, w2v_length)
        train_x, train_y, test_x, test_y = split_data(data)
        model = get_model(cfg, weight)
        train_y_preds.append(model.predict(train_x, batch_size=test_batch_size).reshape(-1))
        test_y_preds.append(model.predict(test_x, batch_size=test_batch_size).reshape(-1))
    # train_y_preds的shape=(训练模型个数,训练样本个数)
    train_y_preds, test_y_preds = np.array(train_y_preds), np.array(test_y_preds)
    pd.to_pickle([train_y_preds, train_y, test_y_preds, test_y], evaluate_path)


blending_path = os.path.join(MODEL_DIR, "blending_gdbm.pkl")


def train_blending():
    """ 根据配置文件和验证集的值计算融合模型 """
    train_y_preds, train_y, valid_y_preds, valid_y = pd.read_pickle(evaluate_path)
    train_y_preds = train_y_preds.T
    valid_y_preds = valid_y_preds.T

    '''融合使用的模型'''
    clf = LogisticRegression()
    clf.fit(valid_y_preds, valid_y)

    train_y_preds_blend = clf.predict_proba(train_y_preds)[:, 1]
    r, f1, train_thresh = r_f1_thresh(train_y_preds_blend, train_y)

    valid_y_preds_blend = clf.predict_proba(valid_y_preds)[:, 1]
    r, f1, valid_thresh = r_f1_thresh(valid_y_preds_blend, valid_y)
    pd.to_pickle(((train_thresh + valid_thresh) / 2, clf), blending_path)


def result():
    global df1
    all_cfgs = json.loads(open(CONFIG_PATH, 'r', encoding="utf8").read())
    num_clfs = len(all_cfgs)
    test_y_preds = []
    X = {}
    for cfg in all_cfgs.values():
        model_type, dtype, input_length, ebed_type, w2v_length, n_hidden, n_epoch, patience = cfg
        key_ = "{}_{}".format(dtype, input_length)
        if key_ not in X: X[key_] = input_data(df1["sent1"], df1["sent2"], dtype=dtype, input_length=input_length)

    for weight, cfg in all_cfgs.items():
        K.clear_session()
        model_type, dtype, input_length, ebed_type, w2v_length, n_hidden, n_epoch, patience = cfg
        key_ = "{}_{}".format(dtype, input_length)
        model = get_model(cfg, weight)
        test_y_preds.append(model.predict(X[key_], batch_size=test_batch_size).reshape(-1))

    test_y_preds = np.array(test_y_preds).T
    thresh, clf = pd.read_pickle(blending_path)
    result = clf.predict_proba(test_y_preds)[:, 1].reshape(-1) > thresh

    df_output = pd.concat([df1["id"], pd.Series(result, name="label", dtype=np.int32)], axis=1)

    # topai(1, df_output)


if __name__ == '__main__':
    train_all_models()
    #evaluate_models()
    #train_blending()
