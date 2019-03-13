from train_model import get_model
from constant import *


def train_all_models():
    count = 0
    for cfg in cfgs:
        count += 1
        print("start %d model train" % count)
        K.clear_session()
        model = get_model(cfg, None)
