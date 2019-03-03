from keras.callbacks import Callback

"""
    From the paper:
        Averaging Weights Leads to Wider Optima and Better Generalization
        Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, Andrew Gordon Wilson
        https://arxiv.org/abs/1803.05407
        2018

    Author's implementation: https://github.com/timgaripov/swa
"""


class SWA(Callback):
    def __init__(self, model, swa_model, swa_start):
        super().__init__()
        self.model, self.swa_model, self.swa_start = model, swa_model, swa_start

    def on_train_begin(self, logs=None):
        self.epoch = 0
        self.swa_n = 0

    def on_epoch_end(self, epoch, logs=None):
        if (self.epoch + 1) >= self.swa_start:
            self.update_average_model()
            self.swa_n += 1

        self.epoch += 1

    def update_average_model(self):
        # update running average of parameters
        alpha = 1. / (self.swa_n + 1)
        for layer, swa_layer in zip(self.model.layers, self.swa_model.layers):
            weights = []
            for w1, w2 in zip(swa_layer.get_weights(), layer.get_weights()):
                weights.append((1 - alpha) * w1 + alpha * w2)
            swa_layer.set_weights(weights)
