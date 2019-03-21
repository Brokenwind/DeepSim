from get_embedding import *
from models_new import *
from keras.utils import plot_model


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
    if model_weights is not None:
        model.load_weights(model_weights)

    return model


def print_all_models():
    for cfg in cfgs:
        model_type, dtype, input_length, ebed_type, w2v_length, n_hidden, n_epoch, patience = cfg
        K.clear_session()
        model = get_model(cfg, None)
        plot_model(model, to_file=MODEL_DIR + model_type + "_" + dtype + '.png', show_shapes=True,
                   show_layer_names=False,
                   rankdir='TB')
        # model.summary()
        # plot_model(model,to_file='mode.png')


if __name__ == '__main__':
    print_all_models()
