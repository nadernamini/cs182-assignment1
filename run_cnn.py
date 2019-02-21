# As usual, a bit of setup

import numpy as np
from deeplearning.classifiers.cnn import *
from deeplearning.data_utils import get_CIFAR10_data
from deeplearning.fast_layers import *
from deeplearning.solver import Solver


def load_data():
    # Load the (preprocessed) CIFAR10 data.
    data = get_CIFAR10_data()
    for k, v in data.iteritems():
        print '%s: ' % k, v.shape
    return data


def run_cnns():
    data = load_data()
    num_train = 20000
    small_data = {
        'X_train': data['X_train'][:num_train],
        'y_train': data['y_train'][:num_train],
        'X_val': data['X_val'],
        'y_val': data['y_val'],
    }
    best_acc, best_model = float('-inf'), None
    learning_rates = np.logspace(-4, -2, 8)
    weight_scales = np.logspace(-4, -2.5, 8)
    for lr in learning_rates:
        for ws in weight_scales:
            # Train a really good model on CIFAR-10
            model = ConvNet(hidden_dims=(500,), dropouts=(0.2, 0.3, 0.4), num_filters=(32, 64), N=2, M=1,
                            input_dim=(3, 32, 32), filter_size=3, num_classes=10, weight_scale=ws,
                            dtype=np.float32, seed=None, use_batchnorm=True)

            solver = Solver(model, small_data,
                            num_epochs=2, batch_size=200,
                            update_rule='adam',
                            optim_config={
                                'learning_rate': lr,
                            },
                            lr_decay=1 - 1e-6,
                            verbose=True, print_every=20)
            solver.train()
            if solver.best_val_acc > best_acc:
                print "******************************************************"
                best_model, best_acc = model, solver.best_val_acc
                print("best valid acc yet: %f   lr: %e ws: %e" % (best_acc, lr, ws))
                print "******************************************************"


if __name__ == "__main__":
    run_cnns()
