import argparse
from tensorflow import keras  # ignore error


def options_set():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=int, default=1, help="image channels")
    parser.add_argument("--batchSize", type=int, default=8, help="")
    parser.add_argument("--epochs", type=int, default=70, help="")
    parser.add_argument("--optimizer", choices=["SGD", "Adam", 'Adamax', "Nadam"], default="Adam", help="")
    parser.add_argument("--loss", choices=["squared_hinge", "binary_crossentropy"], default="??", help="")
    parser.add_argument("--lr", type=float, default=0.001, help="")
    parser.add_argument("--decay", type=float, default=0.92, help="")  #20 - 0.05464744489458512, 50 - 0.4738555173642436


    # parser.add_argument("--net", choices=["resnet_one_atten_model, mobilenet"], help="")


    # parser.add_argument("--momentum", type=float, default=0.9, help="")
    # parser.add_argument("--drop", type=float, default=0.1, help="")
    # parser.add_argument("--estop", type=int, default=5, help="")
    # parser.add_argument("--model", choices=["passport_type", "orientation_passport", "passport_main"], help="")

    opt = parser.parse_args()
    opt.optimizer = "Adam"
    # opt.optimizer = "Adamax"
    opt.loss = 'squared_hinge'
    # opt.loss = 'categorical_crossentropy'
    if opt.optimizer == "SGD":
        opt.optim = keras.optimizers.SGD(lr=opt.lr, decay=opt.decay, momentum=opt.momentum, nesterov=True) # decay=opt.decay
        # opt.optim = keras.optimizers.SGD(lr=opt.lr, nesterov=False)
    if opt.optimizer == "Adamax":
        opt.optim = keras.optimizers.Adamax(lr=opt.lr)
    if opt.optimizer == "Nadam":
        opt.optim = keras.optimizers.Nadam(lr=opt.lr, beta_1=opt.decay)
    else:
        opt.optim = keras.optimizers.Adam(lr=opt.lr, beta_1=opt.decay, amsgrad=True) #, decay=opt.decay)

    return opt


if __name__ == '__main__':
    a = options_set()
    print(a)