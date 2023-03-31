import tensorflow as tf
from tensorflow import keras
import logging
import os
from pathlib import Path
import numpy as np
# own
# from cnncaptcha.sequence import CNNSequence_Simple, ALPHABET_ENCODE, CAPTCHA_LENGTH
from utils.cnn import image_tensorflow_prepare

ALPHABET = ['2', '4', '5', '6', '7', '8', '9', 'б', 'в', 'г', 'д', 'ж', 'к', 'л', 'м', 'н', 'п', 'р', 'с', 'т']

img_width = 200
img_height = 60


alphabet_cats = []
for i in range(len(ALPHABET)):
    categories = keras.utils.to_categorical(i, num_classes=len(ALPHABET))
    alphabet_cats.append(categories)


def num_to_char(n: np.ndarray) -> str:
    """ Mapping characters - integers """
    # for i in n[0]:
    #     import matplotlib.pyplot as plt
    #     plt.bar(x=range(len(i)) ,height=i)
    #     plt.show()

    return ''.join([ALPHABET[i.argmax()] for i in n[0]])

char_to_num = keras.layers.StringLookup(  # ['г'] -> [5] ['г', '2'] -> [ 5, 0 ]
    vocabulary=list(ALPHABET), mask_token=None
)

#
# def char_to_num(chars: str) -> list:
#     """ Mapping characters - integers """
#     cats = []
#     for ch in chars:
#         cats.append(alphabet_cats[ALPHABET.index(ch)])
#         # cats.append(ALPHABET_ENCODE.index(ch))
#         # cat: np.ndarray = keras.utils.to_categorical(n, num_classes=len(ALPHABET_ENCODE))
#         # cats.append(cat)
#
#     # return [np.array(x) for x in cats]
#     # if len(chars) == 4:
#     #     c = np.array([0, 0])
#     # if len(chars) == 5:
#     #     c = np.array([1, 0])
#     # if len(chars) == 6:
#     #     c = np.array([0, 1])
#     return cats


def split_data(images, labels, train_size=0.9, shuffle=True):
    # 1. Get the total size of the dataset
    size = len(images)
    # 2. Make an indices array and shuffle it, if required
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    # 3. Get the size of training samples
    train_samples = int(size * train_size)
    # 4. Split data into training and validation sets
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
    return x_train, x_valid, y_train, y_valid


def encode_single_sample(img_path, label):
    # return {"image": image_tensorflow_prepare(img_path),
    #         "dense2": label,
    #         "dense3": label,
    #         "dense4": label,
    #         "dense5": label,
    #         "dense6": label,
    #         }

    return image_tensorflow_prepare(img_path), label


class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")  # sequence length
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")  # y_true.shape = (None,)

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")  # fill input length for every element in batch
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")  # for every element in batch

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred


def build_model(opt):
    # Inputs to the model
    input_img = keras.layers.Input(
        # shape=(img_height, img_width, 1), name="image", dtype="float32"
    shape = (img_width, img_height, 1), name = "image", dtype = "float32"
    )
    labels = keras.layers.Input(name="label", shape=(None,), dtype="float32")

    # def loss(target_y, predicted_y):
    #     """ Loss for captcha """
    #     print(target_y.shape)
    #     print(predicted_y.shape)
    #     z0 = keras.losses.categorical_crossentropy(target_y[:,0,:], predicted_y[:,0,:], from_logits=False)
    #     z1 = keras.losses.categorical_crossentropy(target_y[:, 1, :], predicted_y[:, 1, :], from_logits=False)
    #     z2 = keras.losses.categorical_crossentropy(target_y[:, 2, :], predicted_y[:, 2, :], from_logits=False)
    #     z3 = keras.losses.categorical_crossentropy(target_y[:, 3, :], predicted_y[:, 3, :], from_logits=False)
    #     z4 = keras.losses.categorical_crossentropy(target_y[:, 4, :], predicted_y[:, 4, :], from_logits=False)
    #     # # # print(z1)
    #     z = tf.keras.layers.average([z0,z1, z2, z3, z4])
    #     # # # # print(z)
    #     z = tf.reduce_mean(z)
    #     # z = keras.losses.categorical_crossentropy(target_y, predicted_y, from_logits=False)
    #     # z = tf.reduce_mean(z)
    #     # print(z)
    #
    #     return z  #tf.reduce_mean(tf.square(target_y - predicted_y))

    # # First conv block
    x = keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = keras.layers.MaxPooling2D((2, 2))(x)  # , name="pool3"

    # Second conv block
    x = keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)  # , name="pool3"

    def cnn():
        # # First conv block 2
        xx = keras.layers.Conv2D(
            64,
            (4, 4),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
            # name="Conv11",
        )(input_img)
        xx = keras.layers.BatchNormalization()(xx)
        xx = keras.layers.MaxPooling2D((4, 4))(xx)  # , name="pool3"
        # xx = keras.layers.AveragePooling2D(2, 2)(xx) # , name="pool11"

        # Second conv block
        # xx = keras.layers.Conv2D(
        #     64,
        #     (6, 6),
        #     activation="relu",
        #     kernel_initializer="he_normal",
        #     padding="same",
        #     # name="Conv22",
        # )(xx)
        # # xx = keras.layers.BatchNormalization()(xx)
        # # xx = keras.layers.MaxPooling2D((2, 2))(xx)  # , name="pool3"
        # xx = keras.layers.AveragePooling2D((2, 2))(xx)  # , name="pool22"

        xx = keras.layers.Flatten()(xx)
        xx = keras.layers.Dense(5, activation='relu')(xx)
        xx = keras.layers.Reshape(target_shape=(5,1))(xx)
        return xx

    # CTCLayer and Bidirectional LSTM expect 2 dimensions,
    # but we have 3 after MaxPooling2D
    # 3 times maxpuling reduced size by 8,
    # 256 is a convolution new dimension
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(20, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)

    # RNNs
    # x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Output layer
    x = keras.layers.Dense(
        len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2"
    )(x)

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="captcha_mode"
    )
    # model = keras.models.Model(
    #     inputs=input_img, outputs=[x1, x2, x3, x4, x5], name="captcha_model"
    # )
    # Optimizer
    # opt = keras.optimizers.Adam()
    # Compile the model and return
    loss = keras.losses.CategoricalCrossentropy(from_logits=False)
    # loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=keras.losses.Reduction.NONE)
    # tf.python.losses.``


    def total_categorical_accuracy(y_true, y_pred):
        # a = tf.cast(tf.math.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1)), dtype=y_pred.dtype)
        a = keras.metrics.categorical_accuracy(y_true, y_pred)
        classes = tf.constant(a.shape[1], a.dtype)
        a2 = tf.reduce_sum(a, axis=-1)
        c = tf.cast(tf.math.equal(a2, classes), dtype=classes.dtype)
        return c


    # tf.argmax(y_true, axis=2), tf.argmax(y_pred, axis=2)

    def diffbatch0(y_true, y_pred):
        diff = tf.abs(y_true - y_pred)
        sum =tf.reduce_sum(diff, axis=1)
        return sum[:,0]

    model.compile(loss=loss, optimizer=opt.optimizer, metrics=["categorical_accuracy",
                                                               # diffbatch0,
                                                               total_categorical_accuracy])  # 'categorical_accuracy'
    # model.compile(loss={'dense2': tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    #                     'dense3': tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    #                     'dense4': tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    #                     'dense5': tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    #                     'dense6': tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    #                     }, optimizer=opt.optimizer,
    #               # metrics={
    #               # 'dense2': 'accuracy',
    #               # 'dense3': 'accuracy',
    #               # 'dense4': 'accuracy',
    #               # 'dense5': 'accuracy',
    #               # 'dense6': 'accuracy'
    #               # }
    #               metrics='accuracy'
    # )










    # model.compile(loss="categorical_crossentropy", optimizer=opt, metrics='accuracy')  # hinge "categorical_crossentropy" # "binary_crossentropy"
    return model


def main(options_set: callable):
    # -- set device manually
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.set_visible_devices(gpus[0], 'GPU') if gpus else print('no gpu!!!!!!!!!!!!!!!!!!!!:', gpus)
    except RuntimeError as e:
        print(e)

    # disable logger
    logging.getLogger('tensorflow').disabled = True

    # get options
    opt = options_set()

    # train_seq = CNNSequence_Simple(opt.batchSize, os.path.join(d, 'train'), opt)
    # test_seq = CNNSequence_Simple(opt.batchSize, os.path.join(d, 'test'), opt)


    images = sorted(list(map(str, list(Path(d + '/gray_allimages/').glob("*.jpg*"))))[:1000])
    images = images[:1000]
    labels = [char_to_num(img.split(os.path.sep)[-1].split("-")[0].lower()) for img in images]
    # print(labels)
    # print(images)
    # exit()


    # Splitting data into training and validation sets
    print(np.array(labels).shape)
    print(np.array(images).shape)

    x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels))
    # x_train = np.array(images)
    # y_train = np.array(labels)
    # images = sorted(list(map(str, list(Path(d + '/test/').glob("*.jpg*")))))
    # labels = [char_to_num(img.split(os.path.sep)[-1].split(".jpg")[0]) for img in images]
    # print(labels)
    # x_valid = np.array(images)
    # y_valid = np.array(labels)
    # y_train0 = [x[:, :, 0] for x in y_train]
    # y_train1 = [x[:, :, 1] for x in y_train]
    # y_train2 = [x[:, :, 2] for x in y_train]
    # y_train3 = [x[:, :, 3] for x in y_train]
    # y_train4 = [x[:, :, 4] for x in y_train]
    #
    # y_valid0 = [x[:, :, 0] for x in y_valid]
    # y_valid1 = [x[:, :, 1] for x in y_valid]
    # y_valid2 = [x[:, :, 2] for x in y_valid]
    # y_valid3 = [x[:, :, 3] for x in y_valid]
    # y_valid4 = [x[:, :, 4] for x in y_valid]
    batch_size = 12
    # train_dataset_x = tf.data.Dataset.from_tensor_slices(x_train).map(encode_single_sample).batch(batch_size).prefetch(1000)
    # train_dataset_y = tf.data.Dataset.from_tensor_slices(y_train).batch(batch_size).prefetch(1000)
    # train_dataset = train_dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.map(encode_single_sample).batch(batch_size).prefetch(3000)
    validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    validation_dataset = validation_dataset.map(encode_single_sample).batch(batch_size).prefetch(3000)
    # for v in train_dataset.take(10):
        # print(v['image'].shape)
        # print(v['label'].shape)
        # print(img, label)
        # print(label.numpy().shape)
        # print()
        # print(elem['label'].numpy().shape)
    # exit()

    epochs = 100
    early_stopping_patience = 1

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
    )

    model = build_model(opt)

    print(model.summary())
    # exit()

    history = model.fit(
        train_dataset,
        # x={'images':train_dataset},
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[early_stopping],
        # shuffle=False,
        validation_batch_size=1
    )

    import cv2 as cv
    # fp = '/home/u2/h4/PycharmProjects/captcha_image/test/2вл24.jpg'
    # img: np.ndarray = cv.imread(fp)
    # gray = clear_captcha(img)

    # def check_gray(fp):
    #     # img = image_tensorflow_prepare(fp)
    #     img = tf.io.read_file(fp)
    #     img = tf.io.decode_jpeg(img, channels=1)
    #     img = tf.image.convert_image_dtype(img, tf.float32)
    #     pred = model.predict(tf.reshape(np.array(img), (1, 60,200,1)))
    #     print(num_to_char(pred))
    #     filename: str = os.path.basename(fp)
    #     print(filename)
    #     return num_to_char(pred) == filename.split('.jpg')[0]
    #
    #
    # c = 0
    # v = '/home/u2/h4/PycharmProjects/captcha_image/test/'
    # for i, filename in enumerate(os.listdir(v)):
    #     c += check_gray(os.path.join(v, filename))
    #
    # print(c, c/100)
    # fp = '/home/u2/h4/PycharmProjects/captcha_image/test/лс657.jpg'
    # fp = '/home/u2/h4/PycharmProjects/captcha_image/train/2r7rб.jpg'

    from keras.models import save_model
    save_model(model, '../m_ctc', include_optimizer=True, save_traces=True)  # , include_optimizer=False, save_traces=False


if __name__ == '__main__':




    # -- copy dataset to memory
    import shutil

    # shutil.rmtree('/dev/shm/train')
    # shutil.rmtree('/dev/shm/test')
    # shutil.copytree('../train', '/dev/shm/train')
    # shutil.copytree('../test', '/dev/shm/test')
    # shutil.rmtree('/dev/shm/gray_allimages')
    # shutil.copytree('../gray_allimages', '/dev/shm/gray_allimages')
    # exit()
    # -- get options
    from options import options_set
    direc = ['..', '/dev/shm']
    d = direc[1]  # CHOOSE! local 0 or memory 1
    main(options_set)
