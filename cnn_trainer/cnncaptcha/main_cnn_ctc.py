# https://keras.io/examples/vision/captcha_ocr/
# https://keras.io/examples/audio/ctc_asr/
import tensorflow as tf
from tensorflow import keras
import logging
import os
from pathlib import Path
import numpy as np
# own
from cnncaptcha.sequence import CNNSequence_Simple, ALPHABET, CAPTCHA_LENGTH
from utils.cnn import image_tensorflow_prepare

ALPHABET_ENCODE = ['2', '4', '5', '6', '7', '8', '9', 'б', 'в', 'г', 'д', 'ж', 'к', 'л', 'м', 'н', 'п', 'р', 'с', 'т']
last_layer_size = len(ALPHABET_ENCODE)*CAPTCHA_LENGTH

img_width = 200
img_height = 60


alphabet_cats = []
for i in range(len(ALPHABET_ENCODE)):
    categories = keras.utils.to_categorical(i, num_classes=len(ALPHABET_ENCODE))
    alphabet_cats.append(categories)


def num_to_char(n: np.ndarray) -> str:
    """ Mapping characters - integers """
    # for i in n[0]:
    #     import matplotlib.pyplot as plt
    #     plt.bar(x=range(len(i)) ,height=i)
    #     plt.show()

    return ''.join([ALPHABET_ENCODE[i.argmax()] for i in n[0]])


# def char_to_num(chars: str) -> list:
#     """ Mapping characters - integers """
#     cats = []
#     for ch in chars:
#         v = ch
#         if ch.lower() == 'г':
#             v = 'r'
#         # print(ch.lower(), set(['г', 'г', 'г']), v.lower(), v.lower() == 'г')
#         cats.append(alphabet_cats[ALPHABET_ENCODE.index(v.lower())])
#         # cats.append(ALPHABET_ENCODE.index(ch))
#         # cat: np.ndarray = keras.utils.to_categorical(n, num_classes=len(ALPHABET_ENCODE))
#         # cats.append(cat)
#
#     # return [np.array(x) for x in cats]
#     return cats
#

char_to_num = keras.layers.StringLookup(  # ['г'] -> [5] ['г', '2'] -> [ 5, 0 ]
    vocabulary=list(ALPHABET_ENCODE), mask_token=None
)


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

    return {"image": image_tensorflow_prepare(img_path), "label":label}

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

        input_length = input_length * tf.ones(shape=(batch_len, 1),
                                              dtype="int64")  # fill input length for every element in batch
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")  # for every element in batch

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred

def build_model(opt):
    # from tensorflow.python.keras import layers
    # Inputs to the model
    input_img = keras.layers.Input(
        shape=(img_width, img_height, 1), name="image", dtype="float32"
    )

    labels = keras.layers.Input(name="label", shape=(None,), dtype="float32")

    # First conv block
    x = keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block
    x = keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(64, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)

    # RNNs
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Output layer
    output = keras.layers.Dense(
        len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2"
    )(x)

    # output = CTCLayer(name="ctc_loss")(labels, x)

    # Add CTC layer for calculating CTC loss at each step
    def CTCLoss(y_true, y_pred):
        # Compute the training-time loss value
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        return loss

    # Optimizer
    opt = keras.optimizers.Adam()
    # Define the model
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output,
         name="captcha_model"
    )

    def total_categorical_accuracy(y_true, y_pred):
        # a = tf.cast(tf.math.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1)), dtype=y_pred.dtype)
        a = keras.metrics.categorical_accuracy(y_true, y_pred)
        classes = tf.constant(a.shape[1], a.dtype)
        a2 = tf.reduce_sum(a, axis=-1)
        c = tf.cast(tf.math.equal(a2, classes), dtype=classes.dtype)
        return c

    model.compile(loss=CTCLoss, optimizer=opt, metrics=["categorical_accuracy", total_categorical_accuracy])
    return model


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

    images = sorted(list(map(str, list(Path(d + '/gray_allimages/').glob("*.jpg*")))))
    images = images[:1000]
    # for img in images:
    #     print(img.split(os.path.sep)[-1].split(".jpg")[0])
    # exit()
    # print(images)
    char_to_num = keras.layers.StringLookup(vocabulary=ALPHABET_ENCODE, oov_token="")
    num_to_char = keras.layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
    )

    # print(char_to_num(['г','д','6']))
    # print()
    # exit()

    #
    # for img in images:
    #     r = char_to_num(img.split(os.path.sep)[-1].split(".jpg")[0][-1])
    #     print(img.split(os.path.sep)[-1].split(".jpg")[0], r)
    #     exit()

    labels = [char_to_num(v) for img in images for v in img.split(os.path.sep)[-1].split("-")[0]]

    print(labels)
    # print(images)
    # exit()


    # Splitting data into training and validation sets
    # print(np.array(labels).shape)
    # print(np.array(images).shape)
    # exit()

    x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels))
    # x_train = np.array(images)
    # y_train = np.array(labels)
    # images = sorted(list(map(str, list(Path(d + '/test/').glob("*.jpg*")))))
    # labels = [char_to_num(img.split(os.path.sep)[-1].split(".jpg")[0]) for img in images]
    # x_valid = np.array(images)
    # y_valid = np.array(labels)

    batch_size = 12
    # train_dataset_x = tf.data.Dataset.from_tensor_slices(x_train).map(encode_single_sample).batch(batch_size).prefetch(1000)
    # train_dataset_y = tf.data.Dataset.from_tensor_slices(y_train).batch(batch_size).prefetch(1000)
    # train_dataset = train_dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.map(encode_single_sample).batch(batch_size).prefetch(3000)
    validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    validation_dataset = validation_dataset.map(encode_single_sample).batch(batch_size).prefetch(3000)
    # for v in train_dataset.take(1):
        # print(v[1].shape)
        # print(v[0].shape)
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
    save_model(model, '../m', include_optimizer=False, save_traces=False)  # , include_optimizer=False, save_traces=False


if __name__ == '__main__':
    # -- copy dataset to memory
    # import shutil

    # shutil.rmtree('/dev/shm/train')
    # shutil.rmtree('/dev/shm/test')
    # shutil.copytree('../train', '/dev/shm/train')
    # shutil.copytree('../test', '/dev/shm/test')

    # -- get options
    from options import options_set
    direc = ['..', '/dev/shm']
    d = direc[1]  # CHOOSE! local 0 or memory 1
    main(options_set)
