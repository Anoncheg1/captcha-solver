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

max_chats = 6

alphabet_cats = keras.utils.to_categorical(list(range(len(ALPHABET))), num_classes=len(ALPHABET))
empty_cats = np.zeros(alphabet_cats[0].shape)

batch_size = 1


def num_to_char(n: np.ndarray) -> str:
    """ Mapping characters - integers """
    # for i in n[0]:
    #     import matplotlib.pyplot as plt
    #     plt.bar(x=range(len(i)) ,height=i)
    #     plt.show()

    return ''.join([ALPHABET[i.argmax()] for i in n[0]])


def char_to_num(chars: str):
    """ Mapping characters - integers """
    cats = []
    for ch in chars:
        cats.append(alphabet_cats[ALPHABET.index(ch)])
        # cats.append(ALPHABET_ENCODE.index(ch))
        # cat: np.ndarray = keras.utils.to_categorical(n, num_classes=len(ALPHABET_ENCODE))
        # cats.append(cat)
    for i in range(6-len(cats)):
        cats.append(empty_cats)

    return [np.array(x) for x in cats]
    # if len(chars) == 4:
    #     c = np.array([0, 0])
    # if len(chars) == 5:
    #     c = np.array([1, 0])
    # if len(chars) == 6:
    #     c = np.array([1, 1])
    # return cats, c


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
    labels, counts = zip(*labels)
    x_train, y_train, y_train2 = images[indices[:train_samples]], np.array(labels)[indices[:train_samples]], \
                                 np.array(counts)[indices[:train_samples]]
    x_valid, y_valid, y_valid2 = images[indices[train_samples:]], np.array(labels)[indices[train_samples:]], \
                                 np.array(counts)[indices[train_samples:]]
    return x_train, x_valid, y_train, y_train2, y_valid, y_valid2


def encode_single_sample(img_path, label, count):
    return {"image": image_tensorflow_prepare(img_path), "count": count, "label": label}  # image_tensorflow_prepare(img_path), label


class LossLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.losses.binary_crossentropy
        self.ch4_c = np.array((0, 0))
        self.ch5_c = tf.constant(np.array((0, 1)), dtype=tf.float32)
        # self.ch5_c = np.tile(self.ch5_c, (batch_size, 1))

        a = np.ones((6, 20))

        self.ch4 = a.copy()
        self.ch4[-2:, :] = 0  # zero two last character
        self.ch4 = tf.constant(self.ch4, tf.float32)
        # print("wtf1", self.ch4.shape)
        # self.ch4 = np.tile(self.ch4, (batch_size, 1, 1))
        # print("wtf", self.ch4.shape)
        # exit()
        self.ch5 = a.copy()
        self.ch5[-1:, :] = 0  # zero one last character
        self.ch5 = tf.constant(self.ch5, tf.float32)
        # self.ch5 = np.tile(self.ch5, (batch_size, 1, 1))

    def call(self, y_pred, y_pred_count, y_true, count):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        # batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        # input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        # label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
        #
        # input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        # label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        # print(count)
        # print(y_pred.shape)
        # y_pred = tf.cond(tf.equal(count, self.ch4_c), lambda: tf.multiply(y_pred, self.ch4), lambda: y_pred)
        # y_pred = tf.cond(tf.equal(count, self.ch5_c), lambda: tf.multiply(y_pred, self.ch5), lambda: y_pred)

        # def a():
        #     return y_pred
        # print(count.shape, self.ch5_c.shape)
        # count = tf.convert_to_tensor(count)
        # print(count)
        # print(count)
        # exit()

        @tf.function
        def aa(v):
            c = v[0]
            pred = v[1]
            # zero 6 and 5 character before calc loss
            # print('as', pred.shape, self.ch4.shape)
            pred = tf.cond(tf.reduce_all(tf.equal(c, self.ch4_c)), lambda: tf.multiply(pred, self.ch4), lambda: pred)
            print(pred)  # !!!! REQUIRED!!!!
            pred = tf.cond(tf.reduce_all(tf.equal(c, self.ch5_c)), lambda: tf.multiply(pred, self.ch5), lambda: pred)
            print(pred)  # !!!! REQUIRED!!!!
            # print('asdd', tf.math.multiply(pred, self.ch5))

            # print(c.shape, self.ch5_c.shape)
            # r = tf.cond(tf.reduce_all(tf.equal(c, self.ch5_c)), lambda: pred, lambda: pred)
            # print(r)
            # print(r.shape)
            # print(r.eval())
            return pred  # tf.cond(tf.equal(c, self.ch5_c), lambda: pred, lambda: pred)
        # print(count.shape, y_pred.shape)
        # print(count)
        # print(y_pred.shape)
        y_pred = tf.map_fn(aa, (count, y_pred), dtype=y_pred.dtype, infer_shape=True)  # dtype=(tf.float32, tf.float32)
        # print(y_pred.shape)
        loss1 = self.loss_fn(y_true, y_pred)
        loss1 = tf.reduce_mean(loss1, axis=1)  # keep batch dimension
        loss2 = self.loss_fn(count, y_pred_count)
        loss = tf.reduce_sum((loss1, loss2))
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred


def build_model(opt):
    # Inputs to the model
    input_img = keras.layers.Input(
        shape=(img_width, img_height, 1), name="image", dtype="float32"
    )
    count = keras.layers.Input(name="count", shape=(2,), dtype="float32")
    label = keras.layers.Input(name="label", shape=(6, 20), dtype="float32")

    # First conv block
    x = keras.layers.Conv2D(
        16,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block
    x = keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

    # # Third conv block
    # x = keras.layers.Conv2D(
    #     256,
    #     (2, 2),
    #     activation="relu",
    #     kernel_initializer="he_normal",
    #     padding="same",
    #     name="Conv3",
    # )(x)
    # x = keras.layers.MaxPooling2D((2, 2), name="pool3")(x)

    # CTCLayer and Bidirectional LSTM expect 2 dimensions,
    # but we have 3 after MaxPooling2D
    # 3 times maxpuling reduced size by 8,
    # 256 is a convolution new dimension
    new_shape = ((img_width // 4), (img_height // 4) * 32)
    # new_shape = (24000,)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(32, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.15)(x)

    # RNNs
    # x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True, dropout=0.25))(x)
    # x = keras.layers.LSTM(64, return_sequences=False, dropout=0.25)(x)

    # x = keras.layers.Concatenate()([x, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13]) # x13, x14, x15, x16, x17, x18, x19
    xv = keras.layers.Flatten()(x)
    x = keras.layers.Dense(120, activation="softmax", name="dense11")(xv)

    x = keras.layers.Reshape(target_shape=(6, 20), name="reshape2")(x)
    output2 = keras.layers.Dense(2, activation="softmax", name="dense12")(xv)  # count of characters

    # x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.Dropout(0.15)(x)
    # output = keras.layers.Flatten()(x)
    # x = keras.layers.Flatten()(x)
    output1 = LossLayer()(x, output2, label, count)

    # Define the model
    model = keras.models.Model(
        inputs=[input_img, label, count], outputs=[output1, output2], name="captcha_model"
    )
    # model = keras.models.Model(
    #     inputs=input_img, outputs=[x1, x2, x3, x4, x5], name="captcha_model"
    # )
    # Optimizer
    # opt = keras.optimizers.Adam()
    # Compile the model and return
    # loss = keras.losses.CategoricalCrossentropy(from_logits=False)
    # loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=keras.losses.Reduction.NONE)
    # tf.python.losses.``

    def total_categorical_accuracy(y_true, y_pred):
        # a = tf.cast(tf.math.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1)), dtype=y_pred.dtype)
        # print("y_true.shape, y_pred.shape")
        # print(y_true.shape, y_pred.shape)
        print(y_true, y_pred)
        a = keras.metrics.categorical_accuracy(y_true, y_pred)  # shape of batches
        # classes = tf.constant(a.shape[1], a.dtype)
        # a2 = tf.reduce_sum(a, axis=-1)
        a2 = tf.reduce_sum(a)
        # c = tf.cast(tf.math.equal(a2, classes), dtype=classes.dtype)
        return a2


    # tf.argmax(y_true, axis=2), tf.argmax(y_pred, axis=2)

    def diffbatch0(y_true, y_pred):
        diff = tf.abs(y_true - y_pred)
        sum = tf.reduce_sum(diff, axis=1)
        return sum[:, 0]
    # loss=loss,
    model.compile(optimizer=opt.optimizer)  # 'categorical_accuracy' , metrics=[total_categorical_accuracy]
    # metrics=["categorical_accuracy",
    #                                                                # diffbatch0,
    #                                                                total_categorical_accuracy]
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

    images = sorted(list(map(str, list(Path(d + '/gray_allimages/').glob("*.jpg*")))))
    # labels = [img.split(os.path.sep)[-1].split("-")[0].lower() for img in images]
    labels = [char_to_num(img.split(os.path.sep)[-1].split("-")[0].lower()) for img in images]
    # print(labels)
    # exit()
    # print(images)

    # Splitting data into training and validation sets
    # print(np.array(labels).shape)
    # print(np.array(images).shape)
    # exit()
    # print(np.array(labels).shape)
    x_train, x_valid, y_train, y_train2, y_valid, y_valid2 = split_data(np.array(images), labels)
    # x_train = np.array(images)
    # y_train = np.array(labels)
    # images = sorted(list(map(str, list(Path(d + '/test/').glob("*.jpg*")))))
    # labels = [char_to_num(img.split(os.path.sep)[-1].split(".jpg")[0]) for img in images]
    # print(labels)
    # x_valid = np.array(images)
    # y_valid = np.array(labels)
    # print(labels)


    # train_dataset_x = tf.data.Dataset.from_tensor_slices(x_train).map(encode_single_sample).batch(batch_size).prefetch(1000)
    # train_dataset_y = tf.data.Dataset.from_tensor_slices(y_train).batch(batch_size).prefetch(1000)
    # train_dataset = train_dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train, y_train2))
    train_dataset = train_dataset.map(encode_single_sample).batch(batch_size).prefetch(3000)
    validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid, y_valid2))
    validation_dataset = validation_dataset.map(encode_single_sample).batch(batch_size).prefetch(3000)
    for v in train_dataset.take(10):
        print(v['image'].shape)
        print(v['label'].shape)
        print(v['count'].shape)
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

    history = model.fit(
        train_dataset,
        # x={'images':train_dataset},
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[early_stopping],
        # shuffle=False,
        validation_batch_size=batch_size
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
    save_model(model, '../m', include_optimizer=True, save_traces=True)  # , include_optimizer=False, save_traces=False


if __name__ == '__main__':

    # -- copy dataset to memory
    import shutil

    # shutil.rmtree('/dev/shm/train')
    # shutil.rmtree('/dev/shm/test')
    # shutil.copytree('../train', '/dev/shm/train')
    # shutil.copytree('../test', '/dev/shm/test')
    # shutil.copytree('../gray_allimages', '/dev/shm/gray_allimages')

    # -- get options
    from options import options_set
    direc = ['..', '/dev/shm']
    d = direc[0]  # CHOOSE! local 0 or memory 1
    main(options_set)
