import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter
from time import sleep
import tensorflow as tf
from tensorflow import keras
# own
from utils.cnn import num_to_char, char_to_num, encode_single_sample, img_width, img_height, ALPHABET
# from utils.grayscale import clear_captcha
# from tensorflow.python.keras import layers


def decode_batch_predictions(pred, max_length):
    """ A utility function to decode the output of the network """
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
              :, :max_length
              ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


# def CTCLoss( labels, logits ): # y_true, y_pred
#     # num_labels = 22
#     # Compute the training-time loss value
#     # batch_len = tf.cast(tf.shape(logits)[0], dtype="int64")
#     # logit_frames = tf.cast(tf.shape(logits)[1], dtype="int64")
#     # logit_length = tf.cast(tf.shape(logits)[2], dtype="int64")
#     label_length = tf.shape(labels)
#     # label_length = tf.random.uniform([batch_len], minval=4,
#     #                                  maxval=6, dtype=tf.int64)
#
#     # input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
#     # label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
#
#     # loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
#     # num_frames = (50, 22)
#     # logit_length = [50] * batch_len
#     logit_length = tf.shape(logits)
#     # logit_length = [num_frames] * batch_size
#     ref_loss = tf.nn.ctc_loss(
#         labels=labels,
#         logits=logits,
#         label_length=label_length,
#         logit_length=logit_length,
#         blank_index=-1,
#         logits_time_major=False
#     )
#     return ref_loss


def CTCLoss(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


def filter_dataset(x_train, y_train, prediction_model, batch_size=15, quantile_min = 0.40, quantile_max = 0.999):

    predicats = []
    lenall = x_train.shape[0]
    i = 0
    while i +batch_size < lenall:
        if i%1000 == 0:
            print(i)
        batch_x = []
        batch_y = []
        for j in range(batch_size):
            x = x_train[i+j]
            y = y_train[i+j]
            v = encode_single_sample(x, y)
            img = v[0]["image"]
            label = v[0]["label"]
            batch_x.append(img)
            batch_y.append(label)
        pred = prediction_model.predict(np.array(batch_x), verbose=0, batch_size=batch_size)
        batch_y = np.array(batch_y)

        loss = CTCLoss(batch_y, pred)
        loss: np.ndarray = loss.numpy().flatten()
        predicats.extend(loss.tolist())  # condition
        i += batch_size
    predicats = np.array(predicats)
    print("predicats.mean()", predicats.mean())
    print("predicats.shape", predicats.shape)
    indexes = np.where((predicats > np.quantile(predicats, quantile_min)) &
                        (predicats < np.quantile(predicats, quantile_max)))
    x_train_filtered = x_train[indexes]
    y_train_filtered = y_train[indexes]
    return x_train_filtered, y_train_filtered


def all():
    # Path to the data directory
    d = '..'


    # Get list of all the images
    # images = sorted(list(map(str, list(data_dir.glob("*.jpg*")))))
    # import shutil
    # shutil.rmtree('/dev/shm/train', ignore_errors=True)
    # shutil.copytree('gray_allimages', '/dev/shm/gray_allimages')
    d = '/dev/shm'
    data_dir = Path(d + '/gray_allimages/')

    images = list(map(str, list(data_dir.glob("*.jpg*"))))
    images = images[1000:115000]

    labels_ch = [img.split(os.path.sep)[-1].split("-")[0].lower() for img in images]
    characters = set(char for label in labels_ch for char in label)
    characters = sorted(list(characters))
    print("Number of images found: ", len(images))
    # print("Number of labels found: ", len(labels))
    print("Number of unique characters: ", len(characters))
    print("Characters present: ", characters)
    z = np.array([len(label) for label in labels_ch])
    bad_inx = np.where((z == 8) | (z == 7) |(z == 2)|(z == 3))
    z = np.delete(z, bad_inx, 0)
    images = np.delete(images, bad_inx, 0)
    # images = np.delete(images, np.where(z == 8), 0)
    # characters = np.delete(characters, np.where(z == 8), 0)
    # z = np.delete(images, np.where(z == 2))
    # z = np.delete(characters, np.where(z == 2))
    max_length = z.max()
    min_length = z.min()
    print()
    print("MAX lenght", max_length, z[z == max_length].shape)
    print("MIN lenght", min_length, z[z == min_length].shape)



    def pad(l: list):
        if len(l) < max_length:
            l.extend([0]*(max_length-len(l)))
        return l



    labels = [pad(char_to_num(img.split(os.path.sep)[-1].split("-")[0].lower())) for img in images]
    z = [len(x) for x in labels]
    assert max(z) == min(z)

    # Batch size for training and validation
    batch_size = 6

    # Factor by which the image is going to be downsampled
    # by the convolutional blocks. We will be using two
    # convolution blocks and each block will have
    # a pooling layer which downsample the features by a factor of 2.
    # Hence total downsampling factor would be 4.
    # downsample_factor = 4

    # Maximum length of any captcha in the dataset

    # labels_ch = [pad(char_to_num(img.split(os.path.sep)[-1].split("-")[0].lower())) for img in images]

    # ----------------------- preprocessing ------------------

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
        # x_train = images[indices[:train_samples]]
        # y_train = [labels[x] for x in indices[:train_samples] ]

        # x_valid = images[indices[train_samples:]]
        # y_valid = [labels[x] for x in indices[train_samples:] ]
        return x_train, x_valid, y_train, y_valid

    # Splitting data into training and validation sets
    # print(np.array(images))
    # print(labels)
    # exit()
    x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels))
    # encode_single_sample(x_train[1000], labels[1000])

    # ------------------------- Create Dataset objects ------------------
    train_dataset = tf.data.Dataset.from_tensor_slices((np.array(x_train), np.array(y_train)))
    # .apply(tf.data.experimental.dense_to_ragged_batch(batch_size))
    train_dataset = train_dataset.map(encode_single_sample).batch(batch_size).prefetch(3000)
    #
    # train_dataset = (
    #     train_dataset.map(
    #         encode_single_sample #, num_parallel_calls=tf.data.AUTOTUNE
    #     )
    #         # .batch(batch_size)
    #         .prefetch(buffer_size=tf.data.AUTOTUNE)
    # )
    # train_dataset = train_dataset.batch(batch_size)
    # exit()
    #
    # print(train_dataset.take(1))
    # for x in train_dataset.take(1):
    #     print(x)
    # exit()

    validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    # .apply(tf.data.experimental.dense_to_ragged_batch(batch_size))
    validation_dataset = validation_dataset.map(encode_single_sample).batch(batch_size).prefetch(3000)

    #
    # validation_dataset = (
    #     validation_dataset.map(
    #         encode_single_sample #, num_parallel_calls=tf.data.AUTOTUNE
    #     )
    #         # .batch(batch_size)
    #         .prefetch(buffer_size=tf.data.AUTOTUNE)
    # )
    # validation_dataset = validation_dataset.batch(batch_size)
    # print(validation_dataset.take(1))
    # exit()

    # ------------------ Model ---------------------

    class CTCLayer(keras.layers.Layer):
        def __init__(self, name=None):
            super().__init__(name=name)
            self.loss_fn = keras.backend.ctc_batch_cost

        def call(self, y_true, y_pred):
            # Compute the training-time loss value and add it
            # to the layer using `self.add_loss()`.
            batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
            input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")  # sequence length
            # print(tf.shape(y_true))
            label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")  # y_true.shape = (None,)

            input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")  # fill input length for every element in batch
            label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")  # for every element in batch

            loss = self.loss_fn(y_true, y_pred, input_length, label_length)
            # loss = self.loss_fn(y_true, y_pred, input_length, 0)
            self.add_loss(loss)

            input_len = tf.ones(shape=(batch_len), dtype="int64") * y_pred.shape[1]
            # Use greedy search. For complex tasks, you can use beam search
            print(label_length)
            print(input_length)
            print(input_len)
            # exit()
            # input_len = tf.ones(y_pred.shape[1]) * y_pred.shape[0]
            # y_pred = tf.reshape(y_pred,[batch_len, y_pred.shape[2], y_pred.shape[1]])[0,:, 0]
            # y_pred = tf.reshape(y_pred[:, :, 0], [batch_len, y_pred.shape[1]])
            results = keras.backend.ctc_decode(y_pred, input_length=input_len
                                               , greedy=True)
            print(results, max_length)
            results = results[:max_length]
            print(results)
            # At test time, just return the computed predictions
            # return y_pred
            return results

    def build_model():
        # Inputs to the model
        input_img = keras.layers.Input(
            shape=(img_width, img_height, 1), name="image", dtype="float32"
        )
        labels = keras.layers.Input(name="label", shape=(None,), dtype="float32")

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
        x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
        x = keras.layers.Dense(40, activation="relu", name="dense1")(x)
        x = keras.layers.Dropout(0.20)(x)

        # RNNs
        # x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True, dropout=0.20))(x)
        x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(x)

        # Output layer
        x = keras.layers.Dense(
            len(ALPHABET) + 1, activation="softmax", name="dense2"
        )(x)
        # output = x

        # Add CTC layer for calculating CTC loss at each step
        output = CTCLayer(name="ctc_loss")(labels, x)

        # Define the model
        model = keras.models.Model(
            inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
        )
        # Optimizer
        opt = keras.optimizers.Adam()

        # # Compile the model and return
        # def my_loss_fn(y_true, y_pred):
        #     squared_difference = tf.square(y_true - y_pred)
        #     return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`
        #
        # def my_accuracy(y_true, y_pred):
        #     squared_difference = tf.square(y_true - y_pred)
        #     return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`
        def tcacc(y_true, y_pred):
            # a = tf.cast(tf.math.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1)), dtype=y_pred.dtype)
            # y_pred = y_pred[:, :max_length]
            a = keras.metrics.categorical_accuracy(y_true, y_pred)
            # classes = tf.constant(a.shape[1], a.dtype)
            a2 = tf.reduce_mean(a, axis=-1)
            # c = tf.cast(tf.math.equal(a2, classes), dtype=classes.dtype)
            # print('a', a)
            return a2

        def CTCLoss2(y_true, y_pred):
            # print(y_true.shape)
            # Compute the training-time loss value
            batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
            input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
            label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

            input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
            label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

            loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
            return loss

        model.compile(optimizer=opt, metrics=[tcacc])
        return model

    # Get the model
    model = build_model()
    model.summary()


    # ------------------- training epoch 1 ---------------------------
    # epochs = 2
    # early_stopping_patience = 1
    # # Add early stopping
    # early_stopping = keras.callbacks.EarlyStopping(
    #     monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
    # )

    # # Train the model
    # history = model.fit(
    #     train_dataset,
    #     validation_data=validation_dataset,
    #     epochs=epochs,
    #     callbacks=[early_stopping]
    # )

    def boost_training(x_train, y_train, x_valid, y_valid, epoches_batches: list):
        """ it is very hard to clear memory with tensorflow
        that is why count of boost steps should be 2 or 3"""
        x_train_filtered = x_train
        y_train_filtered = y_train
        x_valid_filtered = x_valid
        y_valid_filtered = y_valid
        loop_i = 0
        for i, eb in enumerate(epoches_batches):
            epoches, batches = eb
            print("LOOP_I", i)
            if i == 0:
                x_train_filtered, y_train_filtered = x_train, y_train
                train_dataset = tf.data.Dataset.from_tensor_slices(
                    (x_train_filtered, y_train_filtered))
                train_dataset = train_dataset.map(encode_single_sample).batch(batches).prefetch(3000)
                valid_dataset = tf.data.Dataset.from_tensor_slices(
                    (x_valid_filtered, y_valid_filtered))
                valid_dataset = valid_dataset.map(encode_single_sample).batch(batches).prefetch(3000)
                history = model.fit(
                    train_dataset,
                    validation_data=validation_dataset,
                    epochs=epoches
                )
                continue

            # ----------------------------- Inference --------------------------------
            # Get the prediction model by extracting layers till the output layer
            if i > 1:
                del prediction_model
                keras.backend.clear_session()
            prediction_model = keras.models.Model(
                model.get_layer(name="image").input, model.get_layer(name="dense2").output, trainable=False
            )
            prediction_model.summary()

            # x_train, x_valid, y_train, y_valid
            x_train_filtered, y_train_filtered = filter_dataset(x_train_filtered, y_train_filtered, prediction_model)
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (x_train_filtered, y_train_filtered))  # .apply(tf.data.experimental.dense_to_ragged_batch(batch_size))
            train_dataset = train_dataset.map(encode_single_sample).batch(batches).prefetch(3000)
            x_valid_filtered, y_valid_filtered = filter_dataset(x_valid_filtered, y_valid_filtered, prediction_model)
            valid_dataset = tf.data.Dataset.from_tensor_slices(
                (x_valid_filtered, y_valid_filtered))  # .apply(tf.data.experimental.dense_to_ragged_batch(batch_size))
            valid_dataset = valid_dataset.map(encode_single_sample).batch(batches).prefetch(3000)
            print("train_dataset.cardinality", train_dataset.cardinality().numpy())
            print("valid_dataset.cardinality", valid_dataset.cardinality().numpy())
            # ------------------- training epoch 2 ---------------------------

            # early_stopping_patience = 1
            # Add early stopping
            # early_stopping = keras.callbacks.EarlyStopping(
                # monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
            # )
            print("epochs", epoches, "batches", batches)
            # sleep(3)

            # Train the model
            history = model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=epoches,
                # callbacks=[early_stopping],
            )

    boost_training(x_train, y_train, x_valid, y_valid, epoches_batches=[(2,10), (2,2)])

    from keras.models import save_model
    save_model(model, '../m_ctc', save_format='tf', include_optimizer=True, save_traces=True)
    print("MODEL SAVED!")
    # , include_optimizer=False, save_traces=False

    prediction_model = keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="dense2").output
    )
    #  Let's check results on some validation samples
    for batch in validation_dataset.take(1):
        batch_images = batch["image"]
        batch_labels = batch["label"]

        preds = prediction_model.predict(batch_images, batch_size=1)
        pred_texts = decode_batch_predictions(preds, max_length)

        orig_texts = []
        for label in batch_labels:
            label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
            orig_texts.append(label)

        _, ax = plt.subplots(4, 4, figsize=(15, 5))
        for i in range(len(pred_texts)):
            img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
            img = img.T
            title = f"Prediction: {pred_texts[i]}"
            ax[i // 4, i % 4].imshow(img, cmap="gray")
            ax[i // 4, i % 4].set_title(title)
            ax[i // 4, i % 4].axis("off")
    plt.show()


if __name__ == '__main__':
    # cp -r h4/PycharmProjects/captcha_image/gray_allimages /dev/shm/gray_allimages
    all()
