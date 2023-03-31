import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter

import tensorflow as tf
from tensorflow import keras
# own
from utils.cnn import num_to_char, char_to_num, encode_single_sample, img_width, img_height
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


def all():
    # Path to the data directory
    d = '..'
    data_dir = Path(d + '/gray_allimages/')

    # Get list of all the images
    # images = sorted(list(map(str, list(data_dir.glob("*.jpg*")))))
    # import shutil
    # shutil.rmtree('/dev/shm/train', ignore_errors=True)
    # shutil.copytree('../train', '/dev/shm/train')
    # d = '/dev/shm'

    images = sorted(list(map(str, list(data_dir.glob("*.jpg*")))))
    # images = images[:2000]
    labels_ch = [img.split(os.path.sep)[-1].split("-")[0].lower() for img in images]
    # labels = [char_to_num(list(img.split(os.path.sep)[-1].split("-")[0].lower())) for img in images]
    labels = labels_ch
    # print(labels[1], labels_ch[1])
    # a:tf.Tensor = char_to_num(list('11212'))
    # print(a.numpy())
    # exit()
    characters = set(char for label in labels_ch for char in label)
    characters = sorted(list(characters))

    print("Number of images found: ", len(images))
    print("Number of labels found: ", len(labels))
    print("Number of unique characters: ", len(characters))
    print("Characters present: ", characters)

    # Batch size for training and validation
    batch_size = 1


    # Factor by which the image is going to be downsampled
    # by the convolutional blocks. We will be using two
    # convolution blocks and each block will have
    # a pooling layer which downsample the features by a factor of 2.
    # Hence total downsampling factor would be 4.
    # downsample_factor = 4

    # Maximum length of any captcha in the dataset
    max_length = max([len(label) for label in labels_ch])

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
    # print(x_train.shape)
    # encode_single_sample(x_train[1000], labels[1000])
    #
    # exit()

    # ------------------------- Create Dataset objects ------------------
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, np.array(y_train)))
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

            # At test time, just return the computed predictions
            return y_pred

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
        x = keras.layers.Dense(20, activation="relu", name="dense1")(x)
        x = keras.layers.Dropout(0.2)(x)

        # RNNs
        # x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
        x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(x)

        # Output layer
        x = keras.layers.Dense(
            len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2"
        )(x)

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

        model.compile(optimizer=opt)
        return model


    # Get the model
    model = build_model()
    model.summary()
    # ------------------- training ---------------------------
    epochs = 20
    early_stopping_patience = 1
    # Add early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[early_stopping],
    )

    from keras.models import save_model
    save_model(model, '../m_ctc', save_format='tf', include_optimizer=True, save_traces=True)  # , include_optimizer=False, save_traces=False
    exit()
    # ----------------------------- Inference --------------------------------
    # Get the prediction model by extracting layers till the output layer
    prediction_model = keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="dense2").output
    )
    prediction_model.summary()

    #  Let's check results on some validation samples
    for batch in validation_dataset.take(1):
        batch_images = batch["image"]
        batch_labels = batch["label"]

        preds = prediction_model.predict(batch_images)
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
