import tensorflow as tf
import numpy as np
from tensorflow import keras


# used in main_cnn only
def image_tensorflow_prepare(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_height, img_width])
    # img = tf.image.resize(img, [img_width, img_height])
    img = tf.transpose(img, perm=[1, 0, 2])
    return img


ALPHABET = [' ', '2', '4', '5', '6', '7', '8', '9', 'б', 'в', 'г', 'д', 'ж', 'к', 'л', 'м', 'н', 'п', 'р', 'с', 'т']
# Mapping characters to integers
characters = ALPHABET
# char_to_num = keras.layers.StringLookup(  # ['г'] -> [5] ['г', '2'] -> [ 5, 0 ]
#     # max_tokens=23,
#     # pad_to_max_tokens=True,
#     vocabulary=characters, mask_token=None
# )


def char_to_num(chars: str) -> list:
    """ Mapping characters - integers """
    cats = []
    for ch in chars:
        # cats.append(alphabet_cats[ALPHABET.index(ch)])
        cats.append(ALPHABET.index(ch))
        # cat: np.ndarray = keras.utils.to_categorical(n, num_classes=len(ALPHABET_ENCODE))
        # cats.append(cat)
    return cats


def num_to_char(cats: list) -> list:
    """ Mapping characters - integers """
    chars = []
    for c in cats:
        if c <= 0: # ctc undefined
            continue
        chars.append(ALPHABET[c])
        # cat: np.ndarray = keras.utils.to_categorical(n, num_classes=len(ALPHABET_ENCODE))
        # cats.append(cat)
    return chars


# Mapping integers back to original characters
# num_to_char = keras.layers.StringLookup(
#     vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
# )

# num_to_char = keras.layers.StringLookup(
#     vocabulary=ALPHABET, mask_token=None, invert=True
# )


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
        print("wtf", res)
        res = num_to_char(res)
        if len(res) > 0:
            res = tf.strings.reduce_join(res).numpy().decode("utf-8")
        output_text.append(res)

    # output_text = output_text[0].split('[')[0]
    return output_text



# Desired image dimensions
img_width = 200
img_height = 60


def encode_single_sample(img_path, label):
    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    # 6. Map the characters in label to numbers

    # s = tf.strings.unicode_split(label, input_encoding="UTF-8")
    # label: tf.Tensor = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))

    # tf.pad(label)
    # tf.strings.unicode_split_with_offsets
    # exit()
    # 7. Return a dict as our model is expecting two inputs
    # return img, label
    return {"image": img, "label":label}, label
