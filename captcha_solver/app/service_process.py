import sys
import os
import logging
import tensorflow as tf
from keras.models import load_model
from keras.models import Model
from tensorflow import keras
# own
# from utils.grayscale import clear_captcha
from utils.cnn import encode_single_sample, decode_batch_predictions

MODEL_PATH = './m_ctc'

max_length = 6  # 4-6

if __name__ == '__main__':
    # exit()
    try:
        cpu = tf.config.experimental.list_physical_devices('CPU')
        tf.config.set_visible_devices(cpu[0], 'CPU') if cpu else print('no gpu!!!!!!!!!!!!!!!!!!!!:', cpu, flush=True)
    except RuntimeError as e:
        print(e, flush=True)

    # disable logger
    logging.getLogger('tensorflow').disabled = True

    # -- load model
    model: Model = load_model(MODEL_PATH, compile=False)
    prediction_model = keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="dense2").output
    )
    print('ready', flush=True)
    for line in sys.stdin:
        if 'Exit' == line.rstrip():
            break
        fp = line[:-1]

        if not os.path.isfile(fp):
            print('Error not file path', flush=True)
            continue

        encsample = encode_single_sample(fp, '2222')
        img2 = tf.expand_dims(encsample[0]["image"], axis=0)

        pred = prediction_model.predict(img2, use_multiprocessing=False, verbose=False)
        pred_label = decode_batch_predictions(pred, max_length)[0]
        print('answer:'+pred_label, flush=True)

    # print("Done", flush=True)
