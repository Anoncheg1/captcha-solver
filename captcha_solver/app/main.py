from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for
import os
import uuid
import json
import re
import tempfile
# own
from audioreclibrosa import audio_decode
# image
import cv2 as cv
import subprocess
# own
from utils.grayscale import clear_captcha

tensorflow_process_path = ["/tensorflow/bin/python3", "service_process.py"]

regex_alpha = re.compile(r'[^a-zA-z0-9]')

UPLOAD_FOLDER = '/tmp/'
ALLOWED_EXTENSIONS_AUDIO = {'wav', 'mp3'}
ALLOWED_EXTENSIONS_IMAGE = {'jpg', 'jpeg', 'png', 'webp', 'tiff', 'bmp'}
ct_json = {'Content-Type': 'application/json'}


FILE_UPLOAD_HTML = '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

# ---- start Flask
app = Flask(__name__)
app.test_client()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---- start subprocess with tensorflow
# -- unset PYTHONPATH for Tensorflow - it uses own numpy version
my_env = os.environ.copy()
my_env["PYTHONPATH"] = ""
pipe = subprocess.Popen(tensorflow_process_path, text=True, shell=False,
                        universal_newlines=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=my_env)

if pipe.returncode is not None:
    raise Exception("no subprocess service tensorflow")
while pipe.stdout.readline() != 'ready\n':  # wait for ready response
    pass


def allowed_file_image(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_IMAGE


def allowed_file_audio(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_AUDIO


@app.route('/', methods=['GET', 'POST'])
def index():
    return redirect(url_for('audio_captcha'))


@app.route('/audio_captcha', methods=['GET', 'POST'])
def audio_captcha():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if not file or not allowed_file_audio(file.filename):
            print(file, file.filename, allowed_file_audio(file.filename))
            return {'error': 'Bad Request, file'}, 400, ct_json
        filename = secure_filename(file.filename)

        uuidstr = str(uuid.uuid4().hex)

        # replace filename with uuid + last 4 ASCII symbols
        file_name = uuidstr + '.' + filename[-4:]
        with tempfile.TemporaryDirectory() as tmp_dir:
            fp = os.path.join(tmp_dir, file_name)
            file.save(fp)
            decoded = audio_decode(fp)
        raw = ''

        return json.dumps({'result': decoded, 'original': str(raw)}), 200, ct_json

    return FILE_UPLOAD_HTML


@app.route('/image_captcha', methods=['GET', 'POST'])
def captcha_image():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if not file or not allowed_file_image(file.filename):
            return {'error': 'Bad Request, file'}, 400, ct_json

        filename = secure_filename(file.filename)
        with tempfile.TemporaryDirectory() as tmpdir:
            # -- save input file
            fp = os.path.join(tmpdir, filename)
            file.save(fp)
            img = cv.imread(fp)
            # print("img", img)
            gray = clear_captcha(img)
            cv.imwrite(fp, gray)
            # -- send file to subprocess with Tensorflow
            pipe.stdin.write(fp + "\n")
            pipe.stdin.flush()

            r = pipe.stdout.readline()
            if not r.startswith('answer'):
                print("pipe.stdout.readline()", r)
                return {'error': 'Error in subprocess Tensorflow'}, 500, ct_json
            label = r[:-1][7:]

            # encsample = encode_single_sample(fp, '2222')
            # img2 = tf.expand_dims(encsample["image"], axis=0)
            #
            # pred = prediction_model.predict(img2, use_multiprocessing=True, verbose=False)
            # pred_label = decode_batch_predictions(pred, max_length)
            # pred_label = pred_label[0].split('[')[0]
            return json.dumps({'result': label}), 200, ct_json

    elif request.method == 'GET':
        return FILE_UPLOAD_HTML, 200, {'Content-Type': 'text/html'}



if __name__ == '__main__':
    # print(audio_decode('/home/u2/h4/gitlabprojects/captcha_fssp/app/929014e341a0457f5a90a909b0a51c40.wav'))
    # print(untranslit('12asdD_,').lower())
    app.run(debug=False)
