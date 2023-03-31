# captcha_fssp

- /image_captcha - GET form and POST file for image
- /audio_captcha - GET form and POST file for wav or mp3 record

response: {'result': '888в8'}

# CNN execution path

main.py: subprocess.Popen and captcha_image() -> pipe.stdin.write  and pipe.stdout.readline

service_process.py: load_model('app/m_ctc', encode_single_sample, prediction_model.predict, decode_batch_predictions
- from utils.cnn import encode_single_sample, decode_batch_predictions

# Test

1. docker run -it fssp
2. docker ps
3. docker exec -it 4ca1dcf26c41 bash # id from 2.
4. python test.py

# Captcha image Tensorflow CNN+BiLSTM+CTC

Number of unique characters:  20
Characters present:  ['2', '4', '5', '6', '7', '8', '9', 'б', 'в', 'г', 'д', 'ж', 'к', 'л', 'м', 'н', 'п', 'р', 'с', 'т']

проверка Accurace old:
- Верно 1047
- Всего 1159
- точность 0.903364969801553

проверка Accurace now:
- Верно 1078
- Всего 1159
- точность 0.9301121656600517
