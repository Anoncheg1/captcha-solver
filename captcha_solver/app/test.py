from main import app
from flask.testing import FlaskClient
from flask import Response
from pathlib import Path
app.testing = True
client: FlaskClient
import json
import os

with app.test_client() as client:
    print("os.getcwd()", os.getcwd())
    r: Response = client.get('/audio_captcha')
    assert r.status_code == 200
    # print(r.status_code)
    af = './testdata/929014e341a0457f5a90a909b0a51c40.wav'
    r: Response = client.post('/audio_captcha', data={
        'file': Path(af).open('rb')}
    )

    print("os.path.isfile(af)", os.path.isfile(af))
    # assert r.status_code == 200
    # print(json.loads(r.data))

    r: Response = client.get('/image_captcha')
    assert r.status_code == 200
    # print(r.status_code)
    f = './testdata/888в8.jpg'
    print("os.path.isfile(f)", os.path.isfile(f))

    r: Response = client.post('/image_captcha', data={
        'file': Path(f).open('rb')}
    )
    # assert r.status_code == 200
    # print(r)
    print(r.data)
    print(json.loads(r.data))
    assert json.loads(r.data)['result'] == '888в8'
