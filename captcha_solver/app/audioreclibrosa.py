import librosa
import numpy as np
from typing import List
# own
from utils.audio import Captcha

ALPHABET = ('2', '4', '5', '6', '7', '8', '9', 'б', 'в', 'г', 'д', 'ж', 'к', 'л', 'м', 'н', 'п', 'р', 'с', 'т')
ALPHABET_FEATURE = [105.0, 160.0, 74.0, 76.0, 94.0, 146.0, 148.0, 86.0, 106.0, 92.0, 90.0, 83.0, 91.0, 99.0, 104.0, 96.0, 87.0, 79.0, 65.0, 87.0]
FEATURE_RMS_T = 0.093
FEATURE_RMS_P = 0.086


def splitbysilence(y):
    td = 18.2
    hop_length = 9
    intervals = librosa.effects.split(y, top_db=td, hop_length=hop_length)
    pieces = []
    for iv in intervals:
        p = y[iv[0]:iv[1]]
        pa, _ = librosa.effects.trim(p, ref=0.45, top_db=20, hop_length=3)
        pieces.append(pa)
    return pieces


def get_alpha_by_feature(f: float or List[float]) -> str:
    global ALPHABET, ALPHABET_FEATURE
    if isinstance(f, float):
        return ALPHABET[ALPHABET_FEATURE.index(f)]
    else:
        a = [ALPHABET[ALPHABET_FEATURE.index(fi)] for fi in f]
        return ''.join(a)


def calc_feature(sound: np.ndarray, sr):
    f_d = librosa.get_duration(y=sound)
    f_mfcc = np.mean(librosa.feature.mfcc(y=sound, sr=sr, n_fft=100, n_mfcc=20))
    f_2 = np.median(librosa.feature.rms(y=sound, hop_length=100))
    return (abs(f_mfcc) + f_d*1000 + f_2*800)//4   # 4 is enough


def calc_features(c: Captcha or str) -> List[int] or int:
    """ c file math """
    if isinstance(c, Captcha):
        y, sr = librosa.load(c.filepath)
    else:
        y, sr = librosa.load(c)
    split = splitbysilence(y)
    return [calc_feature(sound, sr) for sound in split]


def max_db(y, n_fft=2048):
    s = librosa.stft(y, n_fft=n_fft, hop_length=n_fft // 2)
    d = librosa.amplitude_to_db(np.abs(s), ref=np.max)
    return np.max(abs(d))


def get_alpphabet_feature(alphabet: str or list, captchas_solved: List[Captcha]):
    """ alpha_features = get_alpphabet_feature(ALPHABET, captchas) """
    features = []
    for a in alphabet:
        for c in captchas_solved:
            if a in c.salvation:
                y, sr = librosa.load(c.filepath)
                pieces = splitbysilence(y)
                position = c.salvation.index(a)
                sound: np.ndarray = pieces[position]
                f = calc_feature(sound, sr)
                features.append(f)
                break
    assert len(features) == len(alphabet)
    return features


def audio_decode(file_patch: str) -> str:
    y, sr = librosa.load(file_patch)
    yl = splitbysilence(y)
    features: list = [calc_feature(sound, sr) for sound in yl]
    sol = get_alpha_by_feature(features)
    for i, ch in enumerate(sol):
        if 'п' == ch or 'т' == ch:
            sol_l = list(sol)
            f = np.median(librosa.feature.rms(y=yl[i], hop_length=100))
            if round(float(f), 3) == FEATURE_RMS_P:
                sol_l[i] = 'п'
            elif round(float(f), 3) == FEATURE_RMS_T:
                sol_l[i] = 'т'
            sol = ''.join(sol_l)
    return sol
