import os
from typing import List


class Captcha:
    filepath = None
    salvation = None
    idf = None
    audio_salv = None
    audio_lenght = None


def get_solved(wav_dir_path: str) -> List[Captcha]:
    captchas = []
    files = [file for file in os.listdir(wav_dir_path)]
    # -- parse files
    for i, file in enumerate(files):
        file_path = os.path.join(wav_dir_path, file)
        if os.path.isfile(file_path):
            if '_' in file and '.' in file:  # has solvation in name
                idf = file.split('_')[0]
                solv = file.split('_')[1].split('.')[0]
                # print(idf, solv)
                c = Captcha()
                c.filepath = os.path.join(wav_dir_path, file)
                c.salvation = solv
                c.idf = idf
                captchas.append(c)
    return captchas


def calc_alphabet(captchas) -> list:
    salvations = [c.salvation for c in captchas]
    return sorted(list(set(''.join(salvations))))
