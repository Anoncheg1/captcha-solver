import os
import shutil
import re
# import cv2

import cv2 as cv

import numpy as np
from random import randint
# own
from utils.grayscale import show_two_images, clear_captcha
# from utils.cnn import ALPHABET
ALPHABET = ['2', '4', '5', '6', '7', '8', '9', 'б', 'в', 'г', 'д', 'ж', 'к', 'л', 'м', 'н', 'п', 'р', 'с', 'т']
# 6 и б отличаются
# т и г тяжело отличить
# п и н тяжело отличить
# нету e
# Press the green button in the gutter to run the script.


# https://bestofphp.com/repo/Gregwar-Captcha-php-image-processing
import re

c1 = re.compile("[а-я0-9]*")
# c2 = re.compile("[а-я0-9А-Я]*")
# c3 = re.compile("[0-9А-Я]*")


def chech_solution(s: str):
    in_alpha = all([x in ALPHABET for x in s])
    if in_alpha and len(s)>1:
        return True
    else:
        return False


def collect_imgpath(dp: str):
    images = []
    print(dp)
    ld = os.listdir(dp)
    # print("(len(ld))", len(ld), ld)
    for i, fp in enumerate(ld):
        fullfp = os.path.join(dp, fp)
        # print(fp.split('-')[0])
        # print(fullfp)
        if '-' in fp:
            solution = fp.split('-')[0].lower()
        else:
            solution = fp.split('.')[0].lower()

        if not chech_solution(solution):
            print(solution, fullfp)
            continue
        print((solution.lower(), fullfp))
        images.append((solution.lower(), fullfp))
    return images


def to_gray_random(p1, p2, howm):
    import random
    shutil.rmtree(p2)
    os.mkdir(p2)
    ld = list(os.listdir(p1))
    for i in range(howm):
        ldi = random.randint(0, len(ld)-1)
        print(i, ldi)
        filename = ld[ldi]
        # solv = filename[0:5]
        file = os.path.join(p1, filename)
        # inv = [4, 8, 67, 72, 80, 95]

        img: np.ndarray = cv.imread(file)
        if img is None:
            print("ALERT: BAD image - it is None")
            continue
        gray = clear_captcha(img)
        cv.imwrite(os.path.join(p2, filename), gray)


def to_gray(source_file_path: str, target_dir: str, solution:str):
    random_int = randint(1, 9999999)
    # p1base = os.path.basename(p1)
    dest_file_path = os.path.join(target_dir, solution + '-' + str(random_int) + '.jpg')
    # ------ to gray
    img: np.ndarray = cv.imread(source_file_path)
    gray = clear_captcha(img)
    if gray is None:
        print(f"Gray is None: {source_file_path}")
        return
    cv.imwrite(dest_file_path, gray)


def to_gray_old(p1, p2):
    shutil.rmtree(p2)
    os.mkdir(p2)
    ld = list(os.listdir(p1))
    print("(len(ld))", len(ld))
    for i in range(len(ld)):
        # ldi = random.randint(0, len(ld)-1)
        # print(i)
        filename = ld[i]

        # solv = filename[0:5]
        file = os.path.join(p1, filename)

        # inv = [4, 8, 67, 72, 80, 95]

        img: np.ndarray = cv.imread(file)
        # cv.imshow('image', img)  # show image in window
        # cv.waitKey(0)  # wait for any key indefinitely
        # cv.destroyAllWindows()  # close window q
        gray = clear_captcha(img)

        filename = filename.lower()
        if '-' in filename:
            filename = filename.split('-')[0] + '.jpg'
        if 'r' in filename:
            filename = ''.join(['г' if x == 'r' else x for x in filename])

        cv.imwrite(os.path.join(p2, filename), gray)


if __name__ == '__main__':

    images = []
    p = '/home/u2/h4/fssp_imgs/wrongImg/all'
    ims = collect_imgpath(p)
    images.extend(ims)
    # print(images)

    p = '/home/u2/h4/PycharmProjects/captcha_image/09_01_23_alpha'
    ims = collect_imgpath(p)
    images.extend(ims)
    # print(images)

    p = '/home/u2/h4/PycharmProjects/captcha_image/jpg1'
    ims = collect_imgpath(p)
    images.extend(ims)

    p = '/home/u2/h4/PycharmProjects/captcha_image/jpg2'
    ims = collect_imgpath(p)
    images.extend(ims)
    # print(images)
    # exit()

    p = '/home/u2/h4/fssp_imgs/checkedImg/'
    folders = os.listdir(p)
    # [print(os.path.join(p, x)) for x in folders]
    imgs = [collect_imgpath(os.path.join(p, x)) for x in folders]
    for x in imgs:
        images.extend(x)

    gray_path = './gray_allimages'
    # -- clear folder:
    [os.remove(os.path.join(gray_path, x)) for x in os.listdir(gray_path)]
    # -- to gray:
    l = len(images)
    for i, imp in enumerate(images):
        # if i < 40000:
        #     continue
        # print(imp)
        sol, fp = imp
        print(l, i, sol, fp)
        to_gray(fp, gray_path, sol)



    # p_train = './train/'
    # p_test = './train/'


    # p = '/home/u2/h4/fssp_imgs/c/'
    # p1 = './phptest/gen_train/'

    #
    # # to_gray_random(p1, p2, 40000)
    #
    # p1 = 'jpg1/'
    # p2 = 'test/'
    #
    # # to_gray(p1, p2)  # 408
    #
    # p1 = 'jpg2/'
    # p2 = 'jpg2_gray/'
    #
    # # to_gray(p1, p2)  # 1162
    #
    # # for filename in os.listdir(p2):
    # #     file = os.path.join(p1, filename)
    # #     file2 = os.path.join('./test/', filename)
    # #     shutil.copy(file, file2)
    # p1 = '09_01_23_alpha/'
    # p2 = 'train/'
    # to_gray(p1, p2)
    #
