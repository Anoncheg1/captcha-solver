# captcha_image

(Приватный) Распознавание капчти

Steps:
1) собираем и проверяем все скаченные каптчи в один массив и кодируем в папку gray_allimages в виде "решение-random.jpg"
- файл main_collectimages_traintest.py
- общую папку gray_allimages разбиваем на train, validation и test во время обучения
2) model cnncaptcha/main_ctc_keras_original_batch1_boost_train.py
- обучаем в два этапа - 1) на всей выборке с большими батчами, 2) с малыми батчами на выборке которая ошибочна распознавалась старой моделью


Проверка на валидационной выборке показала:
- old '/mnt/s/wrongImg/16_03_2023'
- 258 752 0.34308510638297873

- old jpg2
- 1047 1159 0.903364969801553

- old jpg1
- 368 405 0.908641975308642

- new2 '/mnt/s/wrongImg/16_03_2023'
- 556 752 0.7393617021276596

- new2 jpg2
- 1078 1159 0.9301121656600517

- new2 jpg1
- 376 405 0.928395061728395