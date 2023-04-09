used with [captcha-image-to-grayscale](https://github.com/Anoncheg1/captcha-image-to-grayscale)
# captcha-solver
Captcha type to solve and used libraries for solvation:
- audio - librosa 
- image - Tensorflow Keras, OpenCV

# Image recognition NN model
CNN + Bidirectional LSTM with CTC loss and CTC decoding, as described:

https://keras.io/examples/vision/captcha_ocr/

modifications to original:
- This model can train on batches. We encode text to fixed length Tensors with zeroes padding.
- Boosting step added.

Each image is cleared and converted to gray with special function with OpenCV library.

# Audio recognition
1. Audio sptitted to characters by silence.

2. For each character we extract features:
- duration
- mfcc
- rms

3. Compare sum of features with precalcuated sums of charactes.
4. For hard to destinguish characters we make additional comparision.

