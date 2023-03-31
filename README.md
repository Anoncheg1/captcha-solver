# captcha-solver
For 
- audio - librosa library
- image - Tensorflow Keras

# Image recognition NN model
CNN + Bidirectional LSTM with CTC loss and CTC decoding

https://keras.io/examples/vision/captcha_ocr/

modifications to original:
- This model can train on batches. We encode text to fixed length Tensors of max text lenght.
- Boostig step added.
