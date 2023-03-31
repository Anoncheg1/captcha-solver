def categorical_entropy_diff_len(y_true, y_pred):
    input_len = np.ones(y_pred.shape[0]) * y_pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(y_pred, input_length=input_len, greedy=True)[0][0][
              :, :max_length
              ]

    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    print("output_text1", output_text)

    output_text = []
    for res in y_true:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    print("output_text2", output_text)

    print(y_true, results)
    y_true = tf.cast(y_true, dtype=tf.float32)
    results = tf.cast(results, dtype=tf.float32)
    max_len = tf.maximum(y_true.shape[1], results.shape[1]).numpy()
    # print(max_len)
    pad1 = [[0, 0], [0, max_len - y_true.shape[1]]]
    pad2 = [[0, 0], [0, max_len - results.shape[1]]]
    # print(pad1, pad2)
    v1_padded = tf.pad(y_true, pad1, mode='CONSTANT')
    v2_padded = tf.pad(results, pad2, mode='CONSTANT')
    print(v1_padded, v2_padded)
    a = keras.metrics.categorical_crossentropy(v1_padded, v2_padded)
    print("categorical_crossentropy", keras.metrics.categorical_crossentropy(v1_padded, v2_padded))
    print("categorical_accuracy", keras.metrics.categorical_accuracy(v1_padded, v2_padded))
    print("binary_accuracy", keras.metrics.binary_accuracy(v1_padded, v2_padded))
    # classes = tf.constant(a.shape[1], a.dtype)
    a2 = tf.reduce_sum(a, axis=-1)
    # c = tf.cast(tf.math.equal(a2, classes), dtype=classes.dtype)
    return a2


print("categorical_entropy", categorical_entropy_diff_len(batch_labels, preds).numpy())