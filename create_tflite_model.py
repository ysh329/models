import os
import tensorflow as tf
import numpy as np

def create_tflite_model(input_shape, output_model_path):
    # create model
    input_tensor = tf.keras.layers.Input(shape=input_shape[1:], batch_size=input_shape[0])
    reshape_tensor = tf.keras.layers.Reshape((50176, 3))(input_tensor)
    transpose_layer = tf.keras.layers.Permute((2, 1))(reshape_tensor)
    concat_layer = tf.keras.layers.Concatenate(axis=-1)([transpose_layer, transpose_layer])
    output_tensor = tf.keras.layers.Reshape((224, 224, 6))(concat_layer)
    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    model.summary()

    # save model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(output_model_path, 'wb') as f:
        f.write(tflite_model)

      
if __name__ == "__main__":
    input_shape = [1, 3, 224, 224]
    output_model_path = "tflite_custom_layers.tflite"
    create_tflite_model(input_shape, output_model_path)
