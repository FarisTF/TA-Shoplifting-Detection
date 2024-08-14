import tensorflow as tf

saved_model_dir = "LSTM_v6.h5"

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('LSTM_v6.tflite', 'wb') as f:
  f.write(tflite_model)