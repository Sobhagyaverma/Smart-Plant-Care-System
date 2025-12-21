import tensorflow as tf

try:
    print("Loading model...")
    model = tf.keras.models.load_model("cnn_mobilenetv2.keras")
    model.summary()
    
    print("\nLast Layer Config:")
    print(model.layers[-1].get_config())
    
    print("\nOutput Shape:")
    print(model.output_shape)
except Exception as e:
    print(f"Error: {e}")
