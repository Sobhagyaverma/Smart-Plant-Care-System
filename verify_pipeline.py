import tensorflow as tf
import joblib
import numpy as np
import os

def verify_pipeline():
    print("1. Loading Models...")
    try:
        cnn = tf.keras.models.load_model("cnn_mobilenetv2.keras")
        # User requested cnn.layers[-3].output
        feature_extractor = tf.keras.Model(inputs=cnn.input, outputs=cnn.layers[-3].output)
        print("   - CNN Loaded")
        
        scaler = joblib.load("scaler.pkl")
        print("   - Scaler Loaded")
        
        svm = joblib.load("svm_linear.pkl")
        print("   - SVM Loaded")
        
        # Load labels
        if os.path.exists("labels.txt"):
             with open("labels.txt", "r") as f:
                  labels = [l.strip() for l in f.readlines()]
             print(f"   - Labels Loaded: {len(labels)} classes")
        else:
             print("   - Labels file NOT found")
             
    except Exception as e:
        print(f"FAILED to load models: {e}")
        return

    print("\n2. Running Dummy Prediction...")
    try:
        # Dummy image: 1, 224, 224, 3
        dummy_img = np.random.rand(1, 224, 224, 3).astype(np.float32)
        
        # Preprocess
        dummy_img = tf.keras.applications.mobilenet_v2.preprocess_input(dummy_img * 255)
        
        # Extract
        features = feature_extractor.predict(dummy_img, verbose=0)
        print(f"   - Features shape: {features.shape}")
        
        # Scale
        scaled = scaler.transform(features)
        print(f"   - Scaled features mean: {scaled.mean()}")
        
        # SVM
        decision = svm.decision_function(scaled)[0]
        
        # Softmax
        exp_scores = np.exp(decision - np.max(decision))
        probs = exp_scores / exp_scores.sum()
        print(f"   - Top prediction index: {np.argmax(probs)}")
        print(f"   - Top prediction label: {labels[np.argmax(probs)]}")
        
        print("\n✅ Verification Successful!")
    except Exception as e:
        print(f"FAILED during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_pipeline()
