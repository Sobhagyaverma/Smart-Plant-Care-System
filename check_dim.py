import pickle
import joblib

def check_dim():
    try:
        scaler = joblib.load('scaler.pkl')
        print(f"Scaler mean shape: {scaler.mean_.shape}")
    except Exception as e:
        print(f"Scaler load failed: {e}")

    try:
        svm = joblib.load('svm_linear.pkl')
        print(f"SVM coef shape: {svm.coef_.shape}")
    except Exception as e:
        print(f"SVM load failed: {e}")

check_dim()
