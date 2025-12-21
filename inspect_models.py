import pickle
import joblib
import sklearn

def try_load(path):
    print(f"--- Loading {path} ---")
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        print(f"Loaded with pickle. Type: {type(obj)}")
        return obj
    except Exception as e:
        print(f"Pickle failed: {e}")
        try:
            obj = joblib.load(path)
            print(f"Loaded with joblib. Type: {type(obj)}")
            return obj
        except Exception as e2:
            print(f"Joblib failed: {e2}")
            return None

print("Checking Scaler:")
scaler = try_load('scaler.pkl')
if scaler:
    print(f"Mean: {scaler.mean_ if hasattr(scaler, 'mean_') else 'N/A'}")

print("\nChecking SVM:")
svm = try_load('svm_linear.pkl')
if svm:
    print(f"Classes: {svm.classes_ if hasattr(svm, 'classes_') else 'N/A'}")
    print(f"Has predict_proba: {hasattr(svm, 'predict_proba')}")
    if hasattr(svm, 'predict_proba'):
        # Check if probability=True was set if it's SVC
        print(f"Probability param: {getattr(svm, 'probability', 'N/A')}")
