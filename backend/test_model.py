import pickle
import sys
import os

print("Testing model loading...")

try:
    # Test numpy import
    import numpy as np
    print(f"✅ NumPy version: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy import error: {e}")
    sys.exit(1)

try:
    # Test scikit-learn import
    import sklearn
    print(f"✅ Scikit-learn version: {sklearn.__version__}")
except ImportError as e:
    print(f"❌ Scikit-learn import error: {e}")
    sys.exit(1)

try:
    # Test pandas import
    import pandas as pd
    print(f"✅ Pandas version: {pd.__version__}")
except ImportError as e:
    print(f"❌ Pandas import error: {e}")
    sys.exit(1)

# Test model file existence
model_path = "../models/real_data_question_model.pkl"
if not os.path.exists(model_path):
    print(f"❌ Model file not found: {model_path}")
    sys.exit(1)

print(f"✅ Model file exists: {model_path}")

# Test model loading
try:
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print("✅ Model loaded successfully")
    print(f"Model keys: {list(model_data.keys())}")
    
    # Test model info
    if 'model_info' in model_data:
        info = model_data['model_info']
        print(f"Total keywords: {info.get('total_keywords', 'Unknown')}")
        print(f"Total questions: {info.get('total_questions', 'Unknown')}")
        print(f"Categories: {info.get('categories', [])}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

print("\n✅ All tests passed! Model is ready to use.")