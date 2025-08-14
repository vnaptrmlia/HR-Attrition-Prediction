# test_integration.py
import os
import joblib
import pickle

def test_files():
    """Test apakah semua files bisa di-load"""
    
    print("🧪 TESTING FILE INTEGRATION...")
    print("="*40)
    
    required_files = [
        'models/logistic_regression_model.pkl',
        'models/scaler.pkl',
        'models/feature_names.pkl',
        'models/model_metadata.pkl',
        'models/feature_mapping.pkl'
    ]
    
    success_count = 0
    
    for file_path in required_files:
        try:
            if file_path.endswith('.pkl') and 'model' in file_path:
                # Test model loading
                model = joblib.load(file_path)
                print(f"✅ {file_path} - Model: {type(model).__name__}")
                
            elif file_path.endswith('.pkl') and 'scaler' in file_path:
                # Test scaler loading
                scaler = joblib.load(file_path)
                print(f"✅ {file_path} - Scaler: {type(scaler).__name__}")
                
            else:
                # Test pickle files
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, list):
                        print(f"✅ {file_path} - List with {len(data)} items")
                    elif isinstance(data, dict):
                        print(f"✅ {file_path} - Dict with {len(data)} keys")
                    else:
                        print(f"✅ {file_path} - {type(data).__name__}")
            
            success_count += 1
            
        except Exception as e:
            print(f"❌ {file_path} - Error: {e}")
    
    print(f"\n📊 RESULT: {success_count}/{len(required_files)} files loaded successfully")
    
    if success_count == len(required_files):
        print("🎉 ALL FILES READY! You can run Streamlit app")
        return True
    else:
        print("⚠️ Some files have issues. Check the errors above.")
        return False

if __name__ == "__main__":
    test_files()