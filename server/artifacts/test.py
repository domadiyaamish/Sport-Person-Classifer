import pickle
import joblib

with open('saved_model.pkl', 'rb') as f:
    __model = joblib.load(f)