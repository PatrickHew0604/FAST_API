from fastapi import FastAPI, HTTPException
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = FastAPI()

# Load the trained model and preprocessing objects
MODEL_PATH = "customer_pref_model.h5"
LABEL_ENCODER_PATH = "label_encoder_classes.npy"
SCALER_PARAMS_PATH = "scaler_params.npy"

try:
    model = load_model(MODEL_PATH)
    label_encoder_classes = np.load(LABEL_ENCODER_PATH, allow_pickle=True)
    scaler_params = np.load(SCALER_PARAMS_PATH, allow_pickle=True)
    
    label_encoder = LabelEncoder()
    label_encoder.classes_ = label_encoder_classes
    
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_ = scaler_params
except Exception as e:
    raise RuntimeError(f"Error loading model or preprocessing objects: {e}")

@app.get("/")
def home():
    return {"message": "Customer Preference Prediction API is running!"}

@app.post("/predict/")
def predict_category(data: dict):
    try:
        print(f"Received data: {data}")  # Debugging line

        # Ensure "purchase_history" exists
        if "purchase_history" not in data:
            raise HTTPException(status_code=400, detail="Missing 'purchase_history' in request data")

        # Extract numerical values
        purchase_history = data["purchase_history"]
        feature_values = list(purchase_history.values())  # Convert dictionary values to a list
        
        print(f"Extracted features: {feature_values}")  # Debugging line

        # Convert to NumPy array and reshape
        features = np.array(feature_values).reshape(1, -1)
        
        print(f"Reshaped features: {features}")  # Debugging line

        # Check if scaler is working
        features_scaled = scaler.transform(features)
        
        print(f"Scaled features: {features_scaled}")  # Debugging line

        # Make prediction
        prediction = model.predict(features_scaled)
        predicted_class = np.argmax(prediction)
        predicted_category = label_encoder.inverse_transform([predicted_class])[0]

        return {"predicted_category": predicted_category}
    except Exception as e:
        print(f"Error: {e}")  # Debugging line
        raise HTTPException(status_code=400, detail=str(e))

# Run FastAPI server if executed as main script
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)