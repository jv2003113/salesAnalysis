import pandas as pd
import joblib

def load_model(filename='sales_model.pkl'):
    return joblib.load(filename)

def make_prediction(model, sample_data):
    return model.predict(sample_data)

if __name__ == "__main__":
    loaded_model = load_model()
    
    # Sample input for prediction
    sample_data = pd.DataFrame({
        'day_of_week_Sunday': [0],
        'day_of_week_Monday': [0],
        'day_of_week_Tuesday': [1],
        'day_of_week_Wednesday': [0],
        'day_of_week_Thursday': [0],
        'day_of_week_Friday': [0],
        'day_of_week_Saturday': [0],
        'hour': [12]
        # Ensure the order matches the training data
    })
     
    prediction = make_prediction(loaded_model, sample_data)
    print(f"Predicted sales: {prediction}")
