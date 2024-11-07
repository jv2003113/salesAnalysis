import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

def train_model(data):
    data['total_sales'] = data['quantity_sold'] * data['price_per_item']
    X = data[['hour', 'day_of_week']]
    y = data['total_sales']
    
    # Convert categorical variable to dummy/indicator variables
    X = pd.get_dummies(X, columns=['day_of_week'], drop_first=False)
    
    # Reorder columns to ensure day of week is from Sunday to Saturday
    day_of_week_order = ['day_of_week_Sunday', 'day_of_week_Monday', 
                         'day_of_week_Tuesday', 'day_of_week_Wednesday', 
                         'day_of_week_Thursday', 'day_of_week_Friday', 
                         'day_of_week_Saturday']
    
    # Ensure the columns are in the correct order
    X = X[day_of_week_order + ['hour']]  # Add 'hour' at the end

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    
    return model

def serialize_model(model, filename='sales_model.pkl'):
    joblib.dump(model, filename)

if __name__ == "__main__":
    data = pd.read_csv('test_data.csv')
    model = train_model(data)
    serialize_model(model)
    print("Model trained and serialized to 'sales_model.pkl'.")
