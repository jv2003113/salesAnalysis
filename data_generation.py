import pandas as pd
import numpy as np

def generate_test_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'menu_item': np.random.choice(['Burger', 'Pizza', 'Salad', 'Soda'], num_samples),
        'quantity_sold': np.random.randint(1, 20, num_samples),
        'price_per_item': np.random.choice([5.99, 8.99, 4.99, 1.99], num_samples),
        'hour': np.random.randint(10, 22, num_samples),  # 10 AM to 10 PM
        'day_of_week': np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], num_samples)
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    test_data = generate_test_data()
    test_data.to_csv('test_data.csv', index=False)
    print("Test data generated and saved to 'test_data.csv'.")
