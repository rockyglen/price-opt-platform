import pandas as pd
import numpy as np
import random

CUSTOMER_SEGMENTS = ['deal-seeker', 'premium', 'loyal', 'new', 'impulsive']
CATEGORIES = ['electronics', 'books', 'clothing', 'home', 'beauty']

# 1. Generate customer table
def generate_customers(n_customers=1000, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    data = []

    for i in range(n_customers):
        customer_id = f"C{i+1:03d}"
        segment = random.choice(CUSTOMER_SEGMENTS)
        preferred_category = random.choice(CATEGORIES)

        # Assign price sensitivity based on segment
        if segment == 'deal-seeker':
            price_sensitivity = np.random.uniform(0.8, 1.0)
        elif segment == 'premium':
            price_sensitivity = np.random.uniform(0.1, 0.3)
        elif segment == 'loyal':
            price_sensitivity = np.random.uniform(0.4, 0.6)
        elif segment == 'impulsive':
            price_sensitivity = np.random.uniform(0.5, 0.7)
        else:  # 'new'
            price_sensitivity = np.random.uniform(0.5, 0.8)

        price_sensitivity = round(price_sensitivity, 2)
        data.append([customer_id, segment, preferred_category, price_sensitivity])

    df = pd.DataFrame(data, columns=['customer_id', 'segment', 'preferred_category', 'price_sensitivity'])
    return df

# 2. Save as CSV
if __name__ == '__main__':
    df = generate_customers(n_customers=1000)
    df.to_csv('data/raw/customers.csv', index=False)
    print("✅ Customer profiles created. Rows:", len(df))
