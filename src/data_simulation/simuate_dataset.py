# simulate_dataset.py

import numpy as np
import pandas as pd
import random
from datetime import datetime

CATEGORIES = ['electronics', 'clothing', 'books', 'home', 'beauty']
CUSTOMER_SEGMENTS = ['deal-seeker', 'loyal', 'new', 'premium']
DAYS_OF_WEEK = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
TIMES_OF_DAY = ['morning', 'afternoon', 'evening', 'night']


def conversion_probability(offered_price, base_price, customer_segment):
    """
    Simulates how likely a customer is to convert based on price and type.
    """
    discount = (base_price - offered_price) / base_price
    base_prob = 0.1 + 0.5 * discount  # 10% base + boost from discount

    if customer_segment == 'deal-seeker':
        base_prob += 0.2
    elif customer_segment == 'premium':
        base_prob -= 0.1

    return min(max(base_prob + np.random.normal(0, 0.05), 0), 1)


def generate_data(n_rows=10000, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    data = []
    for i in range(n_rows):
        product_id = f"P{i%100}"
        category = random.choice(CATEGORIES)
        base_price = round(np.random.uniform(100, 1000), 2)
        discount_percent = np.random.uniform(0, 0.5)
        offered_price = round(base_price * (1 - discount_percent), 2)
        inventory_level = random.choice(['low', 'medium', 'high'])
        customer_segment = random.choice(CUSTOMER_SEGMENTS)
        time_of_day = random.choice(TIMES_OF_DAY)
        day_of_week = random.choice(DAYS_OF_WEEK)
        season = random.choice(['peak', 'off'])

        prob = conversion_probability(offered_price, base_price, customer_segment)
        converted = np.random.binomial(1, prob)

        data.append([
            product_id, category, base_price, offered_price,
            inventory_level, customer_segment, time_of_day,
            day_of_week, season, converted
        ])
    
    columns = [
        'product_id', 'category', 'base_price', 'offered_price',
        'inventory_level', 'customer_segment', 'time_of_day',
        'day_of_week', 'season', 'converted'
    ]
    df = pd.DataFrame(data, columns=columns)
    return df


if __name__ == '__main__':
    df = generate_data(n_rows=10000)
    df.to_csv('data/raw/synthetic_pricing_data.csv', index=False)
    print("✅ Data generation complete. Rows:", len(df))
