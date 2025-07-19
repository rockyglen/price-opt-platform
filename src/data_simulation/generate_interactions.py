import pandas as pd
import numpy as np
import random

# Load product and customer tables
products = pd.read_csv('data/raw/products.csv')
customers = pd.read_csv('data/raw/customers.csv')

# Interaction params
TIMES_OF_DAY = ['morning', 'afternoon', 'evening', 'night']
DAYS_OF_WEEK = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
SEASONS = ['peak', 'off']

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def simulate_conversion(offered_price, base_price, competitor_price,
                        price_sensitivity, elasticity, segment):
    discount = (base_price - offered_price) / base_price
    competitor_gap = (competitor_price - offered_price) / base_price

    # score = weighted sum of all conversion influencers
    score = 5 * discount                     # more discount = better
    score += 2 * competitor_gap              # being cheaper than competitors helps
    score -= 4 * price_sensitivity * (offered_price / base_price)  # penalize high price if user is sensitive
    score += np.random.normal(0, 0.5)        # randomness

    if segment == 'deal-seeker':
        score += 1.5
    elif segment == 'premium':
        score -= 1.0

    prob = sigmoid(score)
    converted = np.random.binomial(1, prob)
    return converted

def simulate_interactions(n_rows=10000, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    data = []

    for i in range(n_rows):
        interaction_id = f"INT{i+1:05d}"

        customer = customers.sample(1).iloc[0]
        product = products.sample(1).iloc[0]

        offered_price = round(product['base_price'] * np.random.uniform(0.7, 1.05), 2)
        competitor_price = round(product['base_price'] * np.random.uniform(0.7, 1.2), 2)

        time_of_day = random.choice(TIMES_OF_DAY)
        day_of_week = random.choice(DAYS_OF_WEEK)
        season = random.choice(SEASONS)

        converted = simulate_conversion(
            offered_price,
            product['base_price'],
            competitor_price,
            customer['price_sensitivity'],
            product['elasticity_score'],
            customer['segment']
        )

        data.append([
            interaction_id,
            customer['customer_id'],
            product['product_id'],
            round(product['base_price'], 2),
            offered_price,
            competitor_price,
            product['category'],
            customer['segment'],
            time_of_day,
            day_of_week,
            season,
            converted
        ])

    columns = [
        'interaction_id', 'customer_id', 'product_id', 'base_price',
        'offered_price', 'competitor_price', 'category', 'segment',
        'time_of_day', 'day_of_week', 'season', 'converted'
    ]
    df = pd.DataFrame(data, columns=columns)
    return df

if __name__ == '__main__':
    df = simulate_interactions(n_rows=10000)
    df.to_csv('data/raw/interactions.csv', index=False)
    print("✅ Interactions generated. Rows:", len(df))

