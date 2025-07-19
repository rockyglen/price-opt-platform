import pandas as pd
import numpy as np
import random

# 1. Define product categories with base price ranges and typical elasticity
CATEGORY_CONFIG = {
    'electronics': {'price_range': (800, 2000), 'elasticity': (0.6, 0.9)},
    'books': {'price_range': (200, 600), 'elasticity': (0.3, 0.6)},
    'clothing': {'price_range': (400, 1200), 'elasticity': (0.5, 0.8)},
    'home': {'price_range': (500, 1500), 'elasticity': (0.4, 0.7)},
    'beauty': {'price_range': (100, 400), 'elasticity': (0.2, 0.5)}
}

# 2. Function to generate N products
def generate_product_catalog(n_products=100, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    data = []

    for i in range(n_products):
        product_id = f"P{i+1:03d}"
        category = random.choice(list(CATEGORY_CONFIG.keys()))
        price_low, price_high = CATEGORY_CONFIG[category]['price_range']
        elasticity_low, elasticity_high = CATEGORY_CONFIG[category]['elasticity']

        base_price = round(np.random.uniform(price_low, price_high), 2)
        elasticity_score = round(np.random.uniform(elasticity_low, elasticity_high), 2)

        data.append([product_id, category, base_price, elasticity_score])

    df = pd.DataFrame(data, columns=['product_id', 'category', 'base_price', 'elasticity_score'])
    return df

# 3. Save the catalog to CSV
if __name__ == '__main__':
    df = generate_product_catalog(n_products=100)
    df.to_csv('data/raw/products.csv', index=False)
    print("✅ Product catalog created. Rows:", len(df))
