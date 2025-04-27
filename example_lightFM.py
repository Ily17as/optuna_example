from lightfm import LightFM
from lightfm.datasets import fetch_movielens

# Load a sample dataset
data = fetch_movielens(min_rating=4.0)

# Instantiate a model
model = LightFM(loss='warp')  # WARP loss = good for ranking

# Train the model
model.fit(data['train'], epochs=30, num_threads=2)

# Predict scores for a user
import numpy as np
scores = model.predict(3, np.arange(data['train'].shape[1]))
top_items = np.argsort(-scores)  # Recommend top scores
print("Top recommendations for user 3:", top_items[:5])
