import implicit
import scipy.sparse as sparse
import numpy as np

# Sample user-item interaction matrix
user_item_matrix = sparse.csr_matrix([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [1, 1, 0, 0]
])

# Initialize model
model = implicit.als.AlternatingLeastSquares(factors=10, regularization=0.1, iterations=20)

# Train model
# IMPORTANT: Implicit expects confidence scores, not just counts → we weight the interactions
model.fit(user_item_matrix * 10)

# Recommend items for user 0
recommended = model.recommend(0, user_item_matrix)
print(recommended)
