import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class CollaborativeFilteringRecommender:
    """Collaborative Filtering Recommendation System"""
    
    def __init__(self, metric='cosine', algorithm='user_based'):
        self.metric = metric
        self.algorithm = algorithm
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.user_means = None
    
    def fit(self, user_item_matrix):
        """Fit the model with user-item matrix"""
        self.user_item_matrix = user_item_matrix.copy()
        self._compute_similarity_matrix()
        self._compute_user_means()
    
    def _compute_similarity_matrix(self):
        """Compute similarity matrix"""
        if self.algorithm == 'user_based':
            # User-to-user similarity
            self.similarity_matrix = cosine_similarity(self.user_item_matrix)
            np.fill_diagonal(self.similarity_matrix, 0)
        else:
            # Item-to-item similarity
            self.similarity_matrix = cosine_similarity(self.user_item_matrix.T)
            np.fill_diagonal(self.similarity_matrix, 0)
    
    def _compute_user_means(self):
        """Compute mean rating per user (for normalization)"""
        self.user_means = np.array([
            np.mean(self.user_item_matrix[i, self.user_item_matrix[i] != 0])
            for i in range(self.user_item_matrix.shape[0])
        ])
        self.user_means = np.nan_to_num(self.user_means)
    
    def predict_rating(self, user_id, item_id, k=5):
        """Predict rating for a user-item pair"""
        if self.algorithm == 'user_based':
            return self._predict_user_based(user_id, item_id, k)
        else:
            return self._predict_item_based(user_id, item_id, k)
    
    def _predict_user_based(self, user_id, item_id, k=5):
        """User-based collaborative filtering prediction"""
        # Find k similar users who rated this item
        similarities = self.similarity_matrix[user_id]
        top_k_users = np.argsort(similarities)[-k:][::-1]
        
        # Get ratings from similar users
        ratings = []
        sim_scores = []
        for u in top_k_users:
            if self.user_item_matrix[u, item_id] != 0:
                ratings.append(self.user_item_matrix[u, item_id])
                sim_scores.append(similarities[u])
        
        if len(ratings) == 0:
            return self.user_means[user_id]
        
        # Weighted average
        return np.sum(np.array(ratings) * np.array(sim_scores)) / np.sum(sim_scores)
    
    def _predict_item_based(self, user_id, item_id, k=5):
        """Item-based collaborative filtering prediction"""
        # Find k similar items rated by this user
        similarities = self.similarity_matrix[item_id]
        rated_items = np.where(self.user_item_matrix[user_id] != 0)[0]
        
        # Get similarities with rated items
        item_sims = similarities[rated_items]
        top_k_items = rated_items[np.argsort(item_sims)[-k:][::-1]]
        
        ratings = self.user_item_matrix[user_id, top_k_items]
        sims = similarities[top_k_items]
        
        return np.sum(ratings * sims) / np.sum(sims)
    
    def recommend(self, user_id, n_recommendations=5, exclude_rated=True):
        """Get recommendations for a user"""
        predictions = []
        
        if exclude_rated:
            rated_items = np.where(self.user_item_matrix[user_id] != 0)[0]
            candidate_items = np.setdiff1d(np.arange(self.user_item_matrix.shape[1]), rated_items)
        else:
            candidate_items = np.arange(self.user_item_matrix.shape[1])
        
        for item in candidate_items:
            pred = self.predict_rating(user_id, item)
            predictions.append((item, pred))
        
        recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:n_recommendations]
        return recommendations

class MatrixFactorization:
    """Matrix Factorization using SVD"""
    
    def __init__(self, n_factors=20, learning_rate=0.01, n_epochs=100):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.user_features = None
        self.item_features = None
    
    def fit(self, user_item_matrix):
        """Fit the model"""
        self.user_item_matrix = user_item_matrix.copy()
        U, s, Vt = np.linalg.svd(self.user_item_matrix, full_matrices=False)
        
        # Keep top n_factors
        self.user_features = U[:, :self.n_factors]
        self.item_features = Vt[:self.n_factors, :].T
    
    def predict(self, user_id, item_id):
        """Predict rating"""
        return np.dot(self.user_features[user_id], self.item_features[item_id])
    
    def predict_all(self):
        """Predict all ratings"""
        return np.dot(self.user_features, self.item_features.T)

# Usage Example
if __name__ == "__main__":
    # Create sample user-item matrix (users x items ratings)
    np.random.seed(42)
    user_item_matrix = np.random.randint(0, 6, size=(50, 30)).astype(float)
    user_item_matrix[user_item_matrix == 0] = 0  # Keep some zeros as unrated
    
    # User-based Collaborative Filtering
    print("User-Based Collaborative Filtering:")
    cf_user = CollaborativeFilteringRecommender(algorithm='user_based')
    cf_user.fit(user_item_matrix)
    recommendations = cf_user.recommend(user_id=0, n_recommendations=5)
    print(f"Top 5 recommendations for user 0: {recommendations}")
    
    # Item-based Collaborative Filtering
    print("\nItem-Based Collaborative Filtering:")
    cf_item = CollaborativeFilteringRecommender(algorithm='item_based')
    cf_item.fit(user_item_matrix)
    recommendations = cf_item.recommend(user_id=0, n_recommendations=5)
    print(f"Top 5 recommendations for user 0: {recommendations}")
    
    # Matrix Factorization
    print("\nMatrix Factorization (SVD):")
    mf = MatrixFactorization(n_factors=10)
    mf.fit(user_item_matrix)
    
    # Get predictions for user 0
    pred = mf.predict_all()[0]
    top_items = np.argsort(pred)[-5:][::-1]
    print(f"Top 5 predicted items for user 0: {top_items}")
    print(f"Predicted ratings: {pred[top_items]}")
