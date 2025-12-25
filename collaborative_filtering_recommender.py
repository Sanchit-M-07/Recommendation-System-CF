import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds


class CollaborativeFilteringRecommender:
    """
    Collaborative filtering recommender system.

    This class implements:
    - User-based collaborative filtering using user–user similarity.
    - Item-based collaborative filtering using item–item similarity.
    - Matrix factorization using truncated SVD for latent factors.

    Methods
    -------
    fit(ratings_df, user_col, item_col, rating_col):
        Fit the model on a user–item rating DataFrame.
    predict(user_id, item_id):
        Predict a rating for a given (user, item) pair.
    recommend(user_id, n=10, filter_seen=True):
        Generate top-N item recommendations for a given user.
    """

    """
    Supports:
    - User-based CF
    - Item-based CF
    - Matrix Factorization via SVD
    """

    def __init__(self, k_neighbors=20, k_factors=20, cf_type="user", svd_reg=0.0):
        """
        k_neighbors: number of neighbors for user/item CF
        k_factors: number of latent factors for SVD
        cf_type: "user", "item", or "svd"
        svd_reg: simple L2 regularization on reconstructed scores (optional, small effect here)
        """
        assert cf_type in ["user", "item", "svd"]
        self.k_neighbors = k_neighbors
        self.k_factors = k_factors
        self.cf_type = cf_type
        self.svd_reg = svd_reg

        # Learned attributes
        self.user_index = None
        self.item_index = None
        self.R = None              # user-item rating matrix (users x items)
        self.user_sim = None
        self.item_sim = None
        self.R_hat = None          # predicted rating matrix
        self.user_means = None


    def _build_matrix(self, ratings_df, user_col, item_col, rating_col):
        """
        Build user-item matrix R and index mappings.
        ratings_df: DataFrame[user_col, item_col, rating_col]
        """
        users = ratings_df[user_col].unique()
        items = ratings_df[item_col].unique()

        self.user_index = {u: i for i, u in enumerate(users)}
        self.item_index = {i: j for j, i in enumerate(items)}

        n_users = len(users)
        n_items = len(items)
        R = np.zeros((n_users, n_items), dtype=np.float32)

        for _, row in ratings_df.iterrows():
            u = self.user_index[row[user_col]]
            it = self.item_index[row[item_col]]
            R[u, it] = row[rating_col]

        self.R = R


    def fit(self, ratings_df, user_col="user_id", item_col="item_id", rating_col="rating"):
        """
        Fit the model:
        - Builds rating matrix
        - Computes similarity or SVD depending on cf_type
        """
        self._build_matrix(ratings_df, user_col, item_col, rating_col)

        if self.cf_type in ["user", "item"]:
            # Mean-center for user-based CF to reduce user rating bias. [web:1][web:5]
            self.user_means = np.where(self.R.sum(axis=1) != 0,
                                       np.true_divide(self.R.sum(axis=1), (self.R != 0).sum(axis=1)),
                                       0.0)
            R_centered = self.R.copy()
            for u in range(self.R.shape[0]):
                nonzero = self.R[u, :] > 0
                R_centered[u, nonzero] -= self.user_means[u]

            if self.cf_type == "user":
                self.user_sim = cosine_similarity(R_centered)
            else:  # item-based
                self.item_sim = cosine_similarity(self.R.T)

        elif self.cf_type == "svd":
            # Simple SVD-based matrix factorization on mean-centered ratings. [web:7]
            # Replace missing with 0 (implicit 0 = no rating).
            R = self.R.copy()
            # Compute user means only on non-zero entries
            self.user_means = np.where(R.sum(axis=1) != 0,
                                       np.true_divide(R.sum(axis=1), (R != 0).sum(axis=1)),
                                       0.0)
            R_demeaned = R.copy()
            for u in range(R.shape[0]):
                nonzero = R[u, :] > 0
                R_demeaned[u, nonzero] -= self.user_means[u]

            # SVD
            k = min(self.k_factors, min(R_demeaned.shape) - 1)
            U, s, Vt = svds(R_demeaned, k=k)
            S = np.diag(s)

            # Reconstruct approximate ratings matrix
            R_hat_centered = np.dot(np.dot(U, S), Vt)
            self.R_hat = R_hat_centered + self.user_means.reshape(-1, 1)

            if self.svd_reg > 0:
                self.R_hat = self.R_hat / (1.0 + self.svd_reg)


    def _predict_user_based(self, u_idx, i_idx):
        """
        Predict rating for a given (user_idx, item_idx) using user-based CF. [web:1][web:5]
        """
        if self.user_sim is None:
            raise ValueError("Model not fitted for user-based CF")

        # Users who rated item i
        item_ratings = self.R[:, i_idx]
        rated_by = np.where(item_ratings > 0)[0]
        if len(rated_by) == 0:
            # No neighbors, return user mean or global mean
            if self.user_means is not None:
                return float(self.user_means[u_idx])
            return float(np.mean(self.R[self.R > 0])) if np.any(self.R > 0) else 0.0

        sims = self.user_sim[u_idx, rated_by]

        # Take top-k neighbors by similarity
        if len(rated_by) > self.k_neighbors:
            top_k_idx = np.argsort(sims)[-self.k_neighbors:]
            rated_by = rated_by[top_k_idx]
            sims = sims[top_k_idx]

        # Remove zero or negative similarity if desired
        mask = sims > 0
        if not mask.any():
            return float(self.user_means[u_idx]) if self.user_means is not None else 0.0

        sims = sims[mask]
        neigh_ratings = self.R[rated_by[mask], i_idx]

        if np.sum(sims) == 0:
            return float(self.user_means[u_idx])

        pred = np.dot(sims, neigh_ratings) / np.sum(sims)
        return float(pred)


    def _predict_item_based(self, u_idx, i_idx):
        """
        Predict rating for a given (user_idx, item_idx) using item-based CF. [web:6]
        """
        if self.item_sim is None:
            raise ValueError("Model not fitted for item-based CF")

        user_ratings = self.R[u_idx, :]
        rated_items = np.where(user_ratings > 0)[0]
        if len(rated_items) == 0:
            # No rating history, fall back to user mean or global mean
            if self.user_means is not None:
                return float(self.user_means[u_idx])
            return float(np.mean(self.R[self.R > 0])) if np.any(self.R > 0) else 0.0

        sims = self.item_sim[i_idx, rated_items]

        if len(rated_items) > self.k_neighbors:
            top_k_idx = np.argsort(sims)[-self.k_neighbors:]
            rated_items = rated_items[top_k_idx]
            sims = sims[top_k_idx]

        mask = sims > 0
        if not mask.any():
            return float(self.user_means[u_idx]) if self.user_means is not None else 0.0

        sims = sims[mask]
        neigh_ratings = user_ratings[rated_items[mask]]

        if np.sum(sims) == 0:
            return float(self.user_means[u_idx])

        pred = np.dot(sims, neigh_ratings) / np.sum(sims)
        return float(pred)


    def _predict_svd(self, u_idx, i_idx):
        """
        Predict rating from SVD reconstructed matrix. [web:7]
        """
        if self.R_hat is None:
            raise ValueError("Model not fitted with SVD")
        return float(self.R_hat[u_idx, i_idx])


    def predict(self, user_id, item_id):
        """
        Predict rating for (user_id, item_id) after fit().
        """
        if user_id not in self.user_index or item_id not in self.item_index:
            # Simple cold-start handling
            # Return global mean if any ratings exist
            if self.R is not None and np.any(self.R > 0):
                return float(np.mean(self.R[self.R > 0]))
            return 0.0

        u_idx = self.user_index[user_id]
        i_idx = self.item_index[item_id]

        if self.cf_type == "user":
            return self._predict_user_based(u_idx, i_idx)
        elif self.cf_type == "item":
            return self._predict_item_based(u_idx, i_idx)
        else:  # "svd"
            return self._predict_svd(u_idx, i_idx)


    def recommend(self, user_id, n=10, filter_seen=True):
        """
        Generate top-n item recommendations for a user.
        Returns list of (item_id, predicted_rating).
        """
        if user_id not in self.user_index:
            return []

        u_idx = self.user_index[user_id]
        n_items = self.R.shape[1]
        preds = []

        for i_idx in range(n_items):
            if filter_seen and self.R[u_idx, i_idx] > 0:
                continue
            pred_rating = self.predict(user_id, list(self.item_index.keys())[i_idx])
            preds.append((list(self.item_index.keys())[i_idx], pred_rating))

        preds.sort(key=lambda x: x[1], reverse=True)
        return preds[:n]


# -------------------------- Example usage --------------------------
if __name__ == "__main__":
    # Example ratings dataframe
    data = {
        "user_id": [1, 1, 1, 2, 2, 3, 3],
        "item_id": [10, 11, 12, 10, 12, 11, 13],
        "rating": [4, 5, 3, 5, 4, 2, 5]
    }
    ratings_df = pd.DataFrame(data)

    # User-based CF
    user_cf = CollaborativeFilteringRecommender(cf_type="user", k_neighbors=2)
    user_cf.fit(ratings_df)
    print("User-based predict:", user_cf.predict(1, 13))
    print("User-based recommend:", user_cf.recommend(1, n=3))

    # Item-based CF
    item_cf = CollaborativeFilteringRecommender(cf_type="item", k_neighbors=2)
    item_cf.fit(ratings_df)
    print("Item-based predict:", item_cf.predict(1, 13))
    print("Item-based recommend:", item_cf.recommend(1, n=3))

    # SVD-based MF
    svd_cf = CollaborativeFilteringRecommender(cf_type="svd", k_factors=2)
    svd_cf.fit(ratings_df)
    print("SVD predict:", svd_cf.predict(1, 13))
    print("SVD recommend:", svd_cf.recommend(1, n=3))
