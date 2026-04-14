import pandas as pd
from sklearn.cluster import KMeans

class TribeEngine:
    def __init__(self, ratings_df, movies_df):
        self.ratings = ratings_df
        self.movies = movies_df
        # Build User-Item Matrix
        self.matrix = self.ratings.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)
        self.kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        self.clusters = self.kmeans.fit_predict(self.matrix)
        self.matrix['tribe'] = self.clusters

    def get_tribe_recs(self, liked_titles, n=5):
        # Find movie IDs for the titles the user likes
        liked_ids = self.movies[self.movies['title'].isin(liked_titles)]['movie_id'].tolist()
        if not liked_ids: return []

        # Create a "pseudo-user" profile based on liked movies
        user_profile = pd.Series(0, index=self.matrix.columns[:-1]) # -1 to exclude 'tribe' col
        user_profile.loc[liked_ids] = 5 # Assume high rating for liked movies
        
        # Predict which tribe this user belongs to
        tribe_id = self.kmeans.predict([user_profile])[0]
        
        # Get top movies from that tribe that aren't the ones they already liked
        tribe_users = self.matrix[self.matrix['tribe'] == tribe_id].index
        tribe_ratings = self.ratings[self.ratings['user_id'].isin(tribe_users)]
        
        top_movies = (tribe_ratings[~tribe_ratings['movie_id'].isin(liked_ids)]
                      .groupby('movie_id')['rating'].mean()
                      .sort_values(ascending=False).head(n).index)
        
        return self.movies[self.movies['movie_id'].isin(top_movies)]['title'].tolist()
