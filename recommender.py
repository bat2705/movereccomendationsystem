import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentEngine:
    def __init__(self, movies_df):
        self.df = movies_df
        # Create tags if they don't exist in the raw file
        if 'tags' not in self.df.columns:
            self.df['tags'] = self.df['title'].fillna('')
        
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['tags'])
        self.indices = pd.Series(self.df.index, index=self.df['title']).drop_duplicates()

    def get_mood_recs(self, mood_query, n=5):
        query_vec = self.vectorizer.transform([mood_query])
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = sim_scores.argsort()[::-1][:n]
        return self.df['title'].iloc[top_indices].tolist()

    def get_anti_recs(self, disliked_titles, n=5):
        bad_indices = [self.indices[t] for t in disliked_titles if t in self.indices]
        if not bad_indices: return []
        # Mean similarity to all disliked movies
        mean_sim = cosine_similarity(self.tfidf_matrix[bad_indices], self.tfidf_matrix).mean(axis=0)
        # Sort by LEAST similar
        least_sim_indices = mean_sim.argsort()[:n]
        return self.df['title'].iloc[least_sim_indices].tolist()
