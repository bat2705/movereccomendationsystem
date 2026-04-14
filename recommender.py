import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentEngine:
    def __init__(self, movies_df):
        self.df = movies_df
        
        # Use genre if tags don't exist
        if 'tags' not in self.df.columns:
            self.df['tags'] = self.df['genre'].fillna('')
        
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['tags'])
        self.indices = pd.Series(self.df.index, index=self.df['title']).drop_duplicates()
        
        # ✨ NEW: Mood to genre mapping
        self.mood_map = {
            'happy': 'comedy uplifting cheerful',
            'sad': 'drama emotional',
            'exciting': 'action thriller adventure',
            'scary': 'horror suspense',
            'relaxing': 'documentary calm peaceful',
            'dark': 'noir crime thriller',
            'funny': 'comedy humorous',
            'romantic': 'romance love',
            'intense': 'thriller drama action'
        }
    
    def expand_mood_query(self, mood_query):
        """Expand mood words to related genre/descriptive words"""
        words = mood_query.lower().split()
        expanded = []
        
        for word in words:
            if word in self.mood_map:
                # Add both original word and mapped genres
                expanded.append(word)
                expanded.append(self.mood_map[word])
            else:
                expanded.append(word)
        
        return ' '.join(expanded)
    
    def get_mood_recs(self, mood_query, n=5):
        # ✨ Expand mood query before transformation
        expanded_query = self.expand_mood_query(mood_query)
        
        query_vec = self.vectorizer.transform([expanded_query])
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = sim_scores.argsort()[::-1][:n]
        return self.df['title'].iloc[top_indices].tolist()
    
    def get_anti_recs(self, disliked_titles, n=5):
        bad_indices = [self.indices[t] for t in disliked_titles if t in self.indices]
        if not bad_indices: 
            return []
        
        mean_sim = cosine_similarity(
            self.tfidf_matrix[bad_indices], 
            self.tfidf_matrix
        ).mean(axis=0)
        
        least_sim_indices = mean_sim.argsort()[:n]
        return self.df['title'].iloc[least_sim_indices].tolist()
