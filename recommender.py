import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

current_folder = Path(__file__).parent
file_path = current_folder / 'u.item'

df = pd.read_csv(file_path, sep='|', header=None, encoding='ISO-8859-1')


genre_list = [
    'unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]

def combine_genres(row):
    
    return " ".join([genre_list[i] for i, val in enumerate(row[5:24]) if val == 1])


df['genre_str'] = df.apply(combine_genres, axis=1)
df['title'] = df[1].fillna('Unknown')
df['tags'] = df['title'].astype(str) + " " + df['genre_str']

df = df[[0, 1, 'tags']]
df.columns = ['movie_id', 'title', 'tags']


tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['tags'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def get_recommendations(title, n=10):
    if title not in indices:
        return f"Movie '{title}' not found in the dataset."
    
  
    idx = indices[title]

    
    sim_scores = list(enumerate(cosine_sim[idx]))

    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    
    sim_scores = sim_scores[1:n+1]

    
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]





def mood_recommendations(mood, n=8):
    
    mood_map = {
        'inspired': 'Adventure Animation Fantasy Sci-Fi',
        'melancholy': 'Drama Film-Noir Documentary',
        'hyped': 'Action Thriller War Western',
        'cozy': 'Comedy Childrens Musical Romance',
        'tense': 'Horror Mystery Thriller Crime'
    }

    
    if mood.lower() not in mood_map:
        return f"Mood '{mood}' not recognized. Try: {', '.join(mood_map.keys())}"

    
    mood_query = mood_map[mood.lower()]

  
    mood_vector = tfidf.transform([mood_query])

 
    mood_sim = cosine_similarity(mood_vector, tfidf_matrix)

    
    sim_scores = list(enumerate(mood_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    
    top_indices = [i[0] for i in sim_scores[:n]]
    return df['title'].iloc[top_indices]


if __name__ == "__main__":
    movie_to_search = 'Toy Story (1995)'
    print(f"--- Top 10 Recommendations for '{movie_to_search}' ---")
    results = get_recommendations(movie_to_search, n=10)
    
    if isinstance(results, str):
        print(results)
    else:
        for i, title in enumerate(results, 1):
            print(f"{i}. {title}")

    print(mood_recommendations('cozy', n=8))        
