import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# ── 1. Load Data ──────────────────────────────────────────
# NOTE: Make sure the 'ml-100k' folder is in the same directory as this script
ratings = pd.read_csv('u.data', sep='\t',
                      names=['user_id','movie_id','rating','timestamp'])

movies = pd.read_csv('u.item', sep='|', encoding='latin-1',
                     names=['movie_id','title'] + [f'f{i}' for i in range(22)],
                     usecols=['movie_id','title'])

print("Data loaded successfully!")
print(f"Total ratings: {len(ratings)}")
print(f"Total movies: {len(movies)}")

# ── 2. Build User-Item Matrix ─────────────────────────────
matrix = ratings.pivot_table(index='user_id',
                             columns='movie_id',
                             values='rating')

# ── 3. Fill missing ratings with 0 ───────────────────────
matrix_filled = matrix.fillna(0)

# ── 4. Run K-Means Clustering ─────────────────────────────
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)

# Run clustering and add the 'tribe' column safely
clusters = kmeans.fit_predict(matrix_filled)
matrix_with_tribes = matrix_filled.copy()
matrix_with_tribes['tribe'] = clusters

print("K-Means done! 5 tribes created.")
print(matrix_with_tribes['tribe'].value_counts())

# ── 5. Recommendation Function ───────────────────────────
def get_tribe_recommendations(user_id, n=10):
    if user_id not in matrix_with_tribes.index:
        print("User not found!")
        return pd.DataFrame() # Return empty DataFrame on error

    # Find user's tribe
    user_tribe = matrix_with_tribes.loc[user_id, 'tribe']
    print(f"\nUser {user_id} belongs to Tribe {user_tribe}")

    # Get all users in same tribe
    tribe_users = matrix_with_tribes[matrix_with_tribes['tribe'] == user_tribe].index

    # Movies already seen by this user
    seen = ratings[ratings['user_id'] == user_id]['movie_id'].tolist()

    # Get top rated unseen movies from tribe
    tribe_ratings = ratings[
        (ratings['user_id'].isin(tribe_users)) &
        (~ratings['movie_id'].isin(seen))
    ]
    
    if tribe_ratings.empty:
        print("No new recommendations found for this tribe.")
        return pd.DataFrame()

    top_movies = (tribe_ratings.groupby('movie_id')['rating']
                               .mean()
                               .sort_values(ascending=False)
                               .head(n)
                               .index.tolist())

    # Return movie titles
    results = movies[movies['movie_id'].isin(top_movies)][['movie_id','title']]
    return results

# ── 6. Test it ────────────────────────────────────────────
recs = get_tribe_recommendations(user_id=1)
if not recs.empty:
    print("\nYour Tribe's Top Picks:")
    print(recs.to_string(index=False))
