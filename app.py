import streamlit as st
import pandas as pd
from pathlib import Path
from recommender import ContentEngine
from clustering import TribeEngine

# --- PAGE CONFIG ---
st.set_page_config(page_title="CineMatch Pro", page_icon="🎬", layout="wide")

# --- CUSTOM UI STYLING ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .movie-card {
        background-color: #262730;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 5px solid #e8c84a;
    }
    .movie-title { color: #ffffff; font-size: 1.2rem; font-weight: bold; }
    .genre-tag {
        background-color: #3e3f4b;
        color: #e8c84a;
        padding: 2px 8px;
        border-radius: 5px;
        font-size: 0.8rem;
        margin-right: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_engines():
    curr = Path(__file__).parent
    
    # Load Ratings
    rat = pd.read_csv(curr / 'u.data', sep='\t', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])
    
    # Load Movies with all genre columns
    genre_cols = ['unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 
                  'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
                  'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    
    mov = pd.read_csv(curr / 'u.item', sep='|', header=None, encoding='latin-1')
    
    # Clean up the movie dataframe
    movie_info = mov[[0, 1]].copy()
    movie_info.columns = ['movie_id', 'title']
    
    # Create a "Genre String" for display and for the recommendation engine
    def extract_genres(row):
        return [genre_cols[i] for i, val in enumerate(row[5:24]) if val == 1]

    movie_info['genre_list'] = mov.apply(extract_genres, axis=1)
    movie_info['tags'] = movie_info['genre_list'].apply(lambda x: " ".join(x))
    
    return ContentEngine(movie_info), TribeEngine(rat, movie_info), movie_info

content_en, tribe_en, full_movie_df = load_engines()

# --- HELPER FUNCTION FOR BEAUTIFUL DISPLAY ---
def display_movie_results(title_list):
    if not title_list:
        st.write("No movies found.")
        return
    
    for title in title_list:
        # Get genre info for this specific movie
        row = full_movie_df[full_movie_df['title'] == title].iloc[0]
        genres = row['genre_list']
        
        # Render HTML Card
        genre_html = "".join([f'<span class="genre-tag">{g}</span>' for g in genres])
        st.markdown(f"""
            <div class="movie-card">
                <div class="movie-title">{title}</div>
                <div style="margin-top:8px;">{genre_html}</div>
            </div>
        """, unsafe_allow_html=True)

# --- APP LAYOUT ---
st.title("🎬 CINEMATCH PRO")
st.caption("AI-Powered Movie Discovery Engine")

tab1, tab2, tab3 = st.tabs(["🎭 MOOD SEARCH", "👥 TRIBE RECS", "🚫 ANTI-REC"])

with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("What's the vibe?")
        mood = st.text_input("e.g. 'Epic Space Adventure' or 'Romantic Comedy'", placeholder="Type here...")
    with col2:
        if mood:
            st.subheader("Recommended for your mood")
            recs = content_en.get_mood_recs(mood, n=6)
            display_movie_results(recs)

with tab2:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Your Favorites")
        liked = st.multiselect("Pick movies you love:", full_movie_df['title'].sort_values().tolist())
    with col2:
        if st.button("Generate Tribe Picks") and liked:
            recs = tribe_en.get_tribe_recs(liked, n=6)
            display_movie_results(recs)

with tab3:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("The Blacklist")
        disliked = st.multiselect("Pick movies you hate:", full_movie_df['title'].sort_values().tolist())
    with col2:
        if st.button("Show the Opposite") and disliked:
            recs = content_en.get_anti_recs(disliked, n=6)
            display_movie_results(recs)
