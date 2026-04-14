import streamlit as st
import pandas as pd
from pathlib import Path
from recommender import ContentEngine
from clustering import TribeEngine

st.set_page_config(page_title="CineMatch", page_icon="🎬")

# --- DATA LOADING ---
@st.cache_resource
def load_engines():
    curr = Path(__file__).parent
    # Load raw MovieLens files
    rat = pd.read_csv(curr / 'u.data', sep='\t', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])
    mov = pd.read_csv(curr / 'u.item', sep='|', header=None, encoding='latin-1', usecols=[0,1], names=['movie_id', 'title'])
    
    return ContentEngine(mov), TribeEngine(rat, mov), mov['title'].tolist()

content_en, tribe_en, all_titles = load_engines()

st.title("🎬 CINEMATCH")

tab1, tab2, tab3 = st.tabs(["🎭 MOOD", "👥 TRIBE", "🚫 ANTI-REC"])

with tab1:
    mood = st.text_input("How are you feeling? (e.g., 'scary and dark' or 'fun adventure')")
    if mood:
        recs = content_en.get_mood_recs(mood)
        for r in recs: st.success(r)

with tab2:
    liked = st.multiselect("Movies you love:", all_titles)
    if st.button("Find Tribe Picks"):
        recs = tribe_en.get_tribe_recs(liked)
        for r in recs: st.info(r)

with tab3:
    disliked = st.multiselect("Movies you hate:", all_titles)
    if st.button("Show Opposites"):
        recs = content_en.get_anti_recs(disliked)
        for r in recs: st.warning(r)
