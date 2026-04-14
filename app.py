

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #0d0d0d;
        color: #f0ede6;
    }
    h1, h2, h3 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 2px; }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.1rem;
        letter-spacing: 1.5px;
        color: #aaa;
        padding: 10px 24px;
    }
    .stTabs [aria-selected="true"] {
        color: #e8c84a !important;
        border-bottom: 2px solid #e8c84a !important;
    }
    .stButton > button {
        background: #e8c84a;
        color: #0d0d0d;
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1rem;
        letter-spacing: 1px;
        border: none;
        border-radius: 4px;
        padding: 10px 28px;
    }
    .stButton > button:hover { background: #f5d96b; }
    .movie-card {
        background: #1a1a1a;
        border-left: 3px solid #e8c84a;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 4px;
        font-size: 0.95rem;
    }
    .tribe-badge {
        display: inline-block;
        background: #e8c84a;
        color: #0d0d0d;
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.3rem;
        letter-spacing: 2px;
        padding: 6px 18px;
        border-radius: 4px;
        margin-bottom: 12px;
    }
    .section-label {
        color: #e8c84a;
        font-family: 'Bebas Neue', sans-serif;
        font-size: 0.85rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='font-size:3rem; margin-bottom:0'>🎬 CINEMATCH</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#888; margin-top:0; margin-bottom:24px'>Smart Movie Recommender — Mood · Tribe · Anti-Rec</p>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SHARED DATA & MODELS (used across all tabs)
# ─────────────────────────────────────────────

movies = pd.DataFrame({
    "title": [
        "Toy Story", "Aladdin", "Batman", "Joker", "Inception",
        "The Notebook", "Interstellar", "Parasite", "Get Out",
        "The Dark Knight", "La La Land", "Avengers", "Her",
        "Whiplash", "Gone Girl", "Spirited Away", "Coco",
        "Moonlight", "Mad Max: Fury Road", "The Shawshank Redemption"
    ],
    "tags": [
        "fun animated adventure friendship toys comedy",
        "magic adventure fun animated princess comedy",
        "dark superhero action crime brooding",
        "dark psychological crime villain chaos",
        "mind-bending thriller heist dream sci-fi",
        "romantic emotional love drama heartbreak",
        "space sci-fi emotional family time adventure",
        "dark thriller class social commentary suspense",
        "horror thriller race social anxiety paranoia",
        "dark superhero action crime psychological",
        "romantic musical dream ambition drama",
        "action superhero ensemble adventure fun",
        "romantic sci-fi loneliness emotional AI drama",
        "intense music ambition drama obsession",
        "dark thriller mystery psychological suspense",
        "fantasy animated adventure magical spiritual",
        "family animated emotional music adventure",
        "emotional coming-of-age drama identity quiet",
        "action intense survival post-apocalyptic chaos",
        "emotional drama hope friendship prison inspiring"
    ]
})

mood_map = {
    "😂 Something Fun & Light": "fun comedy adventure animated",
    "😢 Want to Cry": "emotional heartbreak sad drama love",
    "😱 Edge of My Seat": "thriller suspense dark psychological",
    "🤯 Mind-Bending": "sci-fi mind-bending thriller mystery",
    "❤️ Romantic Mood": "romantic love drama emotional",
    "💪 Feel Inspired": "inspiring hope ambition drama uplifting",
    "👻 Scared But Make It Fun": "horror thriller paranoia dark suspense",
    "🌌 Thoughtful & Deep": "philosophical emotional quiet drama identity"
}

tribe_names = {
    0: "The Thinkers",
    1: "The Thrill-Seekers",
    2: "The Romantics",
    3: "The Dark-Side Gang"
}

tribe_descriptions = {
    0: "You love slow-burn stories, complex characters, and movies that leave you thinking for days.",
    1: "You live for action, plot twists, and movies that don't let you breathe.",
    2: "You're here for the feelings — love, loss, and everything in between.",
    3: "You gravitate toward dark, gritty, morally complex cinema."
}

@st.cache_resource
def build_tfidf():
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(movies["tags"])
    return vectorizer, matrix

vectorizer, tfidf_matrix = build_tfidf()
movie_titles = movies["title"].tolist()

# ─────────────────────────────────────────────
# ANTI-REC LOGIC (merged from anti_rec.py)
# ─────────────────────────────────────────────

def anti_recommendations(disliked_titles, tfidf_matrix, movie_titles):
    # 1. Convert disliked titles to their indices
    bad_indices = [indices[t] for t in disliked_titles if t in indices]
    if not bad_indices: return []
    
    # 2. Get average similarity to all "hated" movies
    mean_sim = cosine_similarity(tfidf_matrix[bad_indices], tfidf_matrix).mean(axis=0)
    
    # 3. Sort from LEAST similar to MOST similar
    least_sim_indices = np.argsort(mean_sim)[:10]
    return movie_titles.iloc[least_sim_indices].tolist()

# ─────────────────────────────────────────────
# MOOD LOGIC
# ─────────────────────────────────────────────

def mood_recommendations(mood_query, matrix, titles, vectorizer, n=5):
    query_vec = vectorizer.transform([mood_query])
    similarities = cosine_similarity(query_vec, matrix).flatten()
    sorted_indices = similarities.argsort()[::-1]
    return [titles[i] for i in sorted_indices[:n]]

# ─────────────────────────────────────────────
# TRIBE LOGIC
# ─────────────────────────────────────────────

def get_tribe(movie_preferences, matrix, n_clusters=4):
    if not movie_preferences:
        return None, None
    indices = [movie_titles.index(m) for m in movie_preferences if m in movie_titles]
    if not indices:
        return None, None
    pref_vectors = matrix[indices].toarray()
    avg_pref = np.mean(pref_vectors, axis=0).reshape(1, -1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    all_vecs = matrix.toarray()
    kmeans.fit(all_vecs)
    tribe_id = kmeans.predict(avg_pref)[0]
    tribe_movies_idx = np.where(kmeans.labels_ == tribe_id)[0]
    tribe_movies = [movie_titles[i] for i in tribe_movies_idx if movie_titles[i] not in movie_preferences]
    return tribe_id, tribe_movies[:5]

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🎭  MOOD PICKS", "👥  YOUR TRIBE", "🚫  ANTI-REC"])

# ── TAB 1: MOOD PICKS ──────────────────────────
with tab1:
    st.markdown("<div class='section-label'>How are you feeling tonight?</div>", unsafe_allow_html=True)

    mood = st.selectbox("Pick your mood", list(mood_map.keys()), label_visibility="collapsed")

    if st.button("Find My Movies", key="mood_btn"):
        query = mood_map[mood]
        results = mood_recommendations(query, tfidf_matrix, movie_titles, vectorizer)
        st.markdown(f"<br><div class='section-label'>Top picks for '{mood}'</div>", unsafe_allow_html=True)
        for i, title in enumerate(results, 1):
            st.markdown(f"<div class='movie-card'>#{i} &nbsp; {title}</div>", unsafe_allow_html=True)

# ── TAB 2: TASTE TRIBE ─────────────────────────
with tab2:
    st.markdown("<div class='section-label'>Select movies you enjoy</div>", unsafe_allow_html=True)
    liked = st.multiselect("Pick 2–5 movies you like", movie_titles, label_visibility="collapsed")

    if st.button("Find My Tribe", key="tribe_btn"):
        if len(liked) < 2:
            st.warning("Pick at least 2 movies so we can find your tribe.")
        else:
            tribe_id, tribe_recs = get_tribe(liked, tfidf_matrix)
            if tribe_id is not None:
                st.markdown(f"<br><div class='tribe-badge'>Tribe: {tribe_names[tribe_id]}</div>", unsafe_allow_html=True)
                st.markdown(f"<p style='color:#aaa'>{tribe_descriptions[tribe_id]}</p>", unsafe_allow_html=True)
                if tribe_recs:
                    st.markdown("<div class='section-label'>Your tribe also loved</div>", unsafe_allow_html=True)
                    for i, title in enumerate(tribe_recs, 1):
                        st.markdown(f"<div class='movie-card'>#{i} &nbsp; {title}</div>", unsafe_allow_html=True)
            else:
                st.error("Couldn't determine your tribe. Try selecting different movies.")

# ── TAB 3: ANTI-REC ────────────────────────────
with tab3:
    st.markdown("<div class='section-label'>Select movies you disliked</div>", unsafe_allow_html=True)
    disliked = st.multiselect("Pick movies you hated", movie_titles, label_visibility="collapsed")

    if st.button("Get Anti-Recommendations", key="anti_btn"):
        if not disliked:
            st.warning("Select at least one movie you disliked.")
        else:
            results = anti_recommendations(disliked, tfidf_matrix, movie_titles)
            st.markdown("<br><div class='section-label'>Based on what you hate, try these instead</div>", unsafe_allow_html=True)
            for i, title in enumerate(results, 1):
                st.markdown(f"<div class='movie-card'>#{i} &nbsp; {title}</div>", unsafe_allow_html=True)
