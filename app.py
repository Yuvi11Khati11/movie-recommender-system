import streamlit as st
import pandas as pd
import os
import time
from concurrent.futures import ThreadPoolExecutor
from utils.api import fetch_movie_details, fetch_trending_movies, fetch_movie_trailer, get_api_key, check_tmdb_connectivity, get_local_fallback
from utils.recommender import load_movies_df, load_similarity_matrix, get_recommendations
from utils.favorites import load_favorites, add_favorite, remove_favorite, clear_favorites
from utils.search_history import load_history, add_search, clear_history

# ---------- PAGE CONFIGURATION ----------
st.set_page_config(
    page_title="CinemAI - Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- SESSION STATE INITIALIZATION ----------
if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None
if "search_query" not in st.session_state:
    st.session_state.search_query = ""

# Session-state cache for loaded details & recommendations to optimize performance
if "last_selected_movie" not in st.session_state:
    st.session_state.last_selected_movie = None
if "last_engine_choice" not in st.session_state:
    st.session_state.last_engine_choice = None
if "active_details" not in st.session_state:
    st.session_state.active_details = None
if "active_recommendations" not in st.session_state:
    st.session_state.active_recommendations = None

# ---------- LOAD CSS STYLES ----------
css_path = os.path.join("assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------- LOAD MODEL FILES (CACHED ONCE, LAZY LOADING SIMILARITY) ----------
movies_df = load_movies_df()

if movies_df is None:
    st.error("Unable to load model datasets. Please check model/ folder files.")
    st.stop()

# ---------- INSTANT CONNECTIVITY CHECK (ONCE ON STARTUP) ----------
if "is_offline_mode" not in st.session_state:
    # 0.5-second fast connection check
    st.session_state.is_offline_mode = not check_tmdb_connectivity()

# ---------- SIDEBAR NAVIGATION & CONTROLS ----------
# Smaller sidebar branding logo (width=130)
logo_path = os.path.join("assets", "logo.png")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=130)
else:
    st.sidebar.title("🎬 CinemAI")

# Offline Mode / Active Key status indicators in sidebar
if st.session_state.is_offline_mode:
    st.sidebar.info("📶 Status: **Offline Mode** (Active)")
else:
    st.sidebar.success("🟢 Status: **Online Mode**")

st.sidebar.markdown("---")

# 2. Engine Selector
st.sidebar.subheader("⚙️ Settings")
engine_choice = st.sidebar.radio(
    "Recommendation Engine",
    ["TF-IDF (Default)", "CountVectorizer"],
    help="TF-IDF weights terms by uniqueness, while CountVectorizer does basic term counts."
)

# LAZY LOAD SIMILARITY MATRIX ON DEMAND
similarity_matrix = load_similarity_matrix(engine_choice)
if similarity_matrix is None:
    st.sidebar.error("⚠️ **Model files missing!** Please run `python model_builder.py` in your terminal to build similarity pickles.")

# 3. Genre Filter
selected_genre = st.sidebar.selectbox(
    "Filter Library by Genre",
    ["All Genres", "Action", "Adventure", "Comedy", "Horror", "Romance", "Thriller", "Sci-Fi", "Animation", "Family", "Drama"],
    index=0
)

# Apply genre filter to movies_df
if selected_genre == "All Genres":
    filtered_df = movies_df
else:
    filtered_df = movies_df[movies_df['genres'].apply(lambda g_list: selected_genre in g_list if isinstance(g_list, list) else False)]

# 4. Sidebar Search History
favorites = load_favorites(movies_df)
history = load_history()

if history:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🕒 Recent Searches")
    for search in history:
        if st.sidebar.button(f"🎬 {search}", key=f"hist_{search}", use_container_width=True):
            st.session_state.selected_movie = search
            st.session_state.search_query = ""
            st.rerun()
            
    if st.sidebar.button("🧹 Clear History", use_container_width=True):
        clear_history()
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("CinemAI Upgrade • Portfolio Project")

# ---------- APP LAYOUT TABS ----------
tab1, tab2 = st.tabs(["🎬 Explore & Recommender", "📊 Analytics Dashboard"])

with tab1:
    # ---------- COMPACT HERO BANNER ----------
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">🎬 CinemAI Recommender</h1>
        <p class="hero-subtitle">Search a movie to get stemmed content-based suggestions instantly.</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------- OFFLINE MODE BADGE NOTICE ----------
    if st.session_state.is_offline_mode:
        st.markdown("""
        <div style="background-color: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 6px; padding: 8px 12px; margin-bottom: 15px; font-size: 0.8rem; color: #f59e0b; display: flex; align-items: center;">
            <span>📶 <strong>Offline Mode Active:</strong> Running entirely from the local database. Live TMDB posters, ratings, and trailers are currently unavailable.</span>
        </div>
        """, unsafe_allow_html=True)

    # ---------- COMPACT SEARCH BAR (REAL-TIME AUTOCOMPLETE) ----------
    st.subheader("🔍 Search Movies")
    search_query = st.text_input(
        "Search Movies Input",
        value=st.session_state.search_query,
        placeholder="Type to search (e.g. Inception, Avatar, The Dark Knight)...",
        key="search_input",
        label_visibility="collapsed"
    )

    # Autocomplete Suggestions List (Clean Vertical Dropdown Suggestions)
    if search_query:
        matches = filtered_df[filtered_df['title'].str.contains(search_query, case=False, na=False)]
        
        if not matches.empty:
            st.markdown("<div style='margin-bottom: 4px; font-size: 0.78rem; color: #8b949e;'>Select suggestion to search:</div>", unsafe_allow_html=True)
            for idx, row in enumerate(matches.head(5).itertuples()):
                if st.button(f"🔍 {row.title}", key=f"sug_{row.movie_id}", use_container_width=True):
                    st.session_state.selected_movie = row.title
                    st.session_state.search_query = "" # Reset search field on selection
                    st.rerun()
        else:
            st.info(f"No movies matching '{search_query}' found in '{selected_genre}'.")
            
    st.markdown("---")

    # ---------- LAZY LOAD RECOMMENDATION PIPELINE (CACHED IN SESSION STATE) ----------
    if st.session_state.selected_movie:
        selected_title = st.session_state.selected_movie
        
        # Check if selected movie or engine selection has changed
        is_new_movie = selected_title != st.session_state.last_selected_movie
        is_new_engine = engine_choice != st.session_state.last_engine_choice
        
        if is_new_movie or is_new_engine:
            st.session_state.last_selected_movie = selected_title
            st.session_state.last_engine_choice = engine_choice
            
            # Record in Search History
            add_search(selected_title)
            
            movie_row = movies_df[movies_df['title'].str.lower() == selected_title.lower()]
            if not movie_row.empty:
                movie_id = int(movie_row.iloc[0]['movie_id'])
                
                # Fetch details (cached API / offline bypass inside details logic)
                start_t = time.time()
                with st.spinner("Retrieving details..."):
                    details = fetch_movie_details(movie_id, movies_df)
                elapsed_t = time.time() - start_t
                print(f"[UI PERF] Loaded details for '{selected_title}' in {elapsed_t:.3f}s")
                st.session_state.active_details = details
                
                # Fetch recommendations and pre-load all detail records in PARALLEL to minimize loading delays
                start_rec = time.time()
                recs_with_details = []
                
                if similarity_matrix is None:
                    st.error("⚠️ **Recommendation similarity model file is missing.** Please compile the database by running `python model_builder.py` in your terminal.")
                else:
                    with st.spinner("Generating recommendations..."):
                        recs = get_recommendations(selected_title, movies_df, similarity_matrix, num_recommendations=5)
                        
                        print(f"\n--- [REC ENGINE DEBUG] Selected Movie: '{selected_title}' (ID: {movie_id}) ---")
                        for rec in recs:
                            print(f"  -> Recommended: '{rec['title']}' (ID: {rec['movie_id']})")
                            
                        with ThreadPoolExecutor(max_workers=5) as executor:
                            futures = {executor.submit(fetch_movie_details, rec["movie_id"], movies_df): rec for rec in recs}
                            for future in futures:
                                rec = futures[future]
                                try:
                                    rec_details = future.result()
                                except Exception:
                                    rec_details = fetch_movie_details(rec["movie_id"], movies_df)
                                
                                # Print confirmation of the custom fetched ID and corresponding poster path/fallback URL
                                print(f"  [FETCH DEBUG] Movie ID: {rec['movie_id']} -> Fetched Title: '{rec_details.get('title')}' -> Poster: {rec_details.get('poster')}")
                                
                                recs_with_details.append({
                                    **rec,
                                    "details": rec_details
                                })
                        print("------------------------------------------------------------\n")
                elapsed_rec = time.time() - start_rec
                print(f"[UI PERF] Concurrently calculated recommendations for '{selected_title}' in {elapsed_rec:.3f}s")
                st.session_state.active_recommendations = recs_with_details
            else:
                st.session_state.active_details = None
                st.session_state.active_recommendations = None

        # ---------- RENDER ACTIVE DETAILS CARD (FROM STATE) ----------
        details = st.session_state.active_details
        if details:
            st.markdown(f'<div class="detail-container">', unsafe_allow_html=True)
            col_post, col_info = st.columns([1, 3])
            
            with col_post:
                st.image(details["poster"], use_container_width=True)
                
                # Clean styled interactive action buttons
                st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
                is_fav = any(f.get("movie_id") == details["movie_id"] for f in favorites)
                
                # Setup details rating fallback displaying "Rating Unavailable" if TMDB success is False
                if details.get("success", False) and details["rating"] != "N/A":
                    active_rating_disp = f"⭐ {details['rating']}"
                else:
                    active_rating_disp = "Rating Unavailable"
                
                if is_fav:
                    if st.button("💔 Remove Favorite", use_container_width=True, type="secondary"):
                        remove_favorite(details["movie_id"])
                        st.success("Removed!")
                        st.rerun()
                else:
                    if st.button("❤️ Add Favorite", use_container_width=True, type="primary"):
                        add_favorite(details["movie_id"], details["title"], details["poster"], active_rating_disp)
                        st.success("Added!")
                        st.rerun()
            
            with col_info:
                st.markdown(f'<div class="detail-title">{details["title"]}</div>', unsafe_allow_html=True)
                
                # Metadata line
                runtime_str = f"⏱️ {details['runtime']} mins" if details['runtime'] != "N/A" else "⏱️ N/A"
                date_str = f"📅 {details['release_date']}" if details['release_date'] != "N/A" else "📅 N/A"
                
                meta_html = f'<div class="detail-meta"><span class="detail-rating">{active_rating_disp}</span>'
                meta_html += f'<span>{runtime_str}</span><span>{date_str}</span>'
                for g in details["genres"]:
                    meta_html += f'<span class="genre-tag">{g}</span>'
                meta_html += '</div>'
                st.markdown(meta_html, unsafe_allow_html=True)
                
                # Overview
                st.markdown('<div class="detail-section-title">Overview</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="detail-text">{details["overview"]}</div>', unsafe_allow_html=True)
                
                # Cast/Crew details
                st.markdown('<div class="detail-section-title">Director & Top Cast</div>', unsafe_allow_html=True)
                cast_str = ", ".join(details["cast"]) if details["cast"] else "Unknown"
                st.markdown(f"""
                <div class="detail-text">
                    <strong>Director:</strong> {details['director']}<br/>
                    <strong>Top Cast:</strong> {cast_str}
                </div>
                """, unsafe_allow_html=True)
                
                # Show trailer section only when online and trailer exists (Hidden completely in Offline Mode)
                if not st.session_state.is_offline_mode:
                    trailer_url = fetch_movie_trailer(details["movie_id"])
                    if trailer_url:
                        st.markdown('<div class="detail-section-title">Trailer</div>', unsafe_allow_html=True)
                        with st.expander("▶ Watch Trailer", expanded=False):
                            st.video(trailer_url)
                        
            st.markdown('</div>', unsafe_allow_html=True)
            
            # ---------- RENDER RECOMMENDATIONS GRID (FROM STATE) ----------
            st.subheader(f"✨ Recommended Movies (via {engine_choice})")
            recs = st.session_state.active_recommendations
            
            if recs:
                rec_cols = st.columns(5)
                for idx, rec in enumerate(recs):
                    rec_details = rec["details"]
                    
                    with rec_cols[idx]:
                        explanation = rec["explanation"]
                        reasons_html = ""
                        
                        if explanation["director"]:
                            reasons_html += f'<span class="reason-badge reason-badge-director">🎬 Dir: {explanation["director"][0]}</span>'
                        for g in explanation["genres"][:2]:
                            reasons_html += f'<span class="reason-badge">{g}</span>'
                        if explanation["cast"]:
                            reasons_html += f'<span class="reason-badge reason-badge-cast">👥 {explanation["cast"][0]}</span>'
                        if not reasons_html and explanation["keywords"]:
                            reasons_html += f'<span class="reason-badge">🏷️ {explanation["keywords"][0]}</span>'
                        if not reasons_html:
                            reasons_html = '<span class="reason-badge">Similar topics</span>'

                        # Setup recommendation card rating fallback
                        if rec_details.get("success", False) and rec_details["rating"] != "N/A":
                            rec_rating_disp = f"⭐ {rec_details['rating']}"
                        else:
                            rec_rating_disp = "Rating Unavailable"

                        # Render Clean Card
                        st.markdown(f"""
                        <div class="movie-card-container">
                            <img src="{rec_details['poster']}" style="width: 100%; border-radius: 6px; object-fit: cover; aspect-ratio: 2/3;" />
                            <div class="movie-title-text" title="{rec['title']}">{rec['title']}</div>
                            <div>
                                <span class="movie-rating-badge">{rec_rating_disp}</span>
                            </div>
                            <div class="explanation-container">
                                <div class="explanation-title">Matches:</div>
                                {reasons_html}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Interaction buttons
                        btn_c1, btn_c2 = st.columns([3, 1])
                        with btn_c1:
                            if st.button("🔍 Details", key=f"rec_view_{rec['movie_id']}", use_container_width=True):
                                st.session_state.selected_movie = rec['title']
                                st.rerun()
                        with btn_c2:
                            rec_is_fav = any(f.get("movie_id") == rec['movie_id'] for f in favorites)
                            if rec_is_fav:
                                if st.button("💔", key=f"rec_fav_{rec['movie_id']}", help="Remove from favorites", use_container_width=True):
                                    remove_favorite(rec['movie_id'])
                                    st.rerun()
                            else:
                                if st.button("❤️", key=f"rec_fav_{rec['movie_id']}", help="Add to favorites", use_container_width=True):
                                    add_favorite(rec['movie_id'], rec['title'], rec_details['poster'], rec_rating_disp)
                                    st.rerun()
            else:
                st.info("No recommendations found.")
        else:
            st.error("Selected movie not found in dataset.")
            
        st.markdown("---")

    # ---------- OFFLINE FEED vs ONLINE TRENDING SECTION ----------
    if st.session_state.is_offline_mode:
        # REDESIGNED OFFLINE HOME LANDING PAGE WITH 3 LOCAL RECOMMENDATION ROWS
        
        # 1. Editor Picks (Hand-picked classic titles)
        st.subheader("💡 Editor Picks")
        editor_titles = ["Inception", "The Dark Knight", "Interstellar", "Pulp Fiction", "The Matrix"]
        edit_cols = st.columns(5)
        for i, title in enumerate(editor_titles):
            row = movies_df[movies_df["title"] == title]
            if not row.empty:
                movie_id = int(row.iloc[0]["movie_id"])
                details_local = get_local_fallback(movie_id, movies_df)
                with edit_cols[i]:
                    st.markdown(f"""
                    <div class="movie-card-container">
                        <img src="{details_local['poster']}" style="width: 100%; border-radius: 6px; object-fit: cover; aspect-ratio: 2/3;" />
                        <div class="movie-title-text" title="{title}">{title}</div>
                        <div>
                            <span class="movie-rating-badge">Rating Unavailable</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("🔍 Details", key=f"edit_view_{movie_id}", use_container_width=True):
                        st.session_state.selected_movie = title
                        st.rerun()
                        
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

        # 2. Popular Recommendations (Sorted locally by popularity column)
        st.subheader("🔥 Popular Recommendations (Local)")
        popular_movies = movies_df.sort_values(by="popularity", ascending=False).head(5)
        pop_cols = st.columns(5)
        for i, row in enumerate(popular_movies.itertuples()):
            movie_id = int(row.movie_id)
            details_local = get_local_fallback(movie_id, movies_df)
            with pop_cols[i]:
                st.markdown(f"""
                <div class="movie-card-container">
                    <img src="{details_local['poster']}" style="width: 100%; border-radius: 6px; object-fit: cover; aspect-ratio: 2/3;" />
                    <div class="movie-title-text" title="{row.title}">{row.title}</div>
                    <div>
                        <span class="movie-rating-badge">Rating Unavailable</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("🔍 Details", key=f"pop_view_{movie_id}", use_container_width=True):
                    st.session_state.selected_movie = row.title
                    st.rerun()

        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

        # 3. Featured Local Recommendations (Top matches similar to "Inception" locally)
        st.subheader("🎯 Featured Recommendations (Similar to 'Inception')")
        inception_recs = get_recommendations("Inception", movies_df, similarity_matrix, num_recommendations=5)
        feat_cols = st.columns(5)
        for i, rec in enumerate(inception_recs):
            movie_id = int(rec["movie_id"])
            details_local = get_local_fallback(movie_id, movies_df)
            with feat_cols[i]:
                st.markdown(f"""
                <div class="movie-card-container">
                    <img src="{details_local['poster']}" style="width: 100%; border-radius: 6px; object-fit: cover; aspect-ratio: 2/3;" />
                    <div class="movie-title-text" title="{rec['title']}">{rec['title']}</div>
                    <div>
                        <span class="movie-rating-badge">Rating Unavailable</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("🔍 Details", key=f"feat_view_{movie_id}", use_container_width=True):
                    st.session_state.selected_movie = rec['title']
                    st.rerun()

    else:
        # ONLINE MODE TRENDING SECTION (TMDB API IN USE)
        st.subheader("🔥 Trending Movies Today")
        trending = fetch_trending_movies()
        
        if trending:
            trend_cols = st.columns(5)
            for i, t_movie in enumerate(trending[:10]):  # display top 10 in two rows of 5
                col_idx = i % 5
                if i > 0 and col_idx == 0:
                    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                    trend_cols = st.columns(5)
                    
                with trend_cols[col_idx]:
                    st.markdown(f"""
                    <div class="movie-card-container">
                        <img src="{t_movie['poster']}" style="width: 100%; border-radius: 6px; object-fit: cover; aspect-ratio: 2/3;" />
                        <div class="movie-title-text" title="{t_movie['title']}">{t_movie['title']}</div>
                        <div>
                            <span class="movie-rating-badge">⭐ {t_movie['rating']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    t_btn_c1, t_btn_c2 = st.columns([3, 1])
                    with t_btn_c1:
                        if st.button("🔍 Details", key=f"trend_view_{t_movie['movie_id']}_{i}", use_container_width=True):
                            st.session_state.selected_movie = t_movie['title']
                            st.rerun()
                    with t_btn_c2:
                        t_is_fav = any(f.get("movie_id") == t_movie['movie_id'] for f in favorites)
                        if t_is_fav:
                            if st.button("💔", key=f"trend_fav_{t_movie['movie_id']}_{i}", use_container_width=True):
                                remove_favorite(t_movie['movie_id'])
                                st.rerun()
                        else:
                            if st.button("❤️", key=f"trend_fav_{t_movie['movie_id']}_{i}", use_container_width=True):
                                add_favorite(t_movie['movie_id'], t_movie['title'], t_movie['poster'], f"⭐ {t_movie['rating']}")
                                st.rerun()
        else:
            st.info("Unable to retrieve trending movies.")

    st.markdown("---")

    # ---------- FAVORITES TRAY ----------
    st.subheader(f"❤️ Your Favorites ({len(favorites)})")
    
    if len(favorites) == 0:
        st.info("Click the ❤️ icon on any movie to bookmark it here!")
    else:
        fav_cols = st.columns(5)
        for idx, fav in enumerate(favorites):
            col_idx = idx % 5
            
            if idx > 0 and col_idx == 0:
                st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                fav_cols = st.columns(5)
                
            with fav_cols[col_idx]:
                rating_disp = fav.get("rating", "Rating Unavailable")
                st.markdown(f"""
                <div class="movie-card-container">
                    <img src="{fav['poster']}" style="width: 100%; border-radius: 6px; object-fit: cover; aspect-ratio: 2/3;" />
                    <div class="movie-title-text" title="{fav['title']}">{fav['title']}</div>
                    <div>
                        <span class="movie-rating-badge">{rating_disp}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                fav_btn_c1, fav_btn_c2 = st.columns([3, 1])
                with fav_btn_c1:
                    if st.button("🔍 Details", key=f"fav_view_{fav.get('movie_id')}_{idx}", use_container_width=True):
                        st.session_state.selected_movie = fav['title']
                        st.rerun()
                with fav_btn_c2:
                    if st.button("💔", key=f"fav_remove_{fav.get('movie_id')}_{idx}", use_container_width=True):
                        remove_favorite(fav.get('movie_id'), fav.get('title'))
                        st.rerun()
                        
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        if st.button("🧹 Clear All Favorites", type="secondary"):
            clear_favorites()
            st.success("Cleared all bookmarks!")
            st.rerun()


with tab2:
    # ---------- DASHBOARD / ANALYTICS ----------
    st.subheader("📊 Recommendation Pipeline Analytics")
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric(label="Total Library Movies", value=f"{len(movies_df)}")
    with m_col2:
        st.metric(label="Total Saved Favorites", value=f"{len(favorites)}")
    with m_col3:
        st.metric(label="Active Search Engine", value="TF-IDF" if engine_choice == "TF-IDF (Default)" else "CountVectorizer")
    with m_col4:
        st.metric(label="System Mode", value="Offline" if st.session_state.is_offline_mode else "Online")
        
    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("<div class='detail-section-title'>Genre Distribution in Database</div>", unsafe_allow_html=True)
        genre_counts = {}
        for g_list in movies_df["genres"]:
            if isinstance(g_list, list):
                for g in g_list:
                    genre_counts[g] = genre_counts.get(g, 0) + 1
                    
        df_genres = pd.DataFrame(list(genre_counts.items()), columns=["Genre", "Count"]).sort_values("Count", ascending=False)
        st.bar_chart(data=df_genres, x="Genre", y="Count", color="#3b82f6")
        
    with col_chart2:
        st.markdown("<div class='detail-section-title'>Similarity Pipeline Settings</div>", unsafe_allow_html=True)
        st.markdown(f"""
        This project pre-processes the TMDB database, stems tokens using NLTK PorterStemmer, and computes the Cosine Similarity of vectors.
        
        * **Preprocessing**: Text fields (`overview`, `keywords`) are tokenized and lowercased. Actor names and director names are space-collapsed to maintain unique vector identities.
        * **Vocabulary Size**: 5,000 features.
        * **Stemming**: `PorterStemmer` reduces grammatical variants (e.g. *activities* $\\rightarrow$ *activ*), improving recommendation matches.
        * **Dynamic Caching**: Active selection recommendations are saved in Streamlit's `st.session_state` to prevent duplicate computations.
        """)
        
        if st.checkbox("Show database preview"):
            st.dataframe(movies_df[["movie_id", "title", "genres", "crew"]].head(10))