import pickle
import streamlit as st
import time

@st.cache_resource
def load_movies_df():
    """Loads and caches the enriched movie dataframe."""
    start_time = time.time()
    try:
        movies_df = pickle.load(open('model/movie_dict.pkl', 'rb'))
        elapsed = time.time() - start_time
        print(f"[PERF] Loaded movies dataframe in {elapsed:.3f}s")
        return movies_df
    except Exception as e:
        st.error(f"Error loading movie database: {e}")
        return None

@st.cache_resource
def load_similarity_matrix(engine_type):
    """
    Lazily loads and caches the specific similarity matrix only when requested.
    This saves memory and cuts application startup time in half.
    """
    start_time = time.time()
    try:
        if engine_type == "TF-IDF (Default)" or engine_type == "TF-IDF":
            try:
                matrix = pickle.load(open('model/similarity_tfidf.pkl', 'rb'))
            except FileNotFoundError:
                matrix = pickle.load(open('model/similarity.pkl', 'rb'))
        else:
            matrix = pickle.load(open('model/similarity_cv.pkl', 'rb'))
            
        elapsed = time.time() - start_time
        print(f"[PERF] Loaded similarity matrix for '{engine_type}' in {elapsed:.3f}s")
        return matrix
    except Exception as e:
        st.error(f"Error loading similarity matrix: {e}")
        return None

def get_recommendations(movie_title, movies_df, similarity_matrix, num_recommendations=5):
    """
    Finds recommendations for a given movie title.
    Returns a list of dicts containing movie_id, title, and explainable AI reasons.
    """
    if similarity_matrix is None:
        print("[ERROR] Similarity matrix is None. Cannot compute recommendations.")
        return []

    start_time = time.time()
    try:
        movie_index = movies_df[movies_df['title'].str.lower() == movie_title.lower()].index[0]
    except IndexError:
        return []

    distances = similarity_matrix[movie_index]

    # Sort movies
    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:num_recommendations+1]

    input_movie = movies_df.iloc[movie_index]
    recommendations = []

    for i in movie_list:
        rec_index = i[0]
        rec_movie = movies_df.iloc[rec_index]
        
        # Calculate Explainable AI features
        shared_genres = list(set(input_movie.get('genres', [])) & set(rec_movie.get('genres', [])))
        shared_director = list(set(input_movie.get('crew', [])) & set(rec_movie.get('crew', [])))
        shared_cast = list(set(input_movie.get('cast', [])) & set(rec_movie.get('cast', [])))
        shared_keywords = list(set(input_movie.get('keywords', [])) & set(rec_movie.get('keywords', [])))
        
        explanation = {
            "genres": shared_genres,
            "director": shared_director,
            "cast": shared_cast,
            "keywords": shared_keywords[:4]
        }

        recommendations.append({
            "movie_id": int(rec_movie['movie_id']),
            "title": rec_movie['title'],
            "score": float(i[1]),
            "explanation": explanation
        })

    elapsed = time.time() - start_time
    print(f"[PERF] get_recommendations for '{movie_title}' took {elapsed:.4f}s")
    return recommendations
