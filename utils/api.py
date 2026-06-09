import os
import requests
import streamlit as st
import time
import pandas as pd
import urllib.parse
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

def get_api_key():
    """
    Safely retrieves the TMDB API key from:
    1. Streamlit Secrets (production hosting)
    2. Environment variables (.env)
    
    Returns None if the key is not configured or is set to placeholder text.
    """
    try:
        if hasattr(st, "secrets") and "TMDB_API_KEY" in st.secrets:
            key = st.secrets["TMDB_API_KEY"]
            if key and key.strip() and key.strip() != "your_tmdb_api_key_here":
                return key.strip()
    except Exception:
        pass

    key = os.getenv("TMDB_API_KEY")
    if key and key.strip() and key.strip() != "your_tmdb_api_key_here":
        return key.strip()

    return None

@st.cache_data(ttl=60, show_spinner=False)
def check_tmdb_connectivity():
    """
    Checks if the TMDB API is reachable with a fast 0.5-second timeout.
    Caches connectivity state for 60 seconds to prevent blocking UI threads.
    """
    api_key = get_api_key()
    if not api_key:
        return False
    url = f"https://api.themoviedb.org/3/configuration?api_key={api_key}"
    try:
        start_time = time.time()
        response = requests.get(url, timeout=0.5)
        elapsed = time.time() - start_time
        if response.status_code == 200:
            print(f"[CONN CHECK] TMDB API is REACHABLE. Response time: {elapsed:.3f}s")
            return True
        else:
            print(f"[CONN CHECK] TMDB API returned status {response.status_code}. Response time: {elapsed:.3f}s")
            return False
    except Exception as e:
        print(f"[CONN CHECK] TMDB API is UNREACHABLE. Error: {e}")
        return False

def get_poster_url(movie_id, title, genres=None):
    """
    Checks if a local poster exists under assets/posters/{movie_id}.jpg or png.
    Otherwise, returns one of 8 curated high-quality cinema-themed Unsplash graphics
    based on movie_id hash, ensuring visual diversity and a professional appearance.
    """
    # 1. Check local assets directory
    local_path_jpg = os.path.join("assets", "posters", f"{movie_id}.jpg")
    local_path_png = os.path.join("assets", "posters", f"{movie_id}.png")
    if os.path.exists(local_path_jpg):
        return local_path_jpg
    if os.path.exists(local_path_png):
        return local_path_png
        
    # 2. Curated premium cinema/movie-themed Unsplash graphics (loads instantly and looks highly artistic)
    fallback_images = [
        "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=500&auto=format&fit=crop", # Classic theater seats
        "https://images.unsplash.com/photo-1505686994434-e3cc5abf1330?q=80&w=500&auto=format&fit=crop", # Vintage film rolls
        "https://images.unsplash.com/photo-1485846234645-a62644f84728?q=80&w=500&auto=format&fit=crop", # Clapboard on director's table
        "https://images.unsplash.com/photo-1517604931442-7e0c8ed2963c?q=80&w=500&auto=format&fit=crop", # Vintage theater marquee
        "https://images.unsplash.com/photo-1536440136628-849c177e76a1?q=80&w=500&auto=format&fit=crop", # Movie reel light
        "https://images.unsplash.com/photo-1507679799987-c73779587ccf?q=80&w=500&auto=format&fit=crop", # Retro cinema curtains
        "https://images.unsplash.com/photo-1478720568477-152d9b164e26?q=80&w=500&auto=format&fit=crop", # Lens bokeh projector lights
        "https://images.unsplash.com/photo-1513151233558-d860c5398176?q=80&w=500&auto=format&fit=crop"  # Abstract theater lights
    ]
    
    # Select image deterministically based on movie_id
    img_idx = int(movie_id) % len(fallback_images)
    return fallback_images[img_idx]

def get_local_fallback(movie_id, local_df=None):
    """Returns metadata from the local database as a fallback."""
    if local_df is not None:
        try:
            match = local_df[local_df["movie_id"] == movie_id]
            if not match.empty:
                row = match.iloc[0]
                
                # Safely extract rating, runtime, and release date from local dataset
                rating = round(float(row["rating"]), 1) if ("rating" in local_df.columns and pd.notna(row["rating"])) else "N/A"
                runtime = int(row["runtime"]) if ("runtime" in local_df.columns and pd.notna(row["runtime"])) else "N/A"
                release_date = str(row["release_date"]) if ("release_date" in local_df.columns and pd.notna(row["release_date"])) else "N/A"
                
                # Retrieve dynamic poster (local check or movie-themed gallery)
                genres = row.get("genres", [])
                poster = get_poster_url(movie_id, row["title"], genres)
                
                return {
                    "movie_id": movie_id,
                    "title": row["title"],
                    "poster": poster,
                    "rating": rating,
                    "genres": genres,
                    "runtime": runtime,
                    "release_date": release_date,
                    "overview": row.get("overview", "No overview available."),
                    "director": row.get("crew", ["Unknown"])[0] if len(row.get("crew", [])) > 0 else "Unknown",
                    "cast": row.get("cast", []),
                    "success": False,
                    "error_msg": "Using offline database"
                }
        except Exception as e:
            print(f"[FALLBACK ERROR] Error extracting fallback fields: {e}")
            
    # Generic ultimate fallback
    return {
        "movie_id": movie_id,
        "title": "Unknown",
        "poster": "https://images.unsplash.com/photo-1485846234645-a62644f84728?q=80&w=500&auto=format&fit=crop",
        "rating": "N/A",
        "genres": [],
        "runtime": "N/A",
        "release_date": "N/A",
        "overview": "Information unavailable.",
        "director": "Unknown",
        "cast": [],
        "success": False
    }

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_movie_details(movie_id, _local_df=None):
    """
    Fetches detailed information for a movie from TMDB.
    Uses append_to_response=credits to get cast and director in one request.
    Falls back to local_df metadata if TMDB request fails or key is missing.
    """
    # Instant offline check bypass
    is_offline = hasattr(st, "session_state") and st.session_state.get("is_offline_mode", False)
    api_key = get_api_key()
    
    if is_offline or not api_key:
        print(f"[API LOG] URL: None (Bypassed: Offline Mode) | Fallback Triggered: True")
        return get_local_fallback(movie_id, _local_df)
        
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US&append_to_response=credits"
    
    # Bypasses requests if network test failed to eliminate slow timeout lag
    if not check_tmdb_connectivity():
        print(f"[API LOG] URL: {url} | Status: Skipped (Network Offline) | Fallback Triggered: True")
        return get_local_fallback(movie_id, _local_df)
        
    start_time = time.time()
    try:
        response = requests.get(url, timeout=5.0)
        elapsed = time.time() - start_time
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get("poster_path")
            
            # Use local/placeholder posters if no poster path on TMDB
            genres = [g["name"] for g in data.get("genres", [])]
            poster = f"https://image.tmdb.org/t/p/w500/{poster_path}" if poster_path else get_poster_url(movie_id, data.get('title', 'Poster'), genres)
            
            rating = round(data.get("vote_average", 0.0), 1)
            runtime = data.get("runtime", "N/A")
            release_date = data.get("release_date", "N/A")
            overview = data.get("overview", "No overview available.")
            
            director = "Unknown"
            cast = []
            credits = data.get("credits", {})
            for crew_member in credits.get("crew", []):
                if crew_member.get("job") == "Director":
                    director = crew_member.get("name")
                    break
            for actor in credits.get("cast", [])[:3]:
                cast.append(actor.get("name"))
                
            print(f"[API LOG] URL: {url} | Status: 200 OK | Time: {elapsed:.3f}s | Fallback Triggered: False")
            return {
                "movie_id": movie_id,
                "title": data.get("title", "Unknown"),
                "poster": poster,
                "rating": rating,
                "genres": genres,
                "runtime": runtime,
                "release_date": release_date,
                "overview": overview,
                "director": director,
                "cast": cast,
                "success": True
            }
        else:
            print(f"[API LOG] URL: {url} | Status: {response.status_code} | Time: {elapsed:.3f}s | Fallback Triggered: True")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[API LOG] URL: {url} | Status: Exception ({type(e).__name__}) | Time: {elapsed:.3f}s | Fallback Triggered: True")
        
    return get_local_fallback(movie_id, _local_df)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_trending_movies():
    """
    Fetches the daily trending movies from TMDB.
    Returns empty list if API key is not configured or TMDB request fails.
    """
    is_offline = hasattr(st, "session_state") and st.session_state.get("is_offline_mode", False)
    api_key = get_api_key()
    if is_offline or not api_key:
        return []
        
    url = f"https://api.themoviedb.org/3/trending/movie/day?api_key={api_key}"
    
    if not check_tmdb_connectivity():
        print(f"[API LOG] URL: {url} | Status: Skipped (Network Offline) | Fallback Triggered: True")
        return []
        
    start_time = time.time()
    try:
        response = requests.get(url, timeout=5.0)
        elapsed = time.time() - start_time
        if response.status_code == 200:
            data = response.json()
            trending = []
            for movie in data.get("results", [])[:15]:
                poster_path = movie.get("poster_path")
                poster = f"https://image.tmdb.org/t/p/w500/{poster_path}" if poster_path else get_poster_url(movie.get("id"), movie.get('title'), [])
                trending.append({
                    "movie_id": movie.get("id"),
                    "title": movie.get("title"),
                    "poster": poster,
                    "rating": round(movie.get("vote_average", 0.0), 1)
                })
            print(f"[API LOG] URL: {url} | Status: 200 OK | Time: {elapsed:.3f}s | Fallback Triggered: False")
            return trending
        else:
            print(f"[API LOG] URL: {url} | Status: {response.status_code} | Time: {elapsed:.3f}s | Fallback Triggered: True")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[API LOG] URL: {url} | Status: Exception ({type(e).__name__}) | Time: {elapsed:.3f}s | Fallback Triggered: True")
        
    return []

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_movie_trailer(movie_id):
    """
    Fetches the YouTube trailer video key for a given movie ID from TMDB.
    Returns the full YouTube watch URL or None.
    """
    is_offline = hasattr(st, "session_state") and st.session_state.get("is_offline_mode", False)
    api_key = get_api_key()
    if is_offline or not api_key:
        return None
        
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={api_key}&language=en-US"
    
    if not check_tmdb_connectivity():
        print(f"[API LOG] URL: {url} | Status: Skipped (Network Offline) | Fallback Triggered: True")
        return None
        
    start_time = time.time()
    try:
        response = requests.get(url, timeout=5.0)
        elapsed = time.time() - start_time
        if response.status_code == 200:
            data = response.json()
            for video in data.get("results", []):
                if video.get("site") == "YouTube" and video.get("type") == "Trailer":
                    key = video.get("key")
                    print(f"[API LOG] URL: {url} | Status: 200 OK | Time: {elapsed:.3f}s | Fallback Triggered: False")
                    return f"https://www.youtube.com/watch?v={key}"
            print(f"[API LOG] URL: {url} | Status: No YouTube Trailer | Time: {elapsed:.3f}s | Fallback Triggered: True")
        else:
            print(f"[API LOG] URL: {url} | Status: {response.status_code} | Time: {elapsed:.3f}s | Fallback Triggered: True")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[API LOG] URL: {url} | Status: Exception ({type(e).__name__}) | Time: {elapsed:.3f}s | Fallback Triggered: True")
        
    return None
