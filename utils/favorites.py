import json
import os

FAVORITES_FILE = "favorites.json"

def load_favorites(movies_df=None):
    """
    Loads favorites from favorites.json.
    Migrates legacy list of strings (movie names) to list of dicts.
    """
    if not os.path.exists(FAVORITES_FILE):
        return []
        
    try:
        with open(FAVORITES_FILE, "r") as f:
            content = f.read().strip()
            if not content:
                return []
            data = json.loads(content)
            
            # Migrate if old format (list of strings)
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
                migrated = []
                for title in data:
                    movie_id = None
                    if movies_df is not None:
                        match = movies_df[movies_df['title'].str.lower() == title.lower()]
                        if not match.empty:
                            movie_id = int(match.iloc[0]['movie_id'])
                            
                    migrated.append({
                        "movie_id": movie_id,
                        "title": title,
                        "poster": "https://images.unsplash.com/photo-1485846234645-a62644f84728?q=80&w=500&auto=format&fit=crop",
                        "rating": "N/A"
                    })
                save_favorites(migrated)
                return migrated
            
            return data
    except Exception as e:
        return []

def save_favorites(favs):
    """Saves the favorites list to favorites.json."""
    try:
        with open(FAVORITES_FILE, "w") as f:
            json.dump(favs, f, indent=4)
        return True
    except Exception as e:
        return False

def add_favorite(movie_id, title, poster, rating):
    """Adds a movie to the favorites list if not already present."""
    favs = load_favorites()
    
    # Check if already exists (by movie_id if present, else by title)
    for fav in favs:
        if movie_id is not None and fav.get("movie_id") == movie_id:
            return False, "Already in favorites!"
        if fav.get("title").lower() == title.lower():
            return False, "Already in favorites!"
            
    favs.append({
        "movie_id": movie_id,
        "title": title,
        "poster": poster,
        "rating": rating
    })
    
    if save_favorites(favs):
        return True, "Added to favorites!"
    return False, "Failed to save favorites."

def remove_favorite(movie_id, title=None):
    """Removes a movie from the favorites list."""
    favs = load_favorites()
    original_len = len(favs)
    
    if movie_id is not None:
        favs = [f for f in favs if f.get("movie_id") != movie_id]
    elif title is not None:
        favs = [f for f in favs if f.get("title").lower() != title.lower()]
        
    if len(favs) < original_len:
        save_favorites(favs)
        return True
    return False

def clear_favorites():
    """Clears all favorites."""
    return save_favorites([])
