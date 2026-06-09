import json
import os

HISTORY_FILE = "search_history.json"

def load_history():
    """Loads search history from search_history.json."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            content = f.read().strip()
            if not content:
                return []
            history = json.loads(content)
            if isinstance(history, list):
                return history
            return []
    except Exception as e:
        return []

def save_history(history):
    """Saves search history to search_history.json."""
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=4)
        return True
    except Exception as e:
        return False

def add_search(movie_title):
    """
    Adds a movie title to search history.
    Pushes it to the top, removes duplicates, and limits length to 10.
    """
    if not movie_title or not isinstance(movie_title, str):
        return False
        
    history = load_history()
    
    # Remove duplicates if already exists
    if movie_title in history:
        history.remove(movie_title)
        
    # Add to beginning
    history.insert(0, movie_title)
    
    # Limit to 10 searches
    history = history[:10]
    
    return save_history(history)

def clear_history():
    """Clears search history."""
    return save_history([])
