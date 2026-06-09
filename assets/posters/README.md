# Offline Movie Posters

This directory is used for local movie poster assets. If the application is running in **Offline Mode** (or if TMDB API is unavailable), it will check this folder for custom image files before falling back to the curated cinema photography prints.

## Naming Convention

Save poster images using the movie's TMDB `movie_id` (available in the dataset/metadata) as the filename:
- Format: `assets/posters/{movie_id}.jpg` or `assets/posters/{movie_id}.png`
- Example: For *Inception* (`movie_id = 27205`), save the poster as `assets/posters/27205.jpg` or `assets/posters/27205.png`.

The backend API wrapper (`utils/api.py`) will automatically scan this folder and display the custom poster if it is present.
