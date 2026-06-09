import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import sys

# Try importing nltk, print message if missing
try:
    import nltk
    from nltk.stem.porter import PorterStemmer
except ImportError:
    print("NLTK is not installed. Please install it using: pip install nltk")
    sys.exit(1)

# Load datasets
print("Loading datasets...")
movies = pd.read_csv('dataset/tmdb_5000_movies.csv')
credits = pd.read_csv('dataset/tmdb_5000_credits.csv')

# Merge datasets
movies = movies.merge(credits, on='title')

# Select important columns (including popularity for offline popular recommendations)
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'vote_average', 'runtime', 'release_date', 'popularity']]

# Remove missing values
movies.dropna(subset=['movie_id', 'title', 'overview'], inplace=True)

# Helper function to convert JSON-like string columns
def convert_json_list(text):
    L = []
    try:
        for i in ast.literal_eval(text):
            L.append(i['name'])
    except Exception as e:
        pass
    return L

# Helper function to extract top 3 actors
def convert_cast(text):
    L = []
    counter = 0
    try:
        for i in ast.literal_eval(text):
            if counter < 3:
                L.append(i['name'])
                counter += 1
            else:
                break
    except Exception as e:
        pass
    return L

# Helper function to extract director
def fetch_director(text):
    L = []
    try:
        for i in ast.literal_eval(text):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
    except Exception as e:
        pass
    return L

# Helper function to collapse spaces (e.g. "Sam Worthington" -> "SamWorthington")
def collapse(L):
    return [i.replace(" ", "") for i in L]

print("Preprocessing metadata...")
# Save raw metadata for movie details, genre filters, and Explainable AI
movies['genres_raw'] = movies['genres'].apply(convert_json_list)
movies['keywords_raw'] = movies['keywords'].apply(convert_json_list)
movies['cast_raw'] = movies['cast'].apply(convert_cast)
movies['crew_raw'] = movies['crew'].apply(fetch_director)
movies['overview_raw'] = movies['overview']

# Process tag metadata by collapsing spaces for better tag similarity matching
movies['genres'] = movies['genres_raw'].apply(collapse)
movies['keywords'] = movies['keywords_raw'].apply(collapse)
movies['cast'] = movies['cast_raw'].apply(collapse)
movies['crew'] = movies['crew_raw'].apply(collapse)
movies['overview'] = movies['overview_raw'].apply(lambda x: x.split() if isinstance(x, str) else [])

# Create tags column
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Create enriched dataframe that preserves both tags, raw fields, ratings, runtime, release date, and popularity
new_df = pd.DataFrame({
    'movie_id': movies['movie_id'],
    'title': movies['title'],
    'overview': movies['overview_raw'],
    'genres': movies['genres_raw'],
    'keywords': movies['keywords_raw'],
    'cast': movies['cast_raw'],
    'crew': movies['crew_raw'],
    'rating': movies['vote_average'],
    'runtime': movies['runtime'],
    'release_date': movies['release_date'],
    'popularity': movies['popularity'],
    'tags': movies['tags'].apply(lambda x: " ".join(x).lower())
})

# Stemming helper
ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

print("Applying PorterStemmer stemming to tags...")
new_df['tags'] = new_df['tags'].apply(stem)

# Vectorization 1: TF-IDF Vectorizer (Default Engine)
print("Computing TF-IDF similarity matrix...")
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_vectors = tfidf.fit_transform(new_df['tags']).toarray()
similarity_tfidf = cosine_similarity(tfidf_vectors)

# Vectorization 2: Count Vectorizer (For comparison)
print("Computing CountVectorizer similarity matrix...")
cv = CountVectorizer(max_features=5000, stop_words='english')
cv_vectors = cv.fit_transform(new_df['tags']).toarray()
similarity_cv = cosine_similarity(cv_vectors)

# Save model files
print("Saving model files to model/ folder...")
pickle.dump(new_df, open('model/movie_dict.pkl', 'wb'))
pickle.dump(similarity_tfidf, open('model/similarity_tfidf.pkl', 'wb'))
pickle.dump(similarity_cv, open('model/similarity_cv.pkl', 'wb'))

# Also keep similarity.pkl pointing to TF-IDF as default fallback
pickle.dump(similarity_tfidf, open('model/similarity.pkl', 'wb'))

print("All models successfully built and saved!")