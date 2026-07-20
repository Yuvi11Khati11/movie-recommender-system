# CinemAI - Movie Recommendation System

# Project Overview
**CinemAI** is a premium, production-grade content-based movie recommendation system built with **Python**, **Streamlit**, **Scikit-Learn**, and the **NLTK** library. It uses PorterStemmer NLP tokenization, a dual recommendation engine (TF-IDF vs. CountVectorizer), a clean dark slate user interface, dynamic placeholder posters, and Explainable AI (XAI) feature matching.

---

# Features
- **🧠 Dual Vectorization Engines**: Switch between **TF-IDF Vectorizer** (default) and **CountVectorizer** dynamically in the sidebar. Compare how term weightings affect your recommendations.
- **⚡ PorterStemmer NLP Preprocessing**: Tags are processed using NLTK's PorterStemmer to reduce words to their linguistic base (e.g. *loving*, *loved* $\rightarrow$ *love*), improving recommendation accuracy.
- **🔍 Real-Time Autocomplete Search**: Text search updates matches instantly as you type, rendering compact vertical suggestion tags below the input box.
- **💡 Explainable AI (XAI)**: Displays why a movie is recommended by showing shared attributes (overlapping genres, keywords, cast members, or directors) as sleek color-coded badges.
- **📶 Robust Offline Mode**: Runs a connection check on startup with a fast 0.5s timeout. If TMDB is offline, the app enters **Offline Mode** instantly to bypass slow network timeouts.
- **🖼️ Curated Local Fallbacks**: If offline, movie ratings show as `"Rating Unavailable"`, and poster paths are mapped to one of **8 high-quality cinema-themed photography prints** from Unsplash using a modulo hash (`movie_id % 8`), keeping the offline UI looking beautiful.
- **🍿 Redesigned Offline Landing Page**: Features three dynamic local recomendation rows: *Editor Picks* (classics), *Popular Recommendations* (sorted locally by popularity score), and *Featured Recommendations* (similar to *Inception* computed locally).
- **❤️ Rich Favorites System**: Bookmarks movies to a JSON database containing title, poster art, TMDB rating, and ID metadata.
- **🕒 Sidebar Search History**: Tracks the last 10 unique searches locally for quick click-to-search access.

---

# Tech Stack
- **Core Languages**: Python
- **Frontend Framework**: Streamlit
- **Machine Learning**: Scikit-Learn (TF-IDF Vectorizer, CountVectorizer, Cosine Similarity)
- **Natural Language Processing**: NLTK (PorterStemmer)
- **API Request Handling**: Requests, Python-dotenv, Pandas

# Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/movie-recommender-system.git
cd movie-recommender-system
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Build Models (Required Setup Step)
> [!IMPORTANT]
> The pre-compiled similarity matrix files are **large (~184 MB each)**, exceeding GitHub's strict 100 MB file size limit. They are intentionally excluded from the online repository.
> 
> You **must** compile the dataset and build the similarity matrices locally before running the application:
```bash
python model_builder.py
```
This script preprocesses the dataset, stems tags using NLTK PorterStemmer, and compiles the TF-IDF and CountVectorizer similarity matrices in under 15 seconds.

### 5. Run the App
```bash
streamlit run app.py
```

---

# Environment Variables
The application supports secure, production-safe API key resolution. 

Create a `.env` file in the root directory:
```env
TMDB_API_KEY=your_tmdb_api_key_here
```
*(You can copy `.env.example` to `.env` as a template. If no key is configured, the application detects this instantly and runs in Offline Mode using local fallback data without crashing).*

For Streamlit Cloud deployment, add the secret under Advanced Settings:
```toml
TMDB_API_KEY = "your_tmdb_api_key"
```

---

# Project Structure
```text
movie-recommender-system/
├── assets/                     # Custom assets & screenshots
│   ├── screenshots/            # Directory for screenshots
│   ├── logo.png                # Generated branding logo
│   └── styles.css              # Custom dark-slate glassmorphism styles
│
├── dataset/                    # Kaggle source datasets
│   ├── tmdb_5000_credits.csv
│   └── tmdb_5000_movies.csv
│
├── model/                      # Precomputed pickle models (Git ignored)
│   ├── movie_dict.pkl          # Enriched dataframe pickle (4.7 MB)
│   ├── similarity_cv.pkl       # Count Vectorizer similarity matrix
│   └── similarity_tfidf.pkl    # TF-IDF similarity matrix (Default)
│
├── utils/                      # Modular backend backend modules
│   ├── api.py                  # TMDB cached API requests & offline fallbacks
│   ├── recommender.py          # Similarity engines and XAI explanation matches
│   ├── favorites.py            # JSON-backed Favorites manager
│   └── search_history.py       # JSON-backed search tracker
│
├── .env                        # Local configurations (Git ignored)
├── .env.example                # Sample environment variables template
├── .gitignore                  # Git tracking exclusion rules
├── app.py                      # Main Streamlit user interface
├── model_builder.py            # Preprocessing and vectorizer compiler pipeline
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

# Future Improvements
1. **👥 Collaborative Filtering**: Incorporate user rating vectors (e.g. SVD, ALS matrix factorization) to build a hybrid recommendation engine.
2. **🔐 User Authentication**: Support user profiles and auth (OAuth, Firebase) to keep favorites synced to a cloud database (Firestore/Postgres) instead of local files.
3. **💬 Sentiment Analysis**: Process live user comments and reviews using NLP (BERT/VADER) to score and filter recommended titles.

---
## 🚀 Live Demo

https://cinemai-recommender.streamlit.app/

# Author
Yuvi Khati
