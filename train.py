import pandas as pd
import joblib
import ast
from sentence_transformers import SentenceTransformer

print("🎬 Loading main metadata and credits...")
df_meta = pd.read_csv("data/movies_metadata.csv", low_memory=False)
df_credits = pd.read_csv("data/credits.csv")

print("🧹 Cleaning data and merging files...")
df_meta = df_meta.dropna(subset=['id', 'title', 'overview'])
df_meta['id'] = pd.to_numeric(df_meta['id'], errors='coerce')
df_credits['id'] = pd.to_numeric(df_credits['id'], errors='coerce')
df = df_meta.merge(df_credits, on='id')

allowed_languages = ['en', 'hi', 'ta', 'te', 'ml', 'kn', 'bn', 'mr']
df = df[df['original_language'].isin(allowed_languages)]
df = df.drop_duplicates(subset=['title']).reset_index(drop=True)

def get_director(crew_string):
    try:
        crew = ast.literal_eval(crew_string)
        for person in crew:
            if person['job'] == 'Director':
                return person['name']
        return ""
    except:
        return ""

def get_cast(cast_string):
    try:
        cast = ast.literal_eval(cast_string)
        names = [actor['name'] for actor in cast[:3]]
        return " ".join(names) if names else ""
    except:
        return ""

df['director'] = df['crew'].apply(get_director)
df['top_cast'] = df['cast'].apply(get_cast)

# We still combine the text so the neural net can read the names!
df['search_text'] = df['title'] + " " + df['overview'] + " " + df['director'] + " " + df['top_cast']
df['search_text'] = df['search_text'].fillna('')

print("🧠 Downloading the Neural Embedding Model (this only happens once)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print(f"🧮 Generating dense embeddings for {len(df)} movies... (This might take 1-2 minutes)")
# Instead of a math matrix, we generate dense 384-dimension neural vectors
movie_embeddings = model.encode(df['search_text'].tolist(), show_progress_bar=True)

print("💾 Saving the Neural Database...")
# Save the pre-computed neural vectors
joblib.dump(movie_embeddings, "movie_embeddings.pkl")
df[['title', 'overview', 'original_language', 'vote_average', 'director', 'top_cast', 'search_text']].to_csv("data/processed_movies.csv", index=False)

print("✅ Enterprise-grade dataset ready!")