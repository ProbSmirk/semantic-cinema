import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel, Field
import json
from pitcher import generate_ai_pitches, client
from sentence_transformers import SentenceTransformer

# --- Page Setup ---
st.set_page_config(page_title="Semantic Cinema", page_icon="🍿", layout="centered")
st.title("🍿 Semantic Cinema v2.0")
st.write("Hybrid Search powered by Vectors & Groq LLaMA.")


# --- Load ML Components ---
@st.cache_resource
def load_ml_components():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    movie_embeddings = joblib.load("movie_embeddings.pkl")
    movies_df = pd.read_csv("data/processed_movies.csv")
    # Ensure missing text doesn't break our name filters
    movies_df['director'] = movies_df['director'].fillna("")
    movies_df['top_cast'] = movies_df['top_cast'].fillna("")
    return model, movie_embeddings, movies_df


try:
    embed_model, database_embeddings, movies_df = load_ml_components()
except FileNotFoundError:
    st.error("⚠️ ML models not found! Make sure you ran train.py first.")
    st.stop()


# --- Define the Strict Schema for HYBRID Search ---
class SmartSearch(BaseModel):
    keywords: str = Field(description="5 to 7 cinematic keywords/genres (e.g., 'mafia crime gritty').")
    director: str = Field(default="",
                          description="The exact name of the director if mentioned (e.g., 'Martin Scorsese'). Leave empty if none.")
    actor: str = Field(default="",
                       description="The exact name of the actor if mentioned (e.g., 'Shahrukh Khan'). Leave empty if none.")


# --- UI Inputs ---
user_mood = st.text_area("What are you in the mood for?",
                         placeholder="e.g., A Martin Scorsese movie starring Robert De Niro...")

lang_map = {"Any": "any", "English": "en", "Hindi": "hi", "Tamil": "ta", "Telugu": "te", "Malayalam": "ml",
            "Kannada": "kn", "Bengali": "bn", "Marathi": "mr"}
selected_lang_name = st.selectbox("Preferred Language:", list(lang_map.keys()))
selected_lang_code = lang_map[selected_lang_name]

if st.button("Find My Movie 🎬"):
    if user_mood.strip() == "":
        st.warning("Please describe your mood first!")
    else:
        with st.spinner("🧠 AI is executing a Hybrid Search..."):

            # --- PHASE 1: Entity Extraction (LLM) ---
            schema_instructions = SmartSearch.model_json_schema()

            expansion_prompt = f"""
            The user searched for: "{user_mood}"
            Extract the cinematic vibe into keywords. If they mention a specific director or actor, extract those into their specific fields.

            IMPORTANT: Output pure JSON matching this schema:
            {json.dumps(schema_instructions)}
            """

            try:
                smart_search_response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": expansion_prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1  # Lower temp for better exact name extraction
                )

                smart_json = json.loads(smart_search_response.choices[0].message.content)
                smart_keywords = smart_json.get('keywords', '')
                target_director = smart_json.get('director', '')
                target_actor = smart_json.get('actor', '')

                st.info(f"**Vibe:** {smart_keywords} | **Dir:** {target_director} | **Cast:** {target_actor}")

            except Exception as e:
                st.error("Smart Search failed. Try a different prompt.")
                st.stop()

            # --- PHASE 2: HYBRID ML Search ---
            # Start by assuming all movies are valid
            mask = pd.Series(True, index=movies_df.index)

            # 1. Hard Filter by Language
            if selected_lang_code != "any":
                mask = mask & (movies_df['original_language'] == selected_lang_code)

            # 2. Hard Filter by Exact Names (This forces 100% accuracy on cast/director!)
            if target_director:
                mask = mask & movies_df['director'].str.contains(target_director, case=False, na=False)
            if target_actor:
                mask = mask & movies_df['top_cast'].str.contains(target_actor, case=False, na=False)

            # Apply the filters
            filtered_df = movies_df[mask].reset_index(drop=True)
            filtered_indices = movies_df.index[mask].tolist()
            filtered_matrix = database_embeddings[filtered_indices]

            if len(filtered_df) == 0:
                st.error("Could not find any movies in our database matching those exact people/languages.")
                st.stop()

            # 3. Vector Math (Sort whatever is left by the Vibe Keywords)
            user_vector = embed_model.encode([smart_keywords])
            similarities = cosine_similarity(user_vector, filtered_matrix).flatten()

            # Safely get up to 3 matches (in case the filter only found 1 or 2 movies)
            num_matches = min(3, len(filtered_df))
            top_indices = similarities.argsort()[-num_matches:][::-1]

            raw_matches = []
            for idx in top_indices:
                movie_data = filtered_df.iloc[idx]
                raw_matches.append({
                    "title": movie_data['title'],
                    "overview": movie_data['overview'],
                    "rating": movie_data['vote_average'],
                    "director": movie_data['director'],
                    "cast": movie_data['top_cast']
                })

            # --- PHASE 3: The GenAI Pitcher ---
            st.toast("Writing custom pitches...")
            try:
                ai_response = generate_ai_pitches(user_mood, raw_matches)

                st.success("Matches Found!")
                st.markdown(f"**AI Analysis:** *{ai_response['analysis']}*")
                st.divider()

                for pitch_data, raw_data in zip(ai_response['recommendations'], raw_matches):
                    st.subheader(f"🎞️ {pitch_data['title']}")
                    st.caption(f"**TMDB Rating:** ⭐ {raw_data['rating']}/10  |  **Director:** {raw_data['director']}")
                    st.caption(f"**Starring:** {raw_data['cast']}")
                    st.write(pitch_data['pitch'])

                    tags_html = "".join([
                                            f"<span style='background-color:#4A3060; color:white; padding:4px 8px; border-radius:12px; margin-right:8px; font-size:12px;'>{tag}</span>"
                                            for tag in pitch_data['vibe_tags']])
                    st.markdown(tags_html, unsafe_allow_html=True)
                    st.write("---")

            except Exception as e:
                st.error(f"Something went wrong with the AI: {e}")