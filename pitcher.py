import os
import json
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

# 1. Load the secret keys from the .env file
load_dotenv(override=True)
print(f"🕵️ DEBUG: My key starts with: {str(os.environ.get('GROQ_API_KEY'))[:8]}")
# 2. Initialize the Groq client using the OpenAI SDK routing
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# 3. Define the Pydantic schemas for documentation and prompt injection
class MoviePitch(BaseModel):
    title: str = Field(description="The exact title of the movie.")
    pitch: str = Field(
        description="A compelling, 2-3 sentence personalized explanation of why this movie fits the user's specific request or mood.")
    vibe_tags: list[str] = Field(
        description="3 short, atmospheric keywords describing the film's energy (e.g., ['Gritty', 'Cerebral', 'Neon-soaked']).")

class RecommendationResponse(BaseModel):
    analysis: str = Field(description="A brief 1-sentence analytical overview of the user's requested cinematic mood.")
    recommendations: list[MoviePitch] = Field(
        description="The list of structured movie pitches tailored to the user's request.")

def generate_ai_pitches(user_mood: str, raw_ml_movies: list[dict]) -> dict:
    """
    Takes the user's input mood and the list of raw movies found by the ML model,
    and returns a structured JSON response via Groq.
    """

    # We must explicitly pass the Pydantic schema structure into the prompt for Groq
    schema_instructions = RecommendationResponse.model_json_schema()

    prompt = f"""
    You are an elite, highly knowledgeable Film Critic and Cinematic Sommelier.

    The user is looking for a movie matching this exact mood/request: "{user_mood}"

    Our machine learning algorithm has mathematically pre-filtered the database and found these potential matching movies:
    {json.dumps(raw_ml_movies, indent=2)}

    Your Task:
    Review these raw choices. You MUST write exactly 3 custom, highly tailored pitches that directly connect the movie's plot to the user's requested mood. Do not skip any movies! Make the user want to watch them right now based on what they asked for!
    IMPORTANT: You must output your response in pure JSON format that strictly adheres to the following schema:
    {json.dumps(schema_instructions, indent=2)}
    """

    print("🤖 Querying Groq API for lightning-fast structured pitches...")

    # Call LLaMA-3.1 via Groq with json_object enforcement
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful, expert cinematic AI. Always reply in strict JSON."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
    )

    # Extract the JSON string and parse it into a Python dictionary
    raw_json_string = response.choices[0].message.content
    return json.loads(raw_json_string)

# Local test to ensure it works before building the UI
if __name__ == "__main__":
    mock_ml_output = [
        {"title": "Inception",
         "overview": "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O."},
        {"title": "Interstellar",
         "overview": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival."}
    ]
    test_mood = "Something mind-bending that makes me question reality but has space stuff."

    try:
        structured_result = generate_ai_pitches(test_mood, mock_ml_output)
        print("\n🎉 Success! Structured Output Received from Groq:")
        print(json.dumps(structured_result, indent=2))
    except Exception as e:
        print(f"❌ Error during API call: {e}")