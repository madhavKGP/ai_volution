from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai

import os
# importing necessary functions from dotenv library
from dotenv import load_dotenv, dotenv_values 

# Initialize FastAPI app
app = FastAPI()

# Set up your OpenAI API key
client = openai.OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))  # Replace with your actual OpenAI API key

# Define request model
class SpeechRequest(BaseModel):
    topic: str
    audience: str
    duration: int  # Duration in minutes

# Function to estimate word count based on duration
def estimate_word_count(duration: int) -> int:
    return duration * 150

@app.get("/")
async def root():
    return {"message": "Welcome to the AI_Volution Speech Generator API!"}

# Endpoint to generate a speech
@app.post("/generate-speech/")
async def generate_speech(request: SpeechRequest):
    try:
        # Extract inputs
        topic = request.topic
        audience = request.audience
        duration = request.duration

        # Estimate word count based on duration
        n_words = estimate_word_count(duration)

        # Step 1: Identify knowledge gaps
        problem_prompt = f"""
        Identify 3 key problems / knowledge gaps / misconceptions that {audience} might have about {topic}.
        Format your response as a bullet list:
        - Knowledge Gap 1
        - Knowledge Gap 2
        - Knowledge Gap 3
        """
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that identifies knowledge gaps."},
                {"role": "user", "content": problem_prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        problems = response.choices[0].message.content

        # Step 2: Generate speech outline
        outline_prompt = f"""
        Based on the following knowledge gaps, create a detailed speech outline in the problem-solution-action format:
        {problems}

        Structure the outline as follows:
        1. Problem: Describe the core issue and its significance.
        2. Solution: Propose realistic solutions to address the problem.
        3. Action: Suggest actionable steps the audience can take.

        Use clear headings and sub-points for each section.
        """
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates structured outlines for speeches."},
                {"role": "user", "content": outline_prompt}
            ],
            max_tokens=400,
            temperature=0.7
        )
        outline = response.choices[0].message.content

        # Step 3: Generate full speech
        speech_prompt = f"""
        Using the following outline, write a full speech script in under {n_words} words, Every thing should strictly be in markdown format:
        {outline}

        Ensure the speech is engaging, uses storytelling techniques, and includes a clear call to action.
        """
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that writes impactful speeches."},
                {"role": "user", "content": speech_prompt}
            ],
            max_tokens=600,
            temperature=0.7
        )
        speech = response.choices[0].message.content

        # Return the generated speech
        return {
            "knowledge_gaps": problems,
            "speech_outline": outline,
            "full_speech": speech
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))