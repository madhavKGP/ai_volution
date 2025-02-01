import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI  # Correct import for OpenAI client

# Suppress symlink warnings (optional)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Set OpenAI API key
client = OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))

# Initialize FastAPI app
app = FastAPI()

# Define request and response models
class TextRequest(BaseModel):
    text: str
    language: str  # 'en' for English, 'hi' for Hindi

class TextResponse(BaseModel):
    enhanced_text: str

def enhance_style(text, language="en"):
    """
    Enhance the style of the text using OpenAI's GPT.
    """
    try:
        if language == "hi":
            # Explicitly ask OpenAI to return the response in Devanagari script
            prompt = f"Correct the grammar and enhance the following Hindi text for better readability and professionalism. Ensure the response is in Devanagari script: {text}"
        else:
            prompt = f"Correct the grammar and enhance the following English text for better readability and professionalism: {text}"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that enhances the style of text."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=100,
            temperature=0.7,
        )
        enhanced_text = response.choices[0].message.content.strip()
        return enhanced_text
    except Exception as e:
        print(f"Error in style enhancement: {e}")
        raise HTTPException(status_code=500, detail="Error enhancing text")

# Endpoint for text enhancement
@app.post("/enhance-text/", response_model=TextResponse)
async def enhance_text(request: TextRequest):
    """
    Endpoint to enhance the style and grammar of the input text.
    """
    # Validate language
    if request.language not in ["en", "hi"]:
        raise HTTPException(status_code=400, detail="Invalid language. Use 'en' for English or 'hi' for Hindi.")
    
    # Enhance the text
    enhanced_text = enhance_style(request.text, request.language)
    return {"enhanced_text": enhanced_text}
