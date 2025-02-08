from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Set up your Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is missing. Please check your .env file.")

# Initialize ChatGroq LLM (model is initialized once)
chat_model = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)

# Function to estimate word count based on duration
def estimate_word_count(duration: int) -> int:
    return duration * 150

# Request and Response Models for Translation and Enhancement
class LanguageDetectionRequest(BaseModel):
    text: str  # Input text to detect language for

class TranslationRequest(BaseModel):
    text: str  # Input text to translate
    target_lang: str  # Target language code (e.g., "hi", "en")

class TranslationResponse(BaseModel):
    translated_text: str

class TextEnhancementRequest(BaseModel):
    text: str  # Input text to enhance
    lang: str  # 'en' for English, 'hi' for Hindi

class TextEnhancementResponse(BaseModel):
    enhanced_text: str

# General Speech Request Model
class SpeechRequest(BaseModel):
    topic: str
    audience: str
    duration: int  # Duration in minutes

# Inspirational Storytelling Speech Request Model
class InspirationalStorytellingRequest(BaseModel):
    story_theme: str
    audience: str
    key_takeaways: List[str]  # Lessons or messages to convey
    duration: int  # Duration in minutes

# Award Acceptance Speech Request Model
class AwardAcceptanceSpeechRequest(BaseModel):
    award_name: str
    recipient_name: str
    people_to_thank: List[str]  # Names or groups to thank
    achievements: List[str]  # Key accomplishments to mention
    duration: int  # Duration in minutes

# Farewell Speech Request Model
class FarewellSpeechRequest(BaseModel):
    event_context: str
    audience: str
    key_memories: List[str]  # Memorable moments to include
    words_of_gratitude: List[str]  # People or things to thank
    duration: int  # Duration in minutes

# Educational Speech Request Model
class EducationalSpeechRequest(BaseModel):
    topic: str
    audience: str
    key_points: Optional[List[str]] = None  # Optional list of key points
    duration: int  # Duration in minutes

# Product Launch Speech Request Model
class ProductLaunchSpeechRequest(BaseModel):
    product_name: str
    features: List[str]  # Key features of the product
    target_audience: str
    call_to_action: str  # Action you want the audience to take
    duration: int  # Duration in minutes

@app.get("/")
async def root():
    return {
        "message": "Welcome to the AI_Volution API!",
        "endpoints": {
            "/detect-language/": "Detect the language of input text.",
            "/translate/": "Translate text from one language to another.",
            "/enhance-text/": "Enhance the grammar and style of input text.",
            "/generate-speech/": "Generate a general-purpose speech.",
            "/generate-educational-speech/": "Generate an educational speech.",
            "/generate-product-launch-speech/": "Generate a product launch speech.",
            "/generate-inspirational-storytelling-speech/": "Generate an inspirational storytelling speech.",
            "/generate-award-acceptance-speech/": "Generate an award acceptance speech.",
            "/generate-farewell-speech/": "Generate a farewell speech."
        }
    }

# Endpoint for language detection
@app.post("/detect-language/", response_model=dict)
async def detect_language(request: LanguageDetectionRequest):
    try:
        # Define prompt for language detection
        language_detection_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that detects the language of text."),
            ("user", f"""
            Detect the language of the following text and return only the ISO 639-1 language code (e.g., 'en', 'hi'):
            {request.text}
            """)
        ])
        language_detection_chain = LLMChain(llm=chat_model, prompt=language_detection_prompt)
        detected_lang = language_detection_chain.run({}).strip().lower()
        return {"detected_language": detected_lang}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for real-time translation with Groq
@app.post("/translate/", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    try:
        # Detect the source language
        language_detection_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that detects the language of text."),
            ("user", f"""
            Detect the language of the following text and return only the ISO 639-1 language code (e.g., 'en', 'hi'):
            {request.text}
            """)
        ])
        language_detection_chain = LLMChain(llm=chat_model, prompt=language_detection_prompt)
        source_lang = language_detection_chain.run({}).strip().lower()
        # Define prompt for translation
        translation_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that translates text."),
            ("user", f"""
            Translate the following text from {source_lang} to {request.target_lang}. Return only the translated text without any additional explanations:
            {request.text}
            """)
        ])
        translation_chain = LLMChain(llm=chat_model, prompt=translation_prompt)
        translated_text = translation_chain.run({}).strip()
        return {"translated_text": translated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for text enhancement with Groq
@app.post("/enhance-text/", response_model=TextEnhancementResponse)
async def enhance_text(request: TextEnhancementRequest):
    try:
        # Define prompt for text enhancement
        if request.lang == "hi":
            enhancement_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that enhances the style of Hindi text."),
                ("user", f"""
                Correct the grammar and enhance the following Hindi text for better readability and professionalism. Ensure the response is in Devanagari script. Return only the enhanced text without any additional explanations:
                {request.text}
                """)
            ])
        else:
            enhancement_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that enhances the style of English text."),
                ("user", f"""
                Correct the grammar and enhance the following English text for better readability and professionalism. Return only the enhanced text without any additional explanations:
                {request.text}
                """)
            ])
        enhancement_chain = LLMChain(llm=chat_model, prompt=enhancement_prompt)
        enhanced_text = enhancement_chain.run({}).strip()
        return {"enhanced_text": enhanced_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# General-purpose speech generation endpoint
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
        problem_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that identifies knowledge gaps."),
            ("user", f"""
            Identify 3 key problems / knowledge gaps / misconceptions that {audience} might have about {topic}.
            Format your response as a bullet list:
            - Knowledge Gap 1
            - Knowledge Gap 2
            - Knowledge Gap 3
            """)
        ])
        problem_chain = LLMChain(llm=chat_model, prompt=problem_prompt)
        problems = problem_chain.run({})
        # Step 2: Generate speech outline
        outline_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that creates structured outlines for speeches."),
            ("user", f"""
            Based on the following knowledge gaps, create a detailed speech outline in the problem-solution-action format:
            {problems}
            Structure the outline as follows:
            1. Problem: Describe the core issue and its significance.
            2. Solution: Propose realistic solutions to address the problem.
            3. Action: Suggest actionable steps the audience can take.
            Use clear headings and sub-points for each section.
            """)
        ])
        outline_chain = LLMChain(llm=chat_model, prompt=outline_prompt)
        outline = outline_chain.run({})
        # Step 3: Generate full speech
        speech_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that writes impactful speeches."),
            ("user", f"""
            Using the following outline, write a full speech script in under {n_words} words. Everything should strictly be in markdown format:
            {outline}
            Ensure the speech is engaging, uses storytelling techniques, and includes a clear call to action.
            """)
        ])
        speech_chain = LLMChain(llm=chat_model, prompt=speech_prompt)
        speech = speech_chain.run({})
        # Return the generated speech
        return {
            "knowledge_gaps": problems,
            "speech_outline": outline,
            "full_speech": speech
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Inspirational Storytelling Speech Endpoint
@app.post("/generate-inspirational-storytelling-speech/")
async def generate_inspirational_storytelling_speech(request: InspirationalStorytellingRequest):
    try:
        # Extract inputs
        story_theme = request.story_theme
        audience = request.audience
        key_takeaways = ", ".join(request.key_takeaways)
        duration = request.duration
        # Estimate word count based on duration
        n_words = estimate_word_count(duration)
        # Define prompt for inspirational storytelling speech
        speech_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that writes inspirational storytelling speeches."),
            ("user", f"""
            Write an inspirational storytelling speech based on the following details:
            Story Theme: {story_theme}
            Audience: {audience}
            Key Takeaways: {key_takeaways}
            Duration: {duration} minutes
            Word Limit: {n_words} words
            Everything should strictly be in markdown format!!
            Structure the speech as follows:
            1. Introduction: Start with a hook to grab attention and introduce the story.
            2. Body: Narrate the story in detail, highlighting the challenges, actions, and outcomes.
            3. Conclusion: Summarize the key takeaways and end with an inspiring message.
        
            """)
        ])
        print("first00")
        speech_chain = LLMChain(llm=chat_model, prompt=speech_prompt)
        print("second")
        speech = speech_chain.run({})
        print("third");
        return {"speech": speech.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Award Acceptance Speech Endpoint
@app.post("/generate-award-acceptance-speech/")
async def generate_award_acceptance_speech(request: AwardAcceptanceSpeechRequest):
    try:
        # Extract inputs
        award_name = request.award_name
        recipient_name = request.recipient_name
        people_to_thank = ", ".join(request.people_to_thank)
        achievements = ", ".join(request.achievements)
        duration = request.duration
        # Estimate word count based on duration
        n_words = estimate_word_count(duration)
        # Define prompt for award acceptance speech
        speech_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that writes award acceptance speeches."),
            ("user", f"""
            Write an award acceptance speech based on the following details:
            Award Name: {award_name}
            Recipient Name: {recipient_name}
            People/Organizations to Thank: {people_to_thank}
            Key Achievements: {achievements}
            Duration: {duration} minutes
            Word Limit: {n_words} words
            Everything should strictly be in markdown format!!
            Structure the speech as follows:
            1. Introduction: Express gratitude and excitement about receiving the award.
            2. Body: Thank the people/organizations who supported you and highlight your achievements.
            3. Conclusion: End with a humble and inspiring message.
    
            """)
        ])
        speech_chain = LLMChain(llm=chat_model, prompt=speech_prompt)
        speech = speech_chain.run({})
        return {"speech": speech.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Farewell Speech Endpoint
@app.post("/generate-farewell-speech/")
async def generate_farewell_speech(request: FarewellSpeechRequest):
    try:
        # Extract inputs
        event_context = request.event_context
        audience = request.audience
        key_memories = ", ".join(request.key_memories)
        words_of_gratitude = ", ".join(request.words_of_gratitude)
        duration = request.duration
        # Estimate word count based on duration
        n_words = estimate_word_count(duration)
        # Define prompt for farewell speech
        speech_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that writes heartfelt farewell speeches."),
            ("user", f"""
            Write a farewell speech based on the following details:
            Event Context: {event_context}
            Audience: {audience}
            Key Memories: {key_memories}
            Words of Gratitude: {words_of_gratitude}
            Duration: {duration} minutes
            Word Limit: {n_words} words
            Everything should strictly be in markdown format!!
            Structure the speech as follows:
            1. Introduction: Acknowledge the occasion and express gratitude.
            2. Body: Share key memories and experiences, and thank those who made an impact.
            3. Conclusion: End with a hopeful and emotional farewell message.

            """)
        ])
        speech_chain = LLMChain(llm=chat_model, prompt=speech_prompt)
        speech = speech_chain.run({})
        return {"speech": speech.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-educational-speech/")
async def generate_educational_speech(request: EducationalSpeechRequest):
    try:
        # Extract inputs
        topic = request.topic
        audience = request.audience
        key_points = ", ".join(request.key_points) if request.key_points else "Not specified"
        duration = request.duration
        # Estimate word count based on duration
        n_words = estimate_word_count(duration)
        # Define prompt for educational speech
        speech_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that writes educational speeches."),
            ("user", f"""
            Write an educational speech based on the following details:
            Topic: {topic}
            Audience: {audience}
            Key Points to Cover: {key_points}
            Duration: {duration} minutes
            Word Limit: {n_words} words
            Everything should strictly be in markdown format!!
            Structure the speech as follows:
            1. Introduction: Start with a hook to grab attention, provide context, and state the purpose.
            2. Body: Discuss the key points in detail, using examples and explanations.
            3. Conclusion: Summarize the main points and end with a memorable closing statement.
            """)
        ])
        speech_chain = LLMChain(llm=chat_model, prompt=speech_prompt)
        speech = speech_chain.run({})
        return {"speech": speech.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-product-launch-speech/")
async def generate_product_launch_speech(request: ProductLaunchSpeechRequest):
    try:
        # Extract inputs
        product_name = request.product_name
        features = ", ".join(request.features)
        target_audience = request.target_audience
        call_to_action = request.call_to_action
        duration = request.duration
        # Estimate word count based on duration
        n_words = estimate_word_count(duration)
        # Define prompt for product launch speech
        speech_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that writes product launch speeches."),
            ("user", f"""
            Write a product launch speech based on the following details:
            Product Name: {product_name}
            Key Features: {features}
            Target Audience: {target_audience}
            Call-to-Action: {call_to_action}
            Duration: {duration} minutes
            Word Limit: {n_words} words
            Everything should strictly be in markdown format!!
            Structure the speech as follows:
            1. Introduction: Start with a hook to grab attention and introduce the product.
            2. Body: Highlight the key features and benefits of the product, and explain how it solves a problem.
            3. Conclusion: End with a strong call-to-action and a memorable closing statement.
            """)
        ])
        speech_chain = LLMChain(llm=chat_model, prompt=speech_prompt)
        speech = speech_chain.run({})
        return {"speech": speech.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

