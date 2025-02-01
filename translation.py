from fastapi import FastAPI
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer, pipeline

# Suppress symlink warnings (optional)
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Initialize FastAPI app
app = FastAPI()

# Define request and response models
class TranslationRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    translated_text: str

class GrammarCorrectionRequest(BaseModel):
    text: str
    lang: str  # 'en' for English, 'hi' for Hindi

class GrammarCorrectionResponse(BaseModel):
    corrected_text: str

# Load Hindi-to-English model
hi_to_en_tokenizer = MarianTokenizer.from_pretrained(r"C:\Users\ACER\Desktop\python\AI_Volution\models\opus-mt-hi-en")
hi_to_en_model = MarianMTModel.from_pretrained(r"C:\Users\ACER\Desktop\python\AI_Volution\models\opus-mt-hi-en")

# Load English-to-Hindi model
en_to_hi_tokenizer = MarianTokenizer.from_pretrained(r"C:\Users\ACER\Desktop\python\AI_Volution\models\opus-mt-en-hi")
en_to_hi_model = MarianMTModel.from_pretrained(r"C:\Users\ACER\Desktop\python\AI_Volution\models\opus-mt-en-hi")

# Load English grammar correction model
english_corrector = pipeline(
    "text2text-generation",
    model=r"C:\Users\ACER\Desktop\python\AI_Volution\models\grammar_error_correcter_v1"
)

def translate(text, tokenizer, model):
    """
    Translate text using the given tokenizer and model.
    
    Args:
        text (str): Input text to translate.
        tokenizer: Loaded tokenizer.
        model: Loaded model.
    
    Returns:
        str: Translated text.
    """
    # Tokenize the input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Generate the translation
    outputs = model.generate(**inputs, max_length=400)
    
    # Decode the output
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

def correct_english(text):
    """
    Correct grammar in English text.
    
    Args:
        text (str): Input English text to correct.
    
    Returns:
        str: Corrected English text.
    """
    try:
        corrected_text = english_corrector(text, max_length=500)[0]['generated_text']
        return corrected_text
    except Exception as e:
        print(f"Error in English grammar correction: {e}")
        return text  # Return original text if correction fails

def correct_hindi(text):
    """
    Correct grammar in Hindi text using translation-based fallback.
    
    Args:
        text (str): Input Hindi text to correct.
    
    Returns:
        str: Corrected Hindi text.
    """
    try:
        # Step 1: Translate Hindi to English
        english_text = translate(text, hi_to_en_tokenizer, hi_to_en_model)
        
        # Step 2: Correct grammar in English
        corrected_english = correct_english(english_text)
        
        # Step 3: Translate corrected English back to Hindi
        corrected_hindi = translate(corrected_english, en_to_hi_tokenizer, en_to_hi_model)
        return corrected_hindi
    except Exception as e:
        print(f"Error in Hindi grammar correction: {e}")
        return text  # Return original text if correction fails
    
@app.get("/")
async def root():
    return {"message": "Welcome to the AI_Volution Translation API!"}

# Endpoint for Hindi-to-English translation with grammar correction
@app.post("/translate-hi-to-en/", response_model=TranslationResponse)
async def translate_hi_to_en(request: TranslationRequest):
    # Correct grammar in Hindi input
    corrected_hindi = correct_hindi(request.text)
    
    # Translate corrected Hindi to English
    translated_text = translate(corrected_hindi, hi_to_en_tokenizer, hi_to_en_model)
    return {"translated_text": translated_text}

# Endpoint for English-to-Hindi translation with grammar correction
@app.post("/translate-en-to-hi/", response_model=TranslationResponse)
async def translate_en_to_hi(request: TranslationRequest):
    # Correct grammar in English input
    corrected_english = correct_english(request.text)
    
    # Translate corrected English to Hindi
    translated_text = translate(corrected_english, en_to_hi_tokenizer, en_to_hi_model)
    return {"translated_text": translated_text}

# Endpoint for standalone grammar correction
@app.post("/correct-grammar/", response_model=GrammarCorrectionResponse)
async def correct_grammar(request: GrammarCorrectionRequest):
    if request.lang == "en":
        corrected_text = correct_english(request.text)
    elif request.lang == "hi":
        corrected_text = correct_hindi(request.text)
    else:
        raise ValueError("Unsupported language. Use 'en' for English or 'hi' for Hindi.")
    return {"corrected_text": corrected_text}

# Endpoint for standalone Hindi grammar correction
@app.post("/correct-hindi-grammar/", response_model=GrammarCorrectionResponse)
async def correct_hindi_grammar(request: GrammarCorrectionRequest):
    if request.lang != "hi":
        raise ValueError("This endpoint is only for Hindi grammar correction. Use 'hi' as the language code.")
    
    corrected_text = correct_hindi(request.text)
    return {"corrected_text": corrected_text}
