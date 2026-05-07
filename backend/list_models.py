import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("Listing available models for your API key:\n")

# Iterate through all available models
for model in client.models.list():
    # Only show models that support text/image generation
    if 'generateContent' in model.supported_actions:
        print(f"Name: {model.name}")
        print(f"  > Version: {model.version}")
        print(f"  > Description: {model.description}")
        print("-" * 30)