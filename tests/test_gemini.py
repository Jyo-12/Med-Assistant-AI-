import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

print("API Key Found:", api_key is not None)

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-flash-latest")

print("Sending request...")

response = model.generate_content("Say hello in one sentence.")

print(response.text)