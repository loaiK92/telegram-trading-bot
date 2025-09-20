# src/core/config.py
import os
from dotenv import load_dotenv

# This line finds and loads the .env file from the project root
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ALLTICK_API_KEY = os.getenv("ALLTICK_API_KEY")

# Basic validation to ensure keys are loaded
if not TELEGRAM_BOT_TOKEN or not ALLTICK_API_KEY:
    raise ValueError("API keys not found. Please check your .env file.")