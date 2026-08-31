import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # project root
CSV_PATH = str(BASE_DIR / "doctor_availability.csv")

XAI_API_KEY = os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "grok-4")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))

# --- Groq (temporary, for testing while xAI credits are exhausted) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Active provider: "groq" (testing) or "xai" (production / grok-4).
# Flip this back to "xai" when credits are restored.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "xai")

VALID_SPECIALIZATIONS = [
    "general_dentist",
    "oral_surgeon",
    "orthodontist",
    "cosmetic_dentist",
    "prosthodontist",
    "pediatric_dentist",
    "emergency_dentist",
]

VALID_DOCTORS = [
    "john doe",
    "emily johnson",
    "sarah wilson",
    "jane smith",
    "michael green",
    "robert martinez",
    "lisa brown",
    "susan davis",
    "daniel miller",
    "kevin anderson",
]

DATE_FORMAT = "%m/%d/%Y %H:%M"
