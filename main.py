import sys
from langchain.chat_models import init_chat_model
import os
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("openai:gpt-5-nano", api_key=os.getenv("OPENAI_API_KEY"))

# Résolution de chemin compatible PyInstaller (_MEIPASS) et exécution script
BASE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
SYSTEM_PATH = os.path.join(BASE_DIR, "PROMPT_SYSTEM.txt")

# Lecture dynamique du prompt système
system_content = open(SYSTEM_PATH, "r", encoding="utf-8").read().strip()
messages = [SystemMessage(system_content)]

# Boucle de conversation avec mémoire en RAM (liste de messages)
while True:
    user_text = input("Vous: ").strip() + "\n"
    if not user_text or user_text.lower() in {"quit", "exit"}:
        break
    messages.append(HumanMessage(user_text))
    ai = model.invoke(messages)
    print("="*100 + "\n" + f"AI: {ai.content}" + "\n" + "="*100)
    messages.append(ai)

