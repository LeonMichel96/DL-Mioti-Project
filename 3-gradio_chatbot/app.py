import gradio as gr
import os
import string
from dotenv import load_dotenv
from openai import OpenAI
from engine import MagicJudgeEngine

# --- 1. SETUP ---
if not load_dotenv():
    load_dotenv("../.env")

print(f"Awakening the owl...")

client = OpenAI()
judge = MagicJudgeEngine(force_rebuild=False)

# --- 2. LOGIC ---

def transcribe(audio_path):
    """Transcribes audio using OpenAI Whisper API."""
    if not audio_path: return ""
    try:
        with open(audio_path, "rb") as f:
            return client.audio.transcriptions.create(model="whisper-1", file=f).text
    except Exception as e:
        return f"[Transcription Error: {e}]"

def add_user_message(user_text, history):
    """Adds user message to history."""
    if not user_text:
        return "", history
    if history is None:
        history = []
    
    # Append new message
    history.append({"role": "user", "content": str(user_text)})
    return "", history

def bot_response(history):
    """Generates response."""
    if not history: return history

    # --- ROBUST CONTENT EXTRACTION ---
    last_msg = history[-1]
    raw_content = last_msg.get("content", "")

    user_text = ""
    
    # Handle Gradio 6.0 Multimodal format (List)
    if isinstance(raw_content, list):
        for item in raw_content:
            if isinstance(item, dict) and "text" in item:
                user_text += item["text"]
            elif isinstance(item, str):
                user_text += item
    # Handle Standard string format
    else:
        user_text = str(raw_content)

    print(f"[DEBUG] Processed Input: {user_text}")

    cleaned = user_text.translate(str.maketrans('', '', string.punctuation)).strip().lower()

    # --- MINIMAL FILTERS ---
    
    # Greetings
    greetings = ["hi", "hello", "hey", "greetings"]
    if cleaned in greetings:
        history.append({"role": "assistant", "content": "Hello! I am Judge's Familiar. Ask me about a Magic card interaction!"})
        yield history
        return

    # Gratitude
    gratitude = ["thanks", "thank you", "thx", "cool"]
    if cleaned in gratitude:
        history.append({"role": "assistant", "content": "You're welcome!"})
        yield history
        return

    # --- ENGINE QUERY ---
    
    history.append({"role": "assistant", "content": ""})
    
    stream = judge.query(user_text)
    
    for token in stream:
        history[-1]["content"] += token.delta
        yield history

# --- 3. UI BUILDER ---

custom_css = """
#chatbot {min_height: 500px;}
footer {visibility: hidden}
"""

# WARNING FIX: 'css' parameter removed from here
with gr.Blocks(title="Judge's Familiar") as app:
    
    gr.Markdown("# 🦉 Judge's Familiar\n*Level 2 Magic: The Gathering Rules Engine*")
    
    chatbot = gr.Chatbot(elem_id="chatbot", height=500)
    
    with gr.Row():
        audio_input = gr.Audio(
            sources=["microphone"], 
            type="filepath", 
            show_label=False,
            container=False
        )
        
        msg_box = gr.Textbox(
            scale=4, 
            placeholder="Type your question here or record audio...", 
            show_label=False, 
            autofocus=True,
            container=False
        )
        
        send_btn = gr.Button("Ask", scale=1, variant="primary")

    clear_btn = gr.ClearButton([msg_box, chatbot, audio_input])

    # --- EVENTS ---
    
    audio_input.stop_recording(
        fn=transcribe, inputs=audio_input, outputs=msg_box
    )

    msg_box.submit(
        fn=add_user_message, inputs=[msg_box, chatbot], outputs=[msg_box, chatbot]
    ).then(
        fn=bot_response, inputs=chatbot, outputs=chatbot
    )
    
    send_btn.click(
        fn=add_user_message, inputs=[msg_box, chatbot], outputs=[msg_box, chatbot]
    ).then(
        fn=bot_response, inputs=chatbot, outputs=chatbot
    )

if __name__ == "__main__":
    app.launch(
        share=True,
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
        css=custom_css
    )
