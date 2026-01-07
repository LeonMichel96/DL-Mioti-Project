import gradio as gr
import os
import string
from dotenv import load_dotenv
from openai import OpenAI
from engine import MagicJudgeEngine

# --- 1. SETUP ---
if not load_dotenv():
    load_dotenv("../.env")

print(f"Awakening the Owl...")

client = OpenAI()
judge = MagicJudgeEngine(force_rebuild=False)

# --- 2. LOGIC ---

def transcribe(audio_path):
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
    if not history: return history

    # --- ROBUST CONTENT EXTRACTION ---
    last_msg = history[-1]
    raw_content = last_msg.get("content", "")

    user_text = ""
    
    if isinstance(raw_content, list):
        for item in raw_content:
            if isinstance(item, dict) and "text" in item:
                user_text += item["text"]
            elif isinstance(item, str):
                user_text += item
    else:
        user_text = str(raw_content)

    print(f"\n[LOG] User Question: {user_text}")

    cleaned = user_text.translate(str.maketrans('', '', string.punctuation)).strip().lower()

    # --- MINIMAL FILTERS ---
    greetings = ["hi", "hello", "hey", "greetings"]
    if cleaned in greetings:
        history.append({"role": "assistant", "content": "Hoo-hoo! I am Judge's Familiar. Present your query regarding Magic: The Gathering rules, and be precise."})
        yield history
        return

    gratitude = ["thanks", "thank you", "thx", "cool"]
    if cleaned in gratitude:
        history.append({"role": "assistant", "content": "You are welcome. Proceed."})
        yield history
        return

    # --- ENGINE QUERY ---
    
    history.append({"role": "assistant", "content": ""})
    
    stream = judge.query(user_text, history=history[:-1])
    
    full_buffer = ""
    is_thinking = True
    
    for token in stream:
        full_buffer += token.delta
        
        # --- CHAIN OF THOUGHT PARSING ---
        if "### FINAL ANSWER ###" in full_buffer:
            if is_thinking:
                is_thinking = False
                print(" >> [End of Thoughts]\n")
            
            # Extract ONLY what comes after the marker for the UI
            visible_part = full_buffer.split("### FINAL ANSWER ###")[-1].lstrip()
            history[-1]["content"] = visible_part
            yield history
            
        elif is_thinking:
            # Print thoughts to terminal for debugging
            print(token.delta, end="", flush=True)
            
        else:
            # Normal streaming after the marker
            history[-1]["content"] += token.delta
            yield history

# --- 3. UI BUILDER (JUDGE'S FAMILIAR THEME) ---

custom_css = """
/* --- JUDGE'S FAMILIAR PALETTE --- */
.gradio-container {
    background-color: #020617 !important; /* Midnight Black */
}

/* General Text */
.prose, .markdown-text, p, span, label {
    color: #f8fafc !important;
}

/* Subheader / Links */
a { color: #60a5fa !important; } 

/* LOGO STYLING (UPDATED: Smaller & Left Aligned) */
#logo_img {
    display: block;
    margin-left: 20px; /* Add slight padding from left edge */
    margin-top: 20px;  /* Add slight top padding */
    width: 30%;        /* Much smaller width */
    max-width: 300px;  /* Smaller max width */
    background-color: transparent !important;
    border: none !important;
}
/* Hide the image container toolbar */
.image-container .upload-button, .image-container .clear-button {
    display: none !important;
}

/* Chatbot Container */
#chatbot {
    background-color: #0f172a !important; /* Slate-900 */
    border: 2px solid #3b82f6 !important; /* Electric Blue Border */
    border-radius: 8px;
    box-shadow: 0 0 20px rgba(59, 130, 246, 0.15);
}

/* User Bubble */
.message.user {
    background-color: #1e40af !important; /* Blue-800 */
    border: 1px solid #3b82f6 !important;
    color: white !important;
}

/* Bot Bubble */
.message.bot {
    background-color: #1e293b !important; /* Slate-800 */
    border: 1px solid #fbbf24 !important; /* Gold Border */
    color: #f1f5f9 !important;
}

/* Input Box */
textarea, input {
    background-color: #1e293b !important;
    color: white !important;
    border: 1px solid #475569 !important;
}

/* Buttons */
button.primary {
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
    color: #fbbf24 !important; /* Gold Text */
    border: 1px solid #fbbf24 !important;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1px;
}
button.primary:hover {
    box-shadow: 0 0 15px rgba(251, 191, 36, 0.5);
}
"""

theme = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="amber",
    neutral_hue="slate"
)

with gr.Blocks(title="Judge's Familiar") as app:
    
    # LOGO REPLACEMENT
    # Ensure 'logo.png' is in the same folder as app.py
    gr.Image(
        value="logo.png", 
        elem_id="logo_img", 
        show_label=False, 
        container=False, 
        interactive=False
    )
    
    chatbot = gr.Chatbot(elem_id="chatbot", height=600)
    
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
        
        send_btn = gr.Button("Ask Judge", scale=1, variant="primary")

    clear_btn = gr.ClearButton([msg_box, chatbot, audio_input])

    # --- EVENTS ---
    
    audio_input.stop_recording(fn=transcribe, inputs=audio_input, outputs=msg_box)
    msg_box.submit(fn=add_user_message, inputs=[msg_box, chatbot], outputs=[msg_box, chatbot]).then(fn=bot_response, inputs=chatbot, outputs=chatbot)
    send_btn.click(fn=add_user_message, inputs=[msg_box, chatbot], outputs=[msg_box, chatbot]).then(fn=bot_response, inputs=chatbot, outputs=chatbot)

if __name__ == "__main__":
    app.launch(
        share=True,
        theme=theme,
        css=custom_css
    )
