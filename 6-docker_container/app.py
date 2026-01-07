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

def get_clean_text(content):
    if content is None: return ""
    if isinstance(content, str): return content
    if isinstance(content, list):
        full_text = ""
        for item in content:
            if isinstance(item, dict):
                full_text += item.get('text', '')
                if 'content' in item:
                    full_text += get_clean_text(item['content'])
        return full_text
    if isinstance(content, dict):
        return get_clean_text(content.get('content') or content.get('text'))
    return str(content)

def format_history_for_engine(ui_history):
    engine_history = []
    for turn in ui_history:
        if isinstance(turn, dict):
            role = turn.get('role')
            content = get_clean_text(turn.get('content'))
            if role and content:
                engine_history.append({"role": role, "content": content})
    return engine_history

def add_user_message(user_text, history):
    if not user_text: return "", history
    if history is None: history = []
    history.append({"role": "user", "content": str(user_text)})
    return "", history

def bot_response(history):
    if not history: return history

    last_msg = history[-1]
    raw_content = last_msg.get('content', '')
    user_text = get_clean_text(raw_content)

    print(f"\n[LOG] User Question: {user_text}")

    cleaned = user_text.translate(str.maketrans('', '', string.punctuation)).strip().lower()

    if cleaned in ["hi", "hello", "hey", "greetings"]:
        history.append({"role": "assistant", "content": "Hoo-hoo! I am Judge's Familiar. Ask your ruling question."})
        yield history
        return

    if cleaned in ["thanks", "thank you", "cool"]:
        history.append({"role": "assistant", "content": "You are welcome. Proceed."})
        yield history
        return

    engine_history = format_history_for_engine(history[:-1])
    history.append({"role": "assistant", "content": ""})
    
    stream = judge.query(user_text, history=engine_history)
    
    for token in stream:
        delta = str(token.delta)
        history[-1]['content'] += delta
        yield history

# --- 3. UI BUILDER ---

custom_css = """
.gradio-container { background-color: #020617 !important; }
.prose, .markdown-text, p, span, label { color: #f8fafc !important; }
a { color: #60a5fa !important; } 
#logo_img {
    display: block; margin-left: 20px; margin-top: 20px; width: 30%; max-width: 300px;
    background-color: transparent !important; border: none !important;
}
.image-container .upload-button, .image-container .clear-button { display: none !important; }
#chatbot {
    background-color: #0f172a !important; border: 2px solid #3b82f6 !important;
    border-radius: 8px; box-shadow: 0 0 20px rgba(59, 130, 246, 0.15);
}
.message.user { background-color: #1e40af !important; border: 1px solid #3b82f6 !important; color: white !important; }
.message.bot { background-color: #1e293b !important; border: 1px solid #fbbf24 !important; color: #f1f5f9 !important; }
textarea, input { background-color: #1e293b !important; color: white !important; border: 1px solid #475569 !important; }
button.primary {
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
    color: #fbbf24 !important; border: 1px solid #fbbf24 !important;
    font-weight: bold; text-transform: uppercase; letter-spacing: 1px;
}
button.primary:hover { box-shadow: 0 0 15px rgba(251, 191, 36, 0.5); }
"""

theme = gr.themes.Base(primary_hue="blue", secondary_hue="amber", neutral_hue="slate")

with gr.Blocks(title="Judge's Familiar") as app:
    gr.Image(value="logo.png", elem_id="logo_img", show_label=False, container=False, interactive=False)
    
    chatbot = gr.Chatbot(elem_id="chatbot", height=600)
    
    with gr.Row():
        audio_input = gr.Audio(sources=["microphone"], type="filepath", show_label=False, container=False)
        msg_box = gr.Textbox(scale=4, placeholder="Type your question here or record audio...", show_label=False, autofocus=True, container=False)
        send_btn = gr.Button("Ask the Owl", scale=1, variant="primary")

    clear_btn = gr.ClearButton([msg_box, chatbot, audio_input])

    audio_input.stop_recording(fn=transcribe, inputs=audio_input, outputs=msg_box)
    msg_box.submit(fn=add_user_message, inputs=[msg_box, chatbot], outputs=[msg_box, chatbot]).then(fn=bot_response, inputs=chatbot, outputs=chatbot)
    send_btn.click(fn=add_user_message, inputs=[msg_box, chatbot], outputs=[msg_box, chatbot]).then(fn=bot_response, inputs=chatbot, outputs=chatbot)

if __name__ == "__main__":
    app.launch(share=False, theme=theme, css=custom_css)
