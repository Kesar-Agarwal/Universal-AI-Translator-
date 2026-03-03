import streamlit as st
import whisper
import google.generativeai as genai
from gtts import gTTS
import tempfile
import os
import gc

# --- 1. SETTINGS (MUST BE FIRST COMMAND) ---
st.set_page_config(page_title="Universal AI Translator", page_icon="🌍", layout="centered")

# --- 2. API & MODEL SETUP ---
# Securely fetch API key from Streamlit Secrets
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY_N"]
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel(
        model_name='gemini-2.5-flash',
        generation_config={"max_output_tokens": 150, "temperature": 0.1}
    )
except Exception:
    st.error("⚠️ Google API Key not found. Please add 'GOOGLE_API_KEY_N' to your Secrets.")
    st.stop()

@st.cache_resource
def load_whisper():
    # 'tiny' is mandatory for stable CPU performance
    return whisper.load_model("tiny")

whisper_model = load_whisper()

# --- 3. STATE & LOGIC ---
LANGUAGES = {
    "Auto-Detect": "auto", "English": "en", "Hindi": "hi", "Spanish": "es", 
    "French": "fr", "German": "de", "Japanese": "ja", "Arabic": "ar"
}

if 'src_lang' not in st.session_state: st.session_state.src_lang = "Hindi"
if 'target_lang' not in st.session_state: st.session_state.target_lang = "English"

def swap_languages():
    if st.session_state.src_lang != "Auto-Detect":
        st.session_state.src_lang, st.session_state.target_lang = \
            st.session_state.target_lang, st.session_state.src_lang

# --- 4. UI DESIGN ---
st.title("🌍 Universal AI Translator")
st.caption("Voice-to-Voice • Powered by Whisper & Gemini • Instant AI Bridge")

# Language Selectors with Native Alignment
col_from, col_swap, col_to = st.columns([4, 1, 4], vertical_alignment="bottom")

with col_from:
    src_lang = st.selectbox("From:", list(LANGUAGES.keys()), key="src_lang")
with col_swap:
    st.button("🔄", on_click=swap_languages, use_container_width=True)
with col_to:
    target_options = [l for l in LANGUAGES.keys() if l != "Auto-Detect"]
    target_lang = st.selectbox("To:", target_options, key="target_lang")

st.divider()

# Input Methods
col_mic, col_up = st.columns(2)
with col_mic:
    # Native mic popup works on HTTPS (Streamlit Cloud/HF)
    audio_mic = st.audio_input("🎤 Record your voice") if hasattr(st, "audio_input") else None
with col_up:
    audio_file = st.file_uploader("📤 Or upload audio", type=["mp3", "wav", "m4a"])

# --- 5. THE ENGINE ---
final_path = None
if audio_mic:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_mic.getvalue())
        final_path = tmp.name
elif audio_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.getvalue())
        final_path = tmp.name

if final_path:
    gc.collect() # Immediate RAM cleanup
    with st.status("🚀 Processing AI Magic...") as status:
        try:
            # Step A: Whisper Transcription
            st.write("👂 Whisper is decoding (Tiny model)...")
            result = whisper_model.transcribe(final_path, fp16=False)
            original_text = result.get("text", "").strip()
            
            if not original_text:
                st.error("No speech detected. Please try again.")
                st.stop()
                
            # Step B: Gemini Translation
            status.update(label="✍️ Translating with Gemini...", state="running")
            prompt = f"Translate the following text to {target_lang}. Output ONLY the translated text: {original_text}"
            response = gemini_model.generate_content(prompt)
            translated_text = response.text.strip()
            
            # Step C: Voice Generation
            status.update(label="🔊 Generating Voice...", state="running")
            tts = gTTS(text=translated_text, lang=LANGUAGES[target_lang])
            tts_path = "output.mp3"
            tts.save(tts_path)
            
            status.update(label="✅ Success!", state="complete")
            
            # Show Results
            st.chat_message("user").write(f"**Original:** {original_text}")
            st.chat_message("assistant").write(f"**Translated:** {translated_text}")
            st.audio(tts_path, autoplay=True)
            
        except Exception as e:
            st.error(f"Engine Error: {e}")
        finally:
            if os.path.exists(final_path):
                os.remove(final_path)
else:
    st.info("💡 Tap the microphone to speak or upload a file to start translating.")
          
