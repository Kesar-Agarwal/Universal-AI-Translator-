import streamlit as st
import whisper
import torch
import google.generativeai as genai
from gtts import gTTS
import tempfile
import os
import gc

# --- 1. SETTINGS ---
st.set_page_config(page_title="SpeakSync AI", page_icon="🎙️", layout="centered")

st.markdown("<h1 style='text-align: center; color: violet;'>🎙️ SpeakSync AI</h1>", unsafe_allow_html=True)
st.caption("Speak Locally, Communicate Globally", unsafe_allow_html=True)
st.divider()

# --- 2. INITIALIZE HISTORY ---
if 'history' not in st.session_state:
    st.session_state.history = []
    
# --- 3. API & MODEL SETUP ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY_N"]
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
except Exception:
    st.error("⚠️ API Key missing in Secrets.")
    st.stop()
    
    
@st.cache_resource(ttl=3600) # Safety Valve: Reload model every hour to clear RAM leaks
def load_whisper():
    # Base is better for translation but heavier on RAM
    return whisper.load_model("base")

whisper_model = load_whisper()

# --- 4. UI LAYOUT ---
LANGUAGES = {
    "Auto-Detect": "auto",
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese (Mandarin)": "zh-cn",
    "Arabic": "ar",
    "Russian": "ru",
    "Portuguese": "pt",
    "Italian": "it",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Urdu": "ur",
    "Turkish": "tr"
}

if 'src_lang' not in st.session_state: st.session_state.src_lang = "Auto-Detect"
if 'target_lang' not in st.session_state: st.session_state.target_lang = "English"

col_from, col_swap, col_to = st.columns([4, 1, 4], vertical_alignment="bottom")

with col_from:
    src_lang = st.selectbox("From:", list(LANGUAGES.keys()), key="src_lang")
with col_swap:
    if st.button("🔄", use_container_width=True) and st.session_state.src_lang != "Auto-Detect":
        st.session_state.src_lang, st.session_state.target_lang = st.session_state.target_lang, st.session_state.src_lang
        st.rerun()
with col_to:
    target_lang = st.selectbox("To:", [l for l in LANGUAGES.keys() if l != "Auto-Detect"], key="target_lang")

st.divider()

col_mic, col_up = st.columns(2)
with col_mic:
    audio_mic = st.audio_input("🎤 Record") if hasattr(st, "audio_input") else None
with col_up:
    audio_file = st.file_uploader("📤 Upload", type=["mp3", "wav", "m4a"])

# --- 5. ENGINE (MEMORY OPTIMIZED) ---
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
    # CLEAR RAM IMMEDIATELY BEFORE STARTING
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # These must be on one line to stay hidden
    original_text = st.session_state.get('original_text', "")
    translated_text = st.session_state.get('translated_text', "")
    detected_lang_name = st.session_state.get('detected_lang_name', "")


    with st.status("🧠 AI Processing (Base Model)...") as status:
        try:
            # Step A: Whisper with No-Grad for RAM safety
            with torch.no_grad():
                st.write("👂 Decoding language...")
                result = whisper_model.transcribe(final_path, fp16=False, task="translate")
                original_text = result.get("text", "").strip()
                det_code = result.get("language", "en")
                detected_lang_name = next((k for k, v in LANGUAGES.items() if v == det_code), det_code)

            # Step B: Gemini
            status.update(label="✍️ Polishing Translation...", state="running")
            prompt = f"Translate to {target_lang}: {original_text}. Output ONLY translation."
            response = gemini_model.generate_content(prompt)
            translated_text = response.text.strip()
            
            # Step C: TTS
            status.update(label="🔊 Generating Voice...", state="running")
            tts = gTTS(text=translated_text, lang=LANGUAGES[target_lang])
            tts.save("out.mp3")
            
            # Step D: SAVE TO HISTORY
            if original_text and translated_text:
                new_entry = {
                    "from": detected_lang_name if src_lang == "Auto-Detect" else src_lang,
                    "to": target_lang,
                    "original": original_text,
                    "translated": translated_text
                }
                # Check to avoid duplicates before adding
                if not st.session_state.history or st.session_state.history[0]['original'] != original_text:
                    st.session_state.history.insert(0, new_entry)
                    
            status.update(label="✅ Success!", state="complete")
        except Exception as e:
            st.error(f"Error: {e}")

    # ---6. OUTPUTS OUTSIDE STATUS ---
    if original_text and translated_text:
        if src_lang == "Auto-Detect":
            st.success(f"📡 Detected: **{detected_lang_name.title()}**")     
        st.chat_message("user").write(original_text)
        st.chat_message("assistant").write(translated_text)
        st.audio("out.mp3", autoplay=True)
        
        if os.path.exists(final_path): os.remove(final_path)
        
else:
    st.info("💡 Record or upload to begin.")
    
# --- 7. SHOW HISTORY ---
if st.session_state.history:
    st.divider()
    st.subheader("📜 Translation History")
    for item in st.session_state.history[:5]:
        with st.expander(f"🕒 {item['from']} ➡ {item['to']}"):
            st.write(f"**Original:** {item['original']}")
            st.write(f"**Translated:** {item['translated']}")
            
