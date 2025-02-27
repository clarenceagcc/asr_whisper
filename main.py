import streamlit as st
import whisper
import tempfile
import torch
import os
import subprocess
import soundfile as sf
from TTS.api import TTS  # Coqui TTS

# Check if GPU is available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Whisper model (cached)
@st.cache_resource
def load_model():
    return whisper.load_model("medium").to(device)

model = load_model()

# List available 🐸TTS models
#print(TTS().list_models())

# Load Coqui TTS Model
@st.cache_resource
def load_tts_model():
    return TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=torch.cuda.is_available())

tts_model = load_tts_model()

# Function to extract audio from video
def extract_audio(video_path, output_audio_path):
    command = [
        'ffmpeg', '-i', video_path, 
        '-vn', '-acodec', 'mp3', '-ar', '16000', '-ac', '1', 
        output_audio_path
    ]
    subprocess.run(command, check=True)

st.title("🎥 Video Speech Transcription & TTS with Whisper + Coqui")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mkv", "avi", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)

    # Save video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.getbuffer())
        video_path = temp_video.name

    # Extract audio from video
    audio_path = video_path.replace(".mp4", ".mp3")
    extract_audio(video_path, audio_path)

    st.write("Transcribing audio... ⏳")
    result = model.transcribe(audio_path, word_timestamps=True)

    # Store transcription with timestamps
    transcription_text = []
    tts_segments = []

    for segment in result["segments"]:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]
        transcription_text.append(f"[{start_time:.2f}s - {end_time:.2f}s] {text}")
        tts_segments.append(text)

    st.subheader("Transcription with Timing:")
    st.write("\n".join(transcription_text))

    # Convert each segment to speech using Coqui TTS
    st.write("Generating speech with Coqui TTS... 🔊")
    
    tts_audio_path = audio_path.replace(".mp3", "_tts.wav")

    # Generate TTS speech from transcribed text
    tts_model.tts_to_file(text=" ".join(tts_segments), file_path=tts_audio_path)

    # Play the generated speech
    st.audio(tts_audio_path)

    # Remove temporary files after processing
    os.remove(video_path)
    os.remove(audio_path)
