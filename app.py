import os
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from gtts import gTTS
import random
import string

from groq import Groq

# Inisialisasi Flask dan konfigurasi folder upload/audio
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["ALLOWED_EXTENSIONS"] = {"webm", "wav", "mp3", "m4a"}

# Inisialisasi Groq client dengan API key
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])


# Mengecek apakah ekstensi file diizinkan
def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


# Transkripsi audio menjadi teks menggunakan Groq Whisper
def transcribe_audio_groq(filepath):
    with open(filepath, "rb") as f:
        response = groq_client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=f,
        )
        return response.text


# Mengambil jawaban dari LLM Groq (LLaMA 3)
def get_answer_groq(question):
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "system",
                "content": (
                    "Kamu adalah chatbot pertanian yang membantu Indonesia. "
                    "Jawablah dalam bahasa Indonesia yang sederhana, singkat, dan mudah dipahami. "
                    "Berikan informasi tentang pertanian lokal, seperti padi, jagung, cabai, hama, pupuk, dan cuaca."
                ),
            },
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


# Mengubah teks jawaban menjadi file audio MP3 menggunakan gTTS
def text_to_audio(text, filename):
    tts = gTTS(text, lang="id")
    audio_path = os.path.join("static/audio", f"{filename}.mp3")
    tts.save(audio_path)
    return audio_path


# Route halaman utama (form input di index.html)
@app.route("/")
def index():
    return render_template("index.html")


# Route untuk proses chat berbasis teks atau audio
@app.route("/chat", methods=["POST"])
def chat():
    if "audio" in request.files:
        # Jika input berupa file audio: simpan → transkrip → jawab → audio output
        audio = request.files["audio"]
        if audio and allowed_file(audio.filename):
            filename = secure_filename(audio.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            audio.save(filepath)
            transcription = transcribe_audio_groq(filepath)
            answer = get_answer_groq(transcription)
            voice_filename = "".join(
                random.choices(string.ascii_uppercase + string.digits, k=8)
            )
            text_to_audio(answer, voice_filename)
            return jsonify(
                {
                    "text": f"🎤 Ditranskripsi: {transcription}\n\n🤖 Jawaban: {answer}",
                    "voice": url_for(
                        "static", filename="audio/" + voice_filename + ".mp3"
                    ),
                }
            )

    elif "text" in request.form:
        # Jika input berupa teks langsung: jawab → audio output
        question = request.form["text"]
        answer = get_answer_groq(question)
        voice_filename = "".join(
            random.choices(string.ascii_uppercase + string.digits, k=8)
        )
        text_to_audio(answer, voice_filename)
        return jsonify(
            {
                "text": answer,
                "voice": url_for("static", filename="audio/" + voice_filename + ".mp3"),
            }
        )

    # Jika tidak ada input valid
    return jsonify({"text": "No valid input found"}), 400


# Jalankan server Flask dan buat folder yang dibutuhkan
if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("static/audio", exist_ok=True)
    app.run(debug=True)
