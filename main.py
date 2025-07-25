import os
import time
from groq import Groq
from gtts import gTTS

# Inisialisasi klien Groq dengan API key
groq_client = Groq(api_key="")


# Transkripsi audio menjadi teks menggunakan Groq Whisper
def transcribe_audio(filepath):
    with open(filepath, "rb") as f:
        response = groq_client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=f,
        )
    return response.text


# Mengirim pertanyaan ke LLM (LLaMA3) dan mengembalikan jawaban chatbot
def get_answer(question):
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


# Menampilkan teks jawaban dengan efek mengetik di terminal
def typing_effect(text, delay=0.03):
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)
    print()


# Mengubah teks jawaban menjadi audio dan menyimpannya sebagai file MP3
def text_to_speech(text, filename):
    tts = gTTS(text, lang="id")
    output_path = f"{filename}.mp3"
    tts.save(output_path)
    return output_path


# Fungsi utama untuk menjalankan program chatbot berbasis input teks atau audio
def main():
    mode = input("Pilih tipe input ('text' atau 'audio'): ").strip().lower()

    if mode == "text":
        question = input("Ketik pertanyaan Anda: ").strip()

    elif mode == "audio":
        filepath = input("Masukkan path file audio Anda: ").strip()
        if not os.path.exists(filepath):
            print("❌ File tidak ditemukan.")
            return
        print("🎤 Menyalin suara ke teks...")
        question = transcribe_audio(filepath)
        print(f"📝 Teks yang Disalin: {question}")

    else:
        print("❌ Tipe input tidak valid. Gunakan 'text' atau 'audio'.")
        return

    print("🤖 Mengambil jawaban dari LLM...")
    answer = get_answer(question)

    print("\n✅ Jawaban:")
    typing_effect(answer)

    print("\n🔊 Mengonversi jawaban ke suara...")
    audio_file = text_to_speech(answer, "response_audio")
    print(f"🎧 Suara disimpan di: {audio_file}")


# Menjalankan program jika file dieksekusi langsung
if __name__ == "__main__":
    main()
