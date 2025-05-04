# --- All imports and configs remain as you posted ---
import whisper
import pyaudio
import numpy as np
import soundfile as sf
import os
import pywt
from scipy.signal import stft, istft
import matplotlib.pyplot as plt
import streamlit as st

# Load Whisper model
model = whisper.load_model("base")

# Audio settings
p = pyaudio.PyAudio()
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
DEVICE_INDEX = 0
RECORD_SECONDS = 10

# --- Audio Processing Functions ---
def list_audio_devices():
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        print(f"Device {i}: {dev['name']}, Input Channels: {dev['maxInputChannels']}")
    p.terminate()

def wavelet_decomposition(x, wavelet='db8', level=4):
    return pywt.wavedec(x, wavelet, level)

def adaptive_threshold(coeffs):
    thresholded = []
    for i, c in enumerate(coeffs):
        if i == 0:
            thresholded.append(c)
        else:
            median = np.median(np.abs(c))
            thr = (median / 0.6745) * np.sqrt(2 * np.log(len(c))) * 0.5
            thresholded.append(pywt.threshold(c, thr, mode='soft'))
    return thresholded

def wavelet_reconstruction(coeffs, wavelet='db8'):
    return pywt.waverec(coeffs, wavelet)

def time_frequency_optimize(x, fs):
    f, t, Zxx = stft(x, fs=fs)
    threshold = np.median(np.abs(Zxx))
    Zxx[np.abs(Zxx) < threshold] = 0
    _, x_recon = istft(Zxx, fs=fs)
    return x_recon

def adaptive_filter(x, d, mu=0.01, M=32):
    N = len(x)
    w = np.zeros(M)
    y = np.zeros(N)
    e = np.zeros(N)
    for n in range(M, N):
        x_n = x[n-M:n][::-1]
        y[n] = np.dot(w, x_n)
        e[n] = d[n] - y[n]
        w += mu * e[n] * x_n
    return y, e

def process_speech(signal, sample_rate):
    coeffs = wavelet_decomposition(signal)
    thresholded = adaptive_threshold(coeffs)
    reconstructed = wavelet_reconstruction(thresholded)
    tf_optimized = time_frequency_optimize(reconstructed, sample_rate)
    denoised, _ = adaptive_filter(signal, tf_optimized)
    return denoised

def visualize_error(original, denoised, sample_rate):
    time = np.arange(len(original)) / sample_rate
    error = original - denoised
    fig, axs = plt.subplots(3, 1, figsize=(15, 8))
    axs[0].plot(time, original, color='orange')
    axs[0].set_title("Original (Noisy) Signal")
    axs[1].plot(time, denoised, color='green')
    axs[1].set_title("Denoised Signal")
    axs[2].plot(time, error, color='red')
    axs[2].set_title("Residual Noise After Denoising")
    plt.tight_layout()
    return fig

def record_audio():
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK,
                        input_device_index=DEVICE_INDEX)
        frames = []
        print("Recording started...")
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            if not st.session_state.get("recording", False):
                print("Recording paused by user.")
                break
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        print("Recording finished.")
        stream.stop_stream()
        stream.close()
        if not frames:
            print("Warning: No audio recorded.")
            return None
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        if np.max(np.abs(audio_data)) == 0:
            print("Warning: Recorded audio is silent")
        return audio_data
    except Exception as e:
        print(f"Error in record_audio: {str(e)}")
        return None

def transcribe_audio(audio_data):
    temp_audio_file = "temp.wav"
    try:
        sf.write(temp_audio_file, audio_data, RATE)
        result = model.transcribe(temp_audio_file)
        transcription = result['text']
        print("Whisper raw output:", result)
        return transcription
    except Exception as e:
        print(f"Whisper error: {str(e)}")
        return ""
    finally:
        if os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)

# --- ISL Conversion Class ---
class EnglishToISLSentenceConverter:
    def __init__(self):
        self.question_words = {"what", "who", "where", "when", "why", "how"}
        self.common_verbs = {"go", "eat", "read", "am", "is", "are", "do", "does"}

    def tokenize(self, sentence):
        return sentence.lower().strip().replace(".", "").split()

    def reorder_to_sov(self, tokens):
        subjects, objects, verbs, others = [], [], [], []
        for i, word in enumerate(tokens):
            if i == 0 and word in {"i", "you", "he", "she", "they"}:
                subjects.append(word)
            elif word in self.common_verbs:
                verbs.append(word)
            elif word in self.question_words:
                others.append(word)
            else:
                objects.append(word)
        return subjects + objects + others + verbs

    def handle_negation(self, reordered):
        if "not" in reordered:
            reordered.remove("not")
            reordered.insert(len(reordered) - 1, "not")
        return reordered

    def handle_questions(self, sentence, reordered):
        tokens = set(self.tokenize(sentence))
        question_word = tokens & self.question_words
        if question_word:
            q_word = question_word.pop()
            if q_word in reordered:
                reordered.remove(q_word)
            reordered = [q_word] + reordered
        return reordered

    def convert(self, sentence):
        if not sentence or not isinstance(sentence, str):
            return "Error: Invalid or empty input"
        tokens = self.tokenize(sentence)
        if not tokens:
            return "Error: Unable to process sentence"
        reordered = self.reorder_to_sov(tokens)
        reordered = self.handle_negation(reordered)
        reordered = self.handle_questions(sentence, reordered)
        return " ".join(reordered)

# --- Full Pipeline: Record, Transcribe, Convert ---
def audio_to_text_pipeline():
    list_audio_devices()
    audio_data = record_audio()
    if audio_data is None:
        return "", None, ""
    noisy_signal = audio_data / (np.max(np.abs(audio_data)) + 1e-10)
    denoised_signal = process_speech(noisy_signal, RATE)
    fig = visualize_error(noisy_signal, denoised_signal, RATE)
    denoised_int16 = (denoised_signal * 32767).astype(np.int16)
    transcription = transcribe_audio(denoised_int16)
    converter = EnglishToISLSentenceConverter()
    isl_output = converter.convert(transcription)
    return transcription, fig, isl_output

if __name__ == "__main__":
    st.session_state["recording"] = True 
    text, fig, isl = audio_to_text_pipeline()
    print("Transcribed Text:", text)
    print("ISL Order:", isl)
