import pyaudio
import wave
import numpy as np
import speech_recognition as sr
import os
import time
import pywt
from scipy.signal import stft, istft
import matplotlib.pyplot as plt

class AudioRecorder:
    def __init__(self, output_directory="recordings", input_device_index=None):
        self.output_directory = output_directory
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # Common sampling rate for speech recognition
        self.chunk = 1024
        self.recording = False
        self.frames = []
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.filename = None
        self.input_device_index = input_device_index
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)  
    def start_recording(self):
        """Start recording audio"""
        try:
            self.recording = True
            self.frames = []
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.filename = os.path.join(self.output_directory, f"recording_{timestamp}.wav")
            
            # Setup and open stream
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
                input_device_index=self.input_device_index
            )
            
            print("Recording started...")
            
            # Record audio continuously until stop_recording is called
            while self.recording:
                try:
                    data = self.stream.read(self.chunk, exception_on_overflow=False)
                    self.frames.append(data)
                except Exception as e:
                    print(f"Error reading audio stream: {e}")
                    break
        except Exception as e:
            print(f"Error during recording setup: {e}")
            self.recording = False
    def stop_recording(self):
        """Stop recording and save the audio file"""
        try:
            self.recording = False
            print("Recording stopped.")
            
            # Give a moment for the recording loop to complete
            time.sleep(0.5)
            
            # Stop and close the stream
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                    self.stream = None
                except Exception as e:
                    print(f"Error closing stream: {e}")
            
            # Check if any frames were recorded
            if not self.frames:
                print("Warning: No audio data was recorded!")
                return None
            
            # Save the recorded audio to a WAV file
            try:
                wf = wave.open(self.filename, 'wb')
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(self.frames))
                wf.close()
                
                # Check if the file has actual audio content
                if os.path.getsize(self.filename) <= 44:  # WAV header is typically 44 bytes
                    print("Warning: The recorded file appears to be empty (only contains header).")
                else:
                    print(f"Audio saved to {self.filename}")
                
                # Add basic audio level check
                self._check_audio_levels()
                
                return self.filename
            except Exception as e:
                print(f"Error saving audio file: {e}")
                return None
                
        except Exception as e:
            print(f"Error in stop_recording: {e}")
            return None
        
    def _check_audio_levels(self):
        """Check if recorded audio has meaningful signal"""
        if not self.frames:
            return
            
        # Convert audio frames to numpy array for analysis
        audio_data = np.frombuffer(b''.join(self.frames), dtype=np.int16)
        
        # Calculate audio metrics
        abs_data = np.abs(audio_data)
        max_amplitude = np.max(abs_data)
        mean_amplitude = np.mean(abs_data)
        
        print(f"Audio diagnostics:")
        print(f"  - Maximum amplitude: {max_amplitude}")
        print(f"  - Average amplitude: {mean_amplitude:.2f}")
        
        # Provide feedback on audio levels
        if max_amplitude < 1000:
            print("Warning: Very low audio levels detected. Your microphone might not be capturing audio properly.")
            print("Suggestions:")
            print("  - Check if your microphone is muted")
            print("  - Increase your microphone volume in system settings")
            print("  - Move closer to the microphone or speak louder")
    
    def transcribe_audio(self, audio_file=None):
        """Transcribe the recorded audio to text"""
        if audio_file is None:
            audio_file = self.filename
            
        if not audio_file or not os.path.exists(audio_file):
            print("No audio file found to transcribe.")
            return None
        
        recognizer = sr.Recognizer()
        
        print("Transcribing audio...")
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            
            try:
                text = recognizer.recognize_google(audio_data)
                print("Transcription:")
                print(text)
                return text
            except sr.UnknownValueError:
                print("Speech Recognition could not understand audio")
                return None
            except sr.RequestError as e:
                print(f"Could not request results from Speech Recognition service; {e}")
                return None
    
    def cleanup(self):
        """Clean up resources"""
        self.audio.terminate()
        print("Audio resources cleaned up.")
    
    def process_audio(self, audio_file=None):
        """Process recorded audio with advanced noise filtering and visualization"""
        if audio_file is None:
            audio_file = self.filename
            
        if not audio_file or not os.path.exists(audio_file):
            print("No audio file found to process.")
            return None, None, None
        
        # Load audio data
        with wave.open(audio_file, 'rb') as wf:
            audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            sample_rate = wf.getframerate()
        
        # Normalize
        noisy_signal = audio_data / (np.max(np.abs(audio_data)) + 1e-10)
        
        # Process the audio
        denoised_signal = self._process_speech(noisy_signal, sample_rate)
        
        # Generate visualization
        fig = self._visualize_error(noisy_signal, denoised_signal, sample_rate)
        
        # Convert back to int16 for speech recognition
        denoised_int16 = (denoised_signal * 32767).astype(np.int16)
        
        # Save processed audio
        processed_filename = audio_file.replace('.wav', '_processed.wav')
        with wave.open(processed_filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(denoised_int16.tobytes())
        
        print(f"Processed audio saved to {processed_filename}")
        
        # Transcribe the processed audio
        transcription = self.transcribe_audio(processed_filename)
        
        return processed_filename, fig, transcription
    
    def _wavelet_decomposition(self, x, wavelet='db8', level=4):
        """Decompose signal using wavelet transform"""
        return pywt.wavedec(x, wavelet, level)

    def _adaptive_threshold(self, coeffs):
        """Apply adaptive thresholding to wavelet coefficients"""
        thresholded = []
        for i, c in enumerate(coeffs):
            if i == 0:
                thresholded.append(c)
            else:
                median = np.median(np.abs(c))
                thr = (median / 0.6745) * np.sqrt(2 * np.log(len(c))) * 0.5
                thresholded.append(pywt.threshold(c, thr, mode='soft'))
        return thresholded

    def _wavelet_reconstruction(self, coeffs, wavelet='db8'):
        """Reconstruct signal from wavelet coefficients"""
        return pywt.waverec(coeffs, wavelet)

    def _time_frequency_optimize(self, x, fs):
        """Optimize signal in time-frequency domain"""
        f, t, Zxx = stft(x, fs=fs)
        threshold = np.median(np.abs(Zxx))
        Zxx[np.abs(Zxx) < threshold] = 0
        _, x_recon = istft(Zxx, fs=fs)
        return x_recon

    def _adaptive_filter(self, x, d, mu=0.01, M=32):
        """Apply adaptive filtering"""
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

    def _process_speech(self, signal, sample_rate):
        """Apply a series of signal processing techniques to denoise speech"""
        coeffs = self._wavelet_decomposition(signal)
        thresholded = self._adaptive_threshold(coeffs)
        reconstructed = self._wavelet_reconstruction(thresholded)
        tf_optimized = self._time_frequency_optimize(reconstructed, sample_rate)
        denoised, _ = self._adaptive_filter(signal, tf_optimized)
        return denoised

    def _visualize_error(self, original, denoised, sample_rate):
        """Create visualization of original vs. denoised signal with error analysis"""
        time = np.arange(len(original)) / sample_rate
        error = original - denoised
        
        # Generate time-domain visualization
        fig, axs = plt.subplots(3, 1, figsize=(15, 10))
        
        # Original signal
        axs[0].plot(time, original, color='orange')
        axs[0].set_title("Original (Noisy) Signal", fontsize=14)
        axs[0].set_ylabel("Amplitude", fontsize=12)
        axs[0].grid(True, alpha=0.3)
        
        # Denoised signal
        axs[1].plot(time, denoised, color='green')
        axs[1].set_title("Denoised Signal", fontsize=14)
        axs[1].set_ylabel("Amplitude", fontsize=12)
        axs[1].grid(True, alpha=0.3)
        
        # Error/residual
        axs[2].plot(time, error, color='red')
        axs[2].set_title("Residual Noise (Error)", fontsize=14)
        axs[2].set_xlabel("Time (seconds)", fontsize=12)
        axs[2].set_ylabel("Amplitude", fontsize=12)
        axs[2].grid(True, alpha=0.3)
        
        # Calculate and display error metrics
        mse = np.mean(error**2)
        snr = 10 * np.log10(np.sum(denoised**2) / np.sum(error**2)) if np.sum(error**2) > 0 else float('inf')
        fig.suptitle(f"Audio Signal Analysis\nMSE: {mse:.6f}, SNR: {snr:.2f} dB", fontsize=16)
        
        plt.tight_layout()
        return fig

def list_audio_devices():
    """List all available audio input devices"""
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')
    
    print("\nAvailable Audio Input Devices:")
    print("------------------------------")
    input_devices = []
    
    for i in range(num_devices):
        device_info = p.get_device_info_by_index(i)
        if device_info.get('maxInputChannels') > 0:  # Check if it's an input device
            input_devices.append((i, device_info))
            print(f"{len(input_devices)}. {device_info.get('name')}")
    
    p.terminate()
    return input_devices

def main():
    print("Audio to Text Converter")
    print("=======================")
    
    # List available audio devices
    input_devices = list_audio_devices()
    
    # If no input devices found
    if not input_devices:
        print("No audio input devices found. Please check your microphone connection.")
        return
    
    # Let user select input device
    selected_device = None
    while selected_device is None:
        try:
            selection = input("\nSelect your microphone by number (or Enter for default): ")
            if selection.strip() == "":
                selected_device = None  # Use default
                break
            
            idx = int(selection) - 1
            if 0 <= idx < len(input_devices):
                device_index, device_info = input_devices[idx]
                selected_device = device_index
                print(f"Selected: {device_info.get('name')}")
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    print("\nCommands:")
    print("  r - Start recording")
    print("  s - Stop recording (while recording)")
    print("  t - Transcribe last recording")
    print("  p - Process last recording")
    print("  q - Quit")
    
    recorder = AudioRecorder(input_device_index=selected_device)
    
    while True:
        command = input("Enter command (r/t/p/q): ").lower()
        
        if command == 'r':
            print("Starting recording... Press 's' to stop.")
            recorder.start_recording()
        elif command == 't':
            recorder.transcribe_audio()
        elif command == 'p':
            recorder.process_audio()
        elif command == 'q':
            recorder.cleanup()
            print("Exiting program.")
            break
        else:
            print("Invalid command. Try again.")

if __name__ == "__main__":
    main()