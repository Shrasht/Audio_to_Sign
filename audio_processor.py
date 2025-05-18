import numpy as np
import os
import wave
import matplotlib.pyplot as plt
from recorder import AudioRecorder
from isl_convert import EnglishToISLSentenceConverter

class IntegratedAudioProcessor:
    """Class that integrates recording, processing, and ISL conversion"""
    
    def __init__(self, input_device_index=None):
        self.recorder = AudioRecorder(input_device_index=input_device_index)
        self.converter = EnglishToISLSentenceConverter()
        self.last_recording = None
        self.last_transcription = None
        self.last_isl_output = None
        self.last_figure = None
        self.last_stats = None
    
    def record_and_process(self):
        """Record audio, process it, and convert to ISL in one step"""
        # Start recording
        print("Starting recording...")
        self.recorder.start_recording()  # This will run until stopped
        
        # After recording is stopped
        audio_file = self.recorder.filename
        if not audio_file or not os.path.exists(audio_file):
            print("No audio was recorded.")
            return None
            
        self.last_recording = audio_file
        
        # Process the audio
        processed_file, fig, transcription = self.recorder.process_audio(audio_file)
        
        if transcription:
            # Convert to ISL
            isl_output = self.converter.convert(transcription)
            
            # Store results
            self.last_transcription = transcription
            self.last_isl_output = isl_output
            self.last_figure = fig
            
            # Calculate statistics about the audio processing
            self._calculate_stats(audio_file, processed_file)
            
            return {
                'audio_file': audio_file,
                'processed_file': processed_file,
                'transcription': transcription,
                'isl_output': isl_output,
                'figure': fig,
                'stats': self.last_stats
            }
        else:
            print("Transcription failed.")
            return None
    
    def _calculate_stats(self, original_file, processed_file):
        """Calculate statistics comparing original and processed audio"""
        # Load original audio
        with wave.open(original_file, 'rb') as wf:
            original_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            sample_rate = wf.getframerate()
        
        # Load processed audio
        with wave.open(processed_file, 'rb') as wf:
            processed_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        
        # Normalize data for comparison
        original_norm = original_data / (np.max(np.abs(original_data)) + 1e-10)
        processed_norm = processed_data / (np.max(np.abs(processed_data)) + 1e-10)
        
        # Make sure lengths match for comparison
        min_len = min(len(original_norm), len(processed_norm))
        original_norm = original_norm[:min_len]
        processed_norm = processed_norm[:min_len]
        
        # Calculate error signal
        error = original_norm - processed_norm
        
        # Calculate statistics
        mse = np.mean(error**2)
        peak_error = np.max(np.abs(error))
        snr = 10 * np.log10(np.sum(processed_norm**2) / np.sum(error**2)) if np.sum(error**2) > 0 else float('inf')
        
        # Calculate frequency-domain metrics (power in different bands)
        from scipy.signal import welch
        freqs, power_original = welch(original_norm, fs=sample_rate, nperseg=1024)
        _, power_processed = welch(processed_norm, fs=sample_rate, nperseg=1024)
        
        # Store statistics
        self.last_stats = {
            'mse': mse,
            'peak_error': peak_error,
            'snr_db': snr,
            'freqs': freqs,
            'power_original': power_original,
            'power_processed': power_processed,
            'duration': len(original_data) / sample_rate
        }
        
        return self.last_stats

    def generate_detailed_report(self):
        """Generate a detailed report with visualizations"""
        if not self.last_stats:
            return None
            
        # Create a multi-panel figure
        fig = plt.figure(figsize=(15, 12))
        
        # Time domain analysis
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.set_title('Audio Signal Processing Results', fontsize=16)
        ax1.text(0.5, 0.5, f"""
        Audio Duration: {self.last_stats['duration']:.2f} seconds
        Signal-to-Noise Ratio: {self.last_stats['snr_db']:.2f} dB
        Mean Squared Error: {self.last_stats['mse']:.6f}
        Peak Error: {self.last_stats['peak_error']:.6f}
        
        Transcription: "{self.last_transcription}"
        
        ISL Structure: "{self.last_isl_output}"
        """, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=12)
        ax1.axis('off')
        
        # Frequency domain analysis
        ax2 = fig.add_subplot(3, 1, 2)
        ax2.semilogy(self.last_stats['freqs'], self.last_stats['power_original'], label='Original')
        ax2.semilogy(self.last_stats['freqs'], self.last_stats['power_processed'], label='Processed')
        ax2.set_title('Frequency Domain Analysis', fontsize=14)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Power Spectrum Density')
        ax2.legend()
        ax2.grid(True)
        
        # Add speech frequency ranges
        speech_ranges = [
            (20, 300, 'Bass', 'lightblue'),
            (300, 1000, 'Low-Mid', 'lightgreen'),
            (1000, 3000, 'Mid-High', 'lightyellow'),
            (3000, 8000, 'High', 'lightcoral')
        ]
        
        ymin, ymax = ax2.get_ylim()
        for start, end, name, color in speech_ranges:
            ax2.fill_between([start, end], [ymin, ymin], [ymax, ymax], color=color, alpha=0.3)
            ax2.text((start + end)/2, ymin*10, name, horizontalalignment='center')
        
        # Add interpretation
        ax3 = fig.add_subplot(3, 1, 3)
        ax3.axis('off')
        ax3.set_title('Audio Processing Interpretation', fontsize=14)
        
        # Interpret the SNR
        snr = self.last_stats['snr_db']
        if snr > 20:
            snr_quality = "Excellent - Very clean audio signal"
        elif snr > 15:
            snr_quality = "Good - Clear speech with minimal noise"
        elif snr > 10:
            snr_quality = "Fair - Speech is understandable but some noise present"
        elif snr > 5:
            snr_quality = "Poor - Speech is difficult to understand due to noise"
        else:
            snr_quality = "Very poor - Speech is likely unintelligible"
        
        ax3.text(0.5, 0.7, f"Signal Quality: {snr_quality}", horizontalalignment='center', transform=ax3.transAxes, fontsize=12)
        
        # Add recommendations based on analysis
        recommendations = []
        if snr < 10:
            recommendations.append("- Try recording in a quieter environment")
            recommendations.append("- Position the microphone closer to your mouth")
        if self.last_stats['peak_error'] > 0.5:
            recommendations.append("- Reduce background noise sources")
            recommendations.append("- Speak more clearly and at a consistent volume")
        
        if recommendations:
            ax3.text(0.5, 0.5, "Recommendations for better results:", horizontalalignment='center', transform=ax3.transAxes, fontsize=12)
            recommendation_text = "\n".join(recommendations)
            ax3.text(0.5, 0.3, recommendation_text, horizontalalignment='center', transform=ax3.transAxes, fontsize=11)
        else:
            ax3.text(0.5, 0.4, "Your recording quality is good! No specific improvements needed.", horizontalalignment='center', transform=ax3.transAxes, fontsize=12)
        
        plt.tight_layout()
        return fig

if __name__ == "__main__":
    # Example usage
    processor = IntegratedAudioProcessor()
    results = processor.record_and_process()
    
    if results:
        print(f"Transcription: {results['transcription']}")
        print(f"ISL Output: {results['isl_output']}")
        
        # Display figure
        results['figure'].show()
        
        # Generate and display detailed report
        report_fig = processor.generate_detailed_report()
        if report_fig:
            report_fig.show()
    else:
        print("Processing failed.")
