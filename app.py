import streamlit as st
import numpy as np
from isl_convert import EnglishToISLSentenceConverter, record_audio, process_speech, visualize_error, transcribe_audio
import matplotlib.pyplot as plt
import soundfile as sf


RATE = 16000


print("Streamlit imported successfully:", st.__version__)


st.title("🎙️ Speech to ISL Converter")

if "recording" not in st.session_state:
    st.session_state.recording = False
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None


button_label = "Stop Recording" if st.session_state.recording else "Start Recording"
if st.button(button_label):
    if st.session_state.recording:
       
        st.session_state.recording = False
        st.info("Recording paused. Processing your speech...")

        try:
          
            if st.session_state.audio_data is not None:
                audio_data = st.session_state.audio_data
                st.session_state.audio_data = None
            else:
                st.warning("No audio data recorded.")
                audio_data = np.zeros(int(RATE * 10), dtype=np.int16) 

           
            sf.write("debug_raw_audio.wav", audio_data, RATE)  
            noisy_signal = audio_data / np.max(np.abs(audio_data) + 1e-10)
            denoised_signal = process_speech(noisy_signal, RATE)
            fig = visualize_error(noisy_signal, denoised_signal, RATE)
            denoised_int16 = (denoised_signal * 32767).astype(np.int16)
            sf.write("debug_denoised_audio.wav", denoised_int16, RATE)
            transcription = transcribe_audio(denoised_int16)
            print("Received transcription:", transcription)

          
            converter = EnglishToISLSentenceConverter()
            isl_output = converter.convert(transcription)
            print("ISL output:", isl_output)

            
            st.success("✅ Transcription:")
            st.markdown(f"**{transcription if transcription else 'No transcription detected'}**")
            st.write("Debug: Transcription rendered")

           
            st.success(" ISL Structured Sentence:")
            st.markdown(f"**{isl_output if isl_output else 'No ISL output'}**")
            st.write("Debug: ISL output rendered")

            
            st.success(" Denoising Visualization:")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error in audio processing: {str(e)}")
            import traceback
            st.text("Full traceback:")
            st.text(traceback.format_exc())
            raise
    else:
       
        st.session_state.recording = True
        st.info("Recording started... Speak now!")


if st.session_state.recording:
    try:
        
        audio_data = record_audio()
        if audio_data is not None:
            st.session_state.audio_data = audio_data
            st.info("Recording complete. Click 'Stop Recording' to process.")
        else:
            st.warning("Recording interrupted.")
            st.session_state.recording = False
            st.session_state.audio_data = None
    except Exception as e:
        st.error(f"Error during recording: {str(e)}")
        st.session_state.recording = False
        st.session_state.audio_data = None