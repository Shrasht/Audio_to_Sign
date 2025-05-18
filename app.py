import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import threading
import datetime
import time
from recorder import AudioRecorder, list_audio_devices
from isl_convert import EnglishToISLSentenceConverter
from translation_evaluator import TranslationEvaluator

st.set_page_config(
    page_title="Speech to ISL Converter",
    page_icon="🎙️",
    layout="wide",
)

# CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
    }
    .info-text {
        font-size: 1.2rem;
    }
    .status-recording {
        color: #FF5252;
        font-weight: bold;
    }
    .status-processing {
        color: #FF9800;
        font-weight: bold;
    }
    .status-complete {
        color: #4CAF50;
        font-weight: bold;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }    .result-text {
        font-size: 1.5rem;
        padding: 15px;
        color:black;
        background-color: #e3f2fd;
        border-radius: 10px;
        margin-top: 10px;
    }
    .metric-card {
        background-color: #f1f8fe;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #1E88E5;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #0D47A1;
    }
    .metric-label {
        font-size: 1rem;
        color: #555;
    }
    .quality-excellent {
        color: #4CAF50;
        font-weight: bold;
    }
    .quality-good {
        color: #8BC34A;
        font-weight: bold;
    }
    .quality-fair {
        color: #FFC107;
        font-weight: bold;
    }
    .quality-poor {
        color: #FF9800;
        font-weight: bold;
    }
    .quality-very-poor {
        color: #F44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🎙️ Speech to Indian Sign Language Converter</h1>", unsafe_allow_html=True)

# Initialize session state variables
if "recorder" not in st.session_state:
    st.session_state.recorder = None
if "recording" not in st.session_state:
    st.session_state.recording = False
if "audio_file" not in st.session_state:
    st.session_state.audio_file = None
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None
if "transcription" not in st.session_state:
    st.session_state.transcription = None
if "isl_output" not in st.session_state:
    st.session_state.isl_output = None
if "fig" not in st.session_state:
    st.session_state.fig = None
if "selected_device" not in st.session_state:
    st.session_state.selected_device = None
if "metrics_fig" not in st.session_state:
    st.session_state.metrics_fig = None
if "bleu_score" not in st.session_state:
    st.session_state.bleu_score = None
if "wer_score" not in st.session_state:
    st.session_state.wer_score = None
if "reference_text" not in st.session_state:
    st.session_state.reference_text = None
if "comparison_fig" not in st.session_state:
    st.session_state.comparison_fig = None
if "other_models" not in st.session_state:
    st.session_state.other_models = {}

# Define cleanup handling function
def cleanup_on_session_end():
    if st.session_state.recorder and st.session_state.recording:
        st.session_state.recorder.recording = False
        st.session_state.recorder.stop_recording()
        st.session_state.recorder.cleanup()

# Register cleanup for when session ends
try:
    st.session_state.update({"_on_session_end": cleanup_on_session_end})
except:
    pass  # Older versions of Streamlit may not support this

# Setup section
with st.expander("Setup Recording Device", expanded=st.session_state.recorder is None):
    st.markdown("<h3 class='sub-header'>Select Your Microphone</h3>", unsafe_allow_html=True)
    
    # Only list devices if we haven't already selected one
    if st.session_state.recorder is None:
        input_devices = list_audio_devices()
        
        if not input_devices:
            st.error("No audio input devices found. Please check your microphone connection.")
        else:
            device_names = [f"{idx+1}. {device_info.get('name')}" for idx, (_, device_info) in enumerate(input_devices)]
            selected_option = st.selectbox("Choose your microphone:", options=["Default"] + device_names)
            
            if st.button("Confirm Microphone Selection"):
                if selected_option == "Default":
                    st.session_state.selected_device = None
                else:
                    idx = int(selected_option.split('.')[0]) - 1
                    device_index, _ = input_devices[idx]
                    st.session_state.selected_device = device_index
                
                st.session_state.recorder = AudioRecorder(input_device_index=st.session_state.selected_device)
                st.success(f"✅ Microphone selected: {selected_option}")
               
    else:
        st.success("✅ Recording device is configured!")
        if st.button("Change Microphone"):
            st.session_state.recorder = None
            st.experimental_rerun()

# Only show the main interface if recorder is configured
if st.session_state.recorder:
    # Main recording interface
    st.markdown("<h3 class='sub-header'>Record Your Speech</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if not st.session_state.recording:
            if st.button("🎙️ Start Recording", use_container_width=True):
                # Reset metrics and comparison state for new recording
                st.session_state.reference_text = None
                st.session_state.bleu_score = None
                st.session_state.wer_score = None
                st.session_state.metrics_fig = None
                st.session_state.comparison_fig = None
                st.session_state.other_models = {}

                st.session_state.recording = True
                
                # Store recorder reference to avoid session state issues
                recorder_instance = st.session_state.recorder
                
                # Start recording in a separate thread to prevent app freezing
                import threading
                def start_recording_thread():
                    recorder_instance.start_recording()
                
                # Start the recording thread
                recording_thread = threading.Thread(target=start_recording_thread)
                recording_thread.daemon = True
                recording_thread.start()
                st.rerun()
        else:
            if st.button("⏹️ Stop Recording", use_container_width=True):
                st.session_state.recording = False
                
                with st.spinner("Processing audio..."):
                    # This will stop the recording that was started in the background
                    st.session_state.audio_file = st.session_state.recorder.stop_recording()
                    
                    if st.session_state.audio_file:
                        # Process and analyze the audio
                        processed_file, fig, transcription = st.session_state.recorder.process_audio()
                        
                        if transcription:
                            # Convert to ISL
                            converter = EnglishToISLSentenceConverter()
                            isl_output = converter.convert(transcription)
                            
                            # Store results in session state
                            st.session_state.processed_file = processed_file
                            st.session_state.fig = fig
                            st.session_state.transcription = transcription
                            st.session_state.isl_output = isl_output
                            
                            st.success("✅ Audio processing complete!")
                        else:
                            st.warning("⚠️ Could not transcribe the audio. Please try again with clearer speech.")
                    else:
                        st.error("❌ No audio was recorded. Please try again.")
                
                st.rerun()
          # Show current status
        if st.session_state.recording:
            st.markdown("<p class='status-recording'>🔴 Recording in progress... Speak now!</p>", unsafe_allow_html=True)
            st.markdown("<p>Recording will continue until you click 'Stop Recording'</p>", unsafe_allow_html=True)
            placeholder = st.empty()
            
            # Show recording duration
            current_time = datetime.datetime.now()
            if 'recording_start_time' not in st.session_state:
                st.session_state.recording_start_time = current_time
                
            duration = current_time - st.session_state.recording_start_time
            placeholder.text(f"Recording time: {duration.seconds}s")
            
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h4>Instructions:</h4>", unsafe_allow_html=True)
        st.markdown("""
        1. Click **Start Recording** and speak clearly into your microphone
        2. Click **Stop Recording** when you're done speaking
        3. Wait for the system to process your speech
        4. View the results below: transcription, ISL conversion, and audio analysis
        """)
        st.markdown("</div>", unsafe_allow_html=True)
      # Results section
    if st.session_state.transcription:
        st.markdown("<h3 class='sub-header'>Results</h3>", unsafe_allow_html=True)
        
        tabs = st.tabs(["Transcription & ISL", "Audio Analysis", "Translation Metrics", "About"])
        
        with tabs[0]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h4>Speech Transcription:</h4>", unsafe_allow_html=True)
                st.markdown(f"<div class='result-text'>{st.session_state.transcription}</div>", unsafe_allow_html=True)
                
                if st.session_state.processed_file and os.path.exists(st.session_state.processed_file):
                    st.audio(st.session_state.processed_file)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h4>Indian Sign Language Structure:</h4>", unsafe_allow_html=True)
                st.markdown(f"<div class='result-text'>{st.session_state.isl_output}</div>", unsafe_allow_html=True)
                
                st.markdown("<p><i>Note: ISL typically follows Subject-Object-Verb order and drops articles.</i></p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        with tabs[1]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h4>Audio Signal Analysis:</h4>", unsafe_allow_html=True)
            if st.session_state.fig:
                st.pyplot(st.session_state.fig)
            else:
                st.info("No audio analysis available.")
            st.markdown("</div>", unsafe_allow_html=True)
        with tabs[2]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h4>Translation Quality Metrics:</h4>", unsafe_allow_html=True)
            
            # Reference text input
            if 'reference_text' not in st.session_state or not st.session_state.reference_text:
                st.markdown("<p>Enter a reference translation to calculate BLEU and WER scores:</p>", unsafe_allow_html=True)
                reference = st.text_area("Reference Text (ideal translation or gold standard)", 
                                          placeholder="Enter the correct/expected translation here...",
                                          key="reference_input")
                
                if st.button("Calculate Metrics"):
                    if reference:
                        st.session_state.reference_text = reference
                        
                        # Initialize evaluator
                        evaluator = TranslationEvaluator()
                        
                        # Calculate BLEU
                        st.session_state.bleu_score = evaluator.calculate_bleu(
                            reference, st.session_state.transcription)
                        
                        # Calculate WER
                        st.session_state.wer_score = evaluator.calculate_wer(
                            reference, st.session_state.transcription)
                        
                        # Generate visualization
                        st.session_state.metrics_fig = evaluator.generate_combined_metrics_visualization(
                            reference, st.session_state.transcription)
                        
                        st.rerun()
                    else:
                        st.warning("Please enter a reference text first.")
            
            # Display metrics if available
            if st.session_state.reference_text and st.session_state.bleu_score is not None and st.session_state.wer_score is not None:
                evaluator = TranslationEvaluator()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown("<p class='metric-label'>BLEU Score (higher is better)</p>", unsafe_allow_html=True)
                    quality_class = "excellent" if st.session_state.bleu_score > 0.8 else \
                                   "good" if st.session_state.bleu_score > 0.6 else \
                                   "fair" if st.session_state.bleu_score > 0.4 else \
                                   "poor" if st.session_state.bleu_score > 0.2 else "very-poor"
                    st.markdown(f"<p class='metric-value quality-{quality_class}'>{st.session_state.bleu_score:.3f}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p>{evaluator.interpret_bleu_score(st.session_state.bleu_score)}</p>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown("<p class='metric-label'>Word Error Rate (lower is better)</p>", unsafe_allow_html=True)
                    quality_class = "excellent" if st.session_state.wer_score < 0.05 else \
                                   "good" if st.session_state.wer_score < 0.1 else \
                                   "fair" if st.session_state.wer_score < 0.2 else \
                                   "poor" if st.session_state.wer_score < 0.3 else "very-poor"
                    st.markdown(f"<p class='metric-value quality-{quality_class}'>{st.session_state.wer_score:.3f}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p>{evaluator.interpret_wer(st.session_state.wer_score)}</p>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Show the reference text
                st.markdown("<h5>Reference Text:</h5>", unsafe_allow_html=True)
                st.markdown(f"<div class='result-text'>{st.session_state.reference_text}</div>", unsafe_allow_html=True)
                
                # Show the visualization
                if st.session_state.metrics_fig:
                    st.pyplot(st.session_state.metrics_fig)
                
                # MODEL COMPARISON SECTION
                st.markdown("<h4>Compare with Other Translation Models:</h4>", unsafe_allow_html=True)
                
                # Add other model outputs
                with st.expander("Add other model translations for comparison"):
                    # Let user add other model translations
                    model_name = st.text_input("Model Name", placeholder="e.g., Google Translate, OpenNMT, etc.")
                    model_output = st.text_area("Model's Translation Output", 
                                           placeholder="Enter the translation output from this model...")
                    
                    if st.button("Add Model to Comparison"):
                        if model_name and model_output:
                            # Add the model to our comparison dictionary
                            st.session_state.other_models[model_name] = model_output
                            st.success(f"✅ Added {model_name} to comparison")
                            
                            # Generate new comparison visualization
                            evaluator = TranslationEvaluator()
                            st.session_state.comparison_fig = evaluator.compare_with_models(
                                st.session_state.reference_text,
                                st.session_state.transcription,
                                st.session_state.other_models
                            )
                            st.rerun()
                        else:
                            st.warning("Please enter both model name and its translation output.")
                
                # Display model comparison visualization if we have other models
                if st.session_state.other_models and len(st.session_state.other_models) > 0:
                    st.markdown("<h5>Model Comparison:</h5>", unsafe_allow_html=True)
                    
                    # Show the comparison chart
                    if st.session_state.comparison_fig:
                        st.pyplot(st.session_state.comparison_fig)
                    
                    # Show all model outputs in a table
                    st.markdown("<h5>All Model Outputs:</h5>", unsafe_allow_html=True)
                    
                    # Create and display a comparison table
                    models_data = {
                        "Model": ["Our ISL Model"] + list(st.session_state.other_models.keys()),
                        "Translation Output": [st.session_state.transcription] + list(st.session_state.other_models.values())
                    }
                    
                    # Display the table
                    st.table(models_data)
                    
                    # Option to remove a model
                    if len(st.session_state.other_models) > 0:
                        model_to_remove = st.selectbox("Select model to remove from comparison:", 
                                                    options=list(st.session_state.other_models.keys()))
                        if st.button("Remove Selected Model"):
                            if model_to_remove in st.session_state.other_models:
                                del st.session_state.other_models[model_to_remove]
                                # Regenerate comparison visualization
                                if len(st.session_state.other_models) > 0:
                                    evaluator = TranslationEvaluator()
                                    st.session_state.comparison_fig = evaluator.compare_with_models(
                                        st.session_state.reference_text,
                                        st.session_state.transcription,
                                        st.session_state.other_models
                                    )
                                else:
                                    st.session_state.comparison_fig = None
                                st.success(f"Removed {model_to_remove} from comparison.")
                                st.rerun()
                
                # Add option to reset metrics
                if st.button("Reset All Metrics and Comparisons"):
                    st.session_state.reference_text = None
                    st.session_state.bleu_score = None
                    st.session_state.wer_score = None
                    st.session_state.metrics_fig = None
                    st.session_state.comparison_fig = None
                    st.session_state.other_models = {}
                    st.rerun()
                
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tabs[3]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h4>About this application:</h4>", unsafe_allow_html=True)
            st.markdown("""
            This application converts spoken language to Indian Sign Language (ISL) structure through:
            
            1. **Audio Recording**: Captures clean speech with adaptive noise filtering
            2. **Signal Processing**: Removes noise using wavelets and adaptive filtering
            3. **Speech Recognition**: Transcribes speech to text 
            4. **ISL Conversion**: Restructures English sentences to follow ISL grammar
            
            The application includes multiple evaluation metrics:
            - **Audio Signal Analysis**: Shows the original and processed audio signals
            - **BLEU Score**: Measures translation quality by comparing n-grams
            - **Word Error Rate (WER)**: Measures the difference between transcription and reference
            
            These metrics help assess both the audio processing quality and the accuracy of the translation.
            """)
            st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("Please configure your microphone to continue.")