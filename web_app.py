from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import time
import base64
import uuid
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
from io import BytesIO

# Import the local modules
from recorder import AudioRecorder
from isl_convert import EnglishToISLSentenceConverter
from translation_evaluator import TranslationEvaluator

app = Flask(__name__, static_folder='static', template_folder='templates')

# Global variables
UPLOAD_FOLDER = 'recordings'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for HTML embedding"""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    """Process uploaded audio file or recorded audio blob"""
    try:
        # Check if the post request has the file part
        if 'audio_file' not in request.files and 'audio_blob' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = None
        if 'audio_file' in request.files:
            file = request.files['audio_file']
            if file and allowed_file(file.filename):
                unique_filename = f"upload_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}.wav"
                audio_path = os.path.join(UPLOAD_FOLDER, unique_filename)
                file.save(audio_path)
                audio_file = audio_path
            else:
                return jsonify({'error': 'Invalid file format'}), 400
        
        elif 'audio_blob' in request.files:
            file = request.files['audio_blob']
            unique_filename = f"recording_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}.wav"
            audio_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            file.save(audio_path)
            audio_file = audio_path
        
        # Process the audio file
        recorder = AudioRecorder()
        processed_file, fig, transcription = recorder.process_audio(audio_file)
        
        if not transcription:
            return jsonify({'error': 'Speech recognition failed. Please try again with clearer audio.'}), 400
        
        # Convert to ISL
        converter = EnglishToISLSentenceConverter()
        isl_output = converter.convert(transcription)
        
        # Calculate BLEU score if reference text is provided
        bleu_fig = None
        bleu_score = None
        bleu_interpretation = None
        bleu_scores = {}
        
        if 'reference_text' in request.form and request.form['reference_text'].strip():
            reference_text = request.form['reference_text']
            evaluator = TranslationEvaluator()
            bleu_score = evaluator.calculate_bleu(reference_text, transcription)
            bleu_scores = evaluator.calculate_bleu_components(reference_text, transcription)
            bleu_interpretation = evaluator.interpret_bleu_score(bleu_score)
            bleu_fig = evaluator.generate_visualization(reference_text, transcription)
        
        # Convert matplotlib figure to base64 string for HTML embedding
        audio_fig_base64 = fig_to_base64(fig)
        bleu_fig_base64 = fig_to_base64(bleu_fig) if bleu_fig else None
        
        # Create relative path for audio playback
        processed_audio_url = f"/audio/{os.path.basename(processed_file)}" if processed_file else None
        
        response = {
            'success': True,
            'transcription': transcription,
            'isl_output': isl_output,
            'processed_audio_url': processed_audio_url,
            'audio_fig': audio_fig_base64,
            'bleu_score': bleu_score,
            'bleu_scores': bleu_scores,
            'bleu_interpretation': bleu_interpretation,
            'bleu_fig': bleu_fig_base64
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': f'Error processing audio: {str(e)}'}), 500

@app.route('/audio/<filename>')
def get_audio(filename):
    """Serve processed audio files"""
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
