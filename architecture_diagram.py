import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.path import Path
import matplotlib.patheffects as PathEffects
import nltk
plt.switch_backend('Agg')  # Use non-interactive backend

def create_box(ax, x, y, width, height, title, content="", color='#3498db', alpha=0.7):
    """Create a box with title and content"""
    # Create the main box
    rect = mpatches.Rectangle((x, y), width, height, 
                             facecolor=color, alpha=alpha, 
                             edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    
    # Add title on top
    title_y = y + height + 0.02
    txt = ax.text(x + width/2, title_y, title, 
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
    txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    
    # Add content inside box
    if content:
        ax.text(x + width/2, y + height/2, content, 
               ha='center', va='center', fontsize=9,
               wrap=True)
    
    return rect

def draw_arrow(ax, start_box, end_box, label="", color='black', connection_type="right-left"):
    """Draw an arrow between boxes with optional label"""
    if connection_type == "right-left":
        start_x = start_box.get_x() + start_box.get_width()
        start_y = start_box.get_y() + start_box.get_height()/2
        end_x = end_box.get_x()
        end_y = end_box.get_y() + end_box.get_height()/2
    elif connection_type == "bottom-top":
        start_x = start_box.get_x() + start_box.get_width()/2
        start_y = start_box.get_y()
        end_x = end_box.get_x() + end_box.get_width()/2
        end_y = end_box.get_y() + end_box.get_height()
    elif connection_type == "top-bottom":
        start_x = start_box.get_x() + start_box.get_width()/2
        start_y = start_box.get_y() + start_box.get_height()
        end_x = end_box.get_x() + end_box.get_width()/2
        end_y = end_box.get_y()
    
    # Create arrow
    arrow = mpatches.FancyArrowPatch(
        (start_x, start_y), (end_x, end_y),
        arrowstyle='-|>', linewidth=1.5,
        color=color, connectionstyle='arc3,rad=0.1'
    )
    ax.add_patch(arrow)
    
    # Add label to arrow
    if label:
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        ax.text(mid_x, mid_y, label, ha='center', va='center', 
                fontsize=8, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

def draw_architecture_diagram():
    """Create a complete architecture diagram for the Speech-to-ISL system"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # Title
    ax.text(5, 6.7, "Speech to Indian Sign Language (ISL) Conversion System", 
           ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Input section
    input_box = create_box(ax, 0.5, 5.5, 1.5, 0.8, "Audio Input", 
                          "Microphone Input\nAudio Files", color='#9b59b6')
    
    # Recorder Module
    recorder_box = create_box(ax, 2.5, 5.5, 1.5, 0.8, "Audio Recorder", 
                             "PyAudio\nRecording Logic\nAudio Files", color='#3498db')
    
    # Audio Processing Module
    processing_box = create_box(ax, 4.5, 5.5, 1.5, 0.8, "Audio Processor", 
                               "Wavelet Denoising\nAdaptive Filtering\nSignal Enhancement", 
                               color='#2ecc71')
    
    # Speech Recognition
    asr_box = create_box(ax, 6.5, 5.5, 1.5, 0.8, "Speech Recognition", 
                        "Whisper ASR Model\nEnglish Transcription", 
                        color='#e74c3c')
    
    # ISL Conversion
    isl_box = create_box(ax, 6.5, 4.0, 1.5, 0.8, "ISL Converter", 
                        "SOV Reordering\nArticle Dropping\nGrammar Restructuring", 
                        color='#f39c12')
    
    # Output Visualization
    output_box = create_box(ax, 8.5, 5.5, 1.0, 0.8, "Output", 
                           "ISL Text Structure", color='#9b59b6')
    
    # Evaluation Module
    evaluation_box = create_box(ax, 4.5, 2.5, 2.5, 1.0, "Evaluation Module", 
                              "BLEU Score\nWord Error Rate\nMetrics Visualization", 
                              color='#16a085')
    
    # Web Interface
    ui_box = create_box(ax, 0.5, 2.5, 3.0, 1.0, "Web Interface", 
                       "Streamlit UI\nInteractive Components\nMetrics Display", 
                       color='#8e44ad')
    
    # Draw arrows connecting components
    draw_arrow(ax, input_box, recorder_box, "Raw Audio")
    draw_arrow(ax, recorder_box, processing_box, "Audio Files")
    draw_arrow(ax, processing_box, asr_box, "Processed Audio")
    draw_arrow(ax, asr_box, output_box, "Transcription")
    draw_arrow(ax, asr_box, isl_box, "English Text", connection_type="bottom-top")
    draw_arrow(ax, isl_box, output_box, "ISL Structure")
    
    draw_arrow(ax, ui_box, evaluation_box, "Reference Text")
    draw_arrow(ax, asr_box, evaluation_box, "Transcription")
    
    # Create a legend for the color scheme
    legend_elements = [
        mpatches.Patch(facecolor='#9b59b6', edgecolor='black', alpha=0.7, label='Input/Output'),
        mpatches.Patch(facecolor='#3498db', edgecolor='black', alpha=0.7, label='Recording Module'),
        mpatches.Patch(facecolor='#2ecc71', edgecolor='black', alpha=0.7, label='Signal Processing'),
        mpatches.Patch(facecolor='#e74c3c', edgecolor='black', alpha=0.7, label='Speech Recognition'),
        mpatches.Patch(facecolor='#f39c12', edgecolor='black', alpha=0.7, label='ISL Conversion'),
        mpatches.Patch(facecolor='#16a085', edgecolor='black', alpha=0.7, label='Evaluation'),
        mpatches.Patch(facecolor='#8e44ad', edgecolor='black', alpha=0.7, label='User Interface')
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.05),
              ncol=4, fontsize=9)
    
    # Add system details and components in a textbox
    components_text = """
    System Components:
    - Recorder Module: Audio capture and device management
    - Signal Processing: Wavelet decomposition, adaptive filtering, noise removal
    - Speech Recognition: Whisper ASR model for transcription
    - ISL Conversion: Rule-based English to ISL structure conversion
    - Evaluation: BLEU and WER metrics for quality assessment
    - User Interface: Streamlit-based web application
    """
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax.text(0.02, 0.2, components_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=props)
    
    # Save the diagram
    plt.tight_layout()
    plt.savefig('architecture_diagram.png', dpi=300, bbox_inches='tight')
    return fig

if __name__ == "__main__":
    # Create and display the architecture diagram
    fig = draw_architecture_diagram()
    plt.show()
