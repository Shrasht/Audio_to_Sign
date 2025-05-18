import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from jiwer import wer
import numpy as np
import matplotlib.pyplot as plt

class TranslationEvaluator:
    """Class for evaluating the quality of translations using BLEU and WER"""

    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def calculate_bleu(self, reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)):
        reference_tokens = nltk.word_tokenize(reference.lower())
        candidate_tokens = nltk.word_tokenize(candidate.lower())
        references = [reference_tokens]
        smoothie = SmoothingFunction().method1
        return sentence_bleu(references, candidate_tokens, weights=weights, smoothing_function=smoothie)

    def calculate_bleu_components(self, reference, candidate):
        return {
            'bleu-1': self.calculate_bleu(reference, candidate, weights=(1, 0, 0, 0)),
            'bleu-2': self.calculate_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)),
            'bleu-3': self.calculate_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)),
            'bleu-4': self.calculate_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)),
            'bleu-cumulative': self.calculate_bleu(reference, candidate)
        }

    def calculate_wer(self, reference, hypothesis):
        if not reference or not hypothesis:
            return None
        return wer(reference.lower(), hypothesis.lower())

    def generate_visualization(self, reference, candidate):
        scores = self.calculate_bleu_components(reference, candidate)
        categories = ['1-gram', '2-gram', '3-gram', '4-gram', 'Cumulative']
        values = [scores['bleu-1'], scores['bleu-2'], scores['bleu-3'], scores['bleu-4'], scores['bleu-cumulative']]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(categories, values, color=['#3498db', '#2980b9', '#1f618d', '#154360', '#5DADE2'])

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.3f}', ha='center', va='bottom', fontsize=10)

        ax.set_xlabel('N-gram Level')
        ax.set_ylabel('BLEU Score')
        ax.set_title('BLEU Score Analysis')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        return fig

    def interpret_bleu_score(self, score):
        if score < 0.1:
            return "Very poor translation quality"
        elif score < 0.2:
            return "Poor translation quality"
        elif score < 0.4:
            return "Fair translation quality"
        elif score < 0.6:
            return "Good translation quality"
        elif score < 0.8:
            return "Very good translation quality"
        else:
            return "Excellent translation quality"
 
    def generate_combined_metrics_visualization(self, reference, candidate):
        """
        Generate a combined bar plot of BLEU scores and WER.

        Args:
            reference (str): Ground truth sentence.
            candidate (str): Predicted sentence.

        Returns:
            matplotlib.figure.Figure: A figure containing BLEU and WER scores.
        """
        bleu_scores = self.calculate_bleu_components(reference, candidate)
        wer_score = self.calculate_wer(reference, candidate)

        labels = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'BLEU-Cumulative', 'WER']
        values = [
            bleu_scores['bleu-1'],
            bleu_scores['bleu-2'],
            bleu_scores['bleu-3'],
            bleu_scores['bleu-4'],
            bleu_scores['bleu-cumulative'],
            wer_score
        ]

        colors = ['#3498db'] * 5 + ['#e74c3c']

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(labels, values, color=colors)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')

        ax.set_title("Translation Metrics: BLEU and WER")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        plt.tight_layout()
        return fig
    
    def interpret_wer(self, wer_score):
        """
        Provide a human-readable interpretation of the Word Error Rate (WER).

        Args:
            wer_score (float): WER value between 0 and 1.

        Returns:
            str: Interpretation of the WER score.
        """
        if wer_score is None:
            return "WER is unavailable."

        if wer_score < 0.1:
            return "Excellent transcription quality"
        elif wer_score < 0.2:
            return "Very good transcription quality"
        elif wer_score < 0.4:
            return "Good transcription quality"
        elif wer_score < 0.6:
            return "Fair transcription quality"
        elif wer_score < 0.8:
            return "Poor transcription quality"
        else:
            return "Very poor transcription quality"


    
    def compare_with_models(self, reference, candidates_dict):
     """
     Compare multiple model outputs using BLEU and WER, and generate a bar chart.

     Args:
        reference (str): The ground truth transcription.
        candidates_dict (dict): Dictionary of model_name -> candidate_transcription.

     Returns:
        matplotlib.figure.Figure: Comparison chart showing BLEU and WER for each model.
     """
     bleu_scores = []
     wer_scores = []
     model_names = []

     for model_name, candidate in candidates_dict.items():
        bleu = self.calculate_bleu(reference, candidate)
        wer_score = self.calculate_wer(reference, candidate)

        model_names.append(model_name)
        bleu_scores.append(bleu)
        wer_scores.append(wer_score)

     # Plotting
     x = np.arange(len(model_names))
     width = 0.35

     fig, ax = plt.subplots(figsize=(10, 6))
     bars1 = ax.bar(x - width / 2, bleu_scores, width, label='BLEU', color='#3498db')
     bars2 = ax.bar(x + width / 2, wer_scores, width, label='WER', color='#e74c3c')

     for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

     ax.set_ylabel('Score')
     ax.set_title('Model Comparison: BLEU and WER')
     ax.set_xticks(x)
     ax.set_xticklabels(model_names)
     ax.legend()
     ax.set_ylim(0, 1)
     ax.grid(axis='y', linestyle='--', alpha=0.4)

     fig.tight_layout()
     return fig
