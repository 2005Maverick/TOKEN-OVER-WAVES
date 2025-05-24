# 🎵 Token over Waves: Advanced Audio Processing System

## Overview

Token over Waves is a state-of-the-art deep learning system for detecting fake audio using transformer-based, multi-modal analysis. It leverages both waveform and spectrogram features, advanced attention mechanisms, and robust training strategies to achieve high accuracy in audio authenticity detection.

---

## System Architecture

### Model Overview

The architecture consists of several key components:

- **Audio Transformer Encoder:** Processes raw waveform data to extract temporal features.
- **Spectrogram Transformer Encoder:** Analyzes mel spectrograms for frequency-based features.
- **Cross-Modal Attention:** Fuses information from both modalities for richer representations.
- **Emotional Coherence Decoder:** Detects inconsistencies in emotional patterns.
- **Artifact Detector:** Identifies artificial artifacts in audio signals.
- **Classifier Head:** Makes the final real/fake prediction.

<div align="center">
  <img src="AUDIO_T/assets/images/model_architecture.png" alt="Model Architecture" width="600"/>
  <br>
  <em>Figure 1: The overall architecture of the Token over Waves system, showing the dual-branch transformer encoders and fusion modules.</em>
</div>

---

## Training and Evaluation

### Training Progress

The following graph shows the model's training and validation loss over epochs, indicating effective learning and convergence.

<div align="center">
  <img src="AUDIO_T/assets/images/training_metrics.jpg" alt="Training Metrics" width="600"/>
  <br>
  <em>Figure 2: Training and validation loss curves. The steady decrease and convergence indicate successful model training.</em>
</div>

---

## Results and Analysis

### Evaluation Metrics

The model achieves high accuracy, precision, recall, and F1-score, as shown below:

<div align="center">
  <img src="AUDIO_T/assets/images/evaluation_metrics.jpg" alt="Evaluation Metrics" width="600"/>
  <br>
  <em>Figure 3: Evaluation metrics for the trained model, demonstrating strong performance on the test set.</em>
</div>

### Confusion Matrix

The confusion matrix provides insight into the model's classification performance, showing the number of true/false positives and negatives.

<div align="center">
  <img src="AUDIO_T/assets/images/confusion_matrix.jpg" alt="Confusion Matrix" width="600"/>
  <br>
  <em>Figure 4: Confusion matrix for the test set predictions.</em>
</div>

### Feature Analysis

#### t-SNE Feature Visualization

This plot visualizes the learned feature representations using t-SNE, showing clear separation between real and fake audio samples.

<div align="center">
  <img src="AUDIO_T/assets/images/tsne_features.jpg" alt="t-SNE Features" width="600"/>
  <br>
  <em>Figure 5: t-SNE visualization of the model's learned features, highlighting the model's ability to distinguish between classes.</em>
</div>

#### Emotional Coherence Distribution

The following graph shows the distribution of emotional coherence scores, which helps in identifying unnatural emotional patterns in fake audio.

<div align="center">
  <img src="AUDIO_T/assets/images/emotional_coherence.jpg" alt="Emotional Coherence" width="600"/>
  <br>
  <em>Figure 6: Distribution of emotional coherence scores for real and fake audio samples.</em>
</div>

### Output Example

Below is an example of the system's output, including the predicted label and confidence score.

<div align="center">
  <img src="AUDIO_T/assets/images/output_example.jpg" alt="Output Example" width="600"/>
  <br>
  <em>Figure 7: Example output from the system, showing prediction and analysis details.</em>
</div>

---

## 📈 Evaluation & Performance

### 🔍 Metrics

| Metric     | Score   |
|------------|---------|
| Accuracy   | 95.8%   |
| F1 Score   | 0.942   |
| Precision  | 0.938   |
| Recall     | 0.946   |

---

## Usage

**Training:**
```python
from audio_detection import DeepfakeAudioDetectionSystem

detection_system = DeepfakeAudioDetectionSystem(
    data_dir='path/to/your/data',
    batch_size=32,
    num_workers=8,
    max_epochs=10
)
best_model_path = detection_system.train()
```

**Inference:**
```python
result = detection_system.predict('path/to/audio/file.wav')
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Emotional Coherence: {result['emotional_coherence']:.2f}")
```

---

## Configuration

Key parameters can be adjusted in the configuration:

```python
EMBEDDING_DIM = 768
NUM_HEADS = 8
NUM_LAYERS = 6
DROPOUT_RATE = 0.1
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
MAX_EPOCHS = 10
```

---

## Acknowledgments

- PyTorch and PyTorch Lightning teams
- Open-source contributors

