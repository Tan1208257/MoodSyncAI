# MoodSyncAI — Multi-Modal Sentiment & Emotion Analyser

## Live Website: https://tanrj-moodsyncai.hf.space
MoodSyncAI is a multi-modal AI application that analyses emotions and sentiment using:

- Facial emotion recognition from images/webcam
- Text sentiment analysis
- Audio transcription and sentiment analysis
- Multimodal fusion logic
- Generative emotional summaries

The system combines image, text, and audio modalities to detect emotional alignment and mismatch.

---

# Features

## Image Emotion Recognition
- Upload face images
- Webcam capture support
- CNN-based facial emotion detection
- Emotion confidence visualization

## Text Sentiment Analysis
- Transformer-based sentiment classification
- Positive / Negative / Neutral prediction

## Audio Sentiment Analysis
- Upload short audio clips
- Automatic speech transcription using Whisper
- Sentiment analysis on transcribed speech

## Multimodal Fusion Layer
- Combines:
  - visual emotion
  - text sentiment
  - audio sentiment
- Detects:
  - Aligned emotions
  - Partial mismatch
  - Full mismatch

## Webcam Emotion Timeline
- Real-time webcam emotion capture
- Timeline visualization of emotion changes

## Generative Summary
- Produces natural-language emotional explanations

---

# Technologies Used

## Frontend
- Streamlit

## Deep Learning Framework
- PyTorch
- Hugging Face Transformers

## Models Used

### Facial Emotion Recognition
```text
dima806/facial_emotions_image_detection
```

### Text Sentiment Analysis
```text
cardiffnlp/twitter-roberta-base-sentiment-latest
```

### Audio Transcription
```text
openai/whisper-tiny
```

## Visualization
- Plotly
- Pandas

## Deployment
- Hugging Face Spaces
- Docker

---

# System Architecture

```text
Image / Webcam
       ↓
CNN Emotion Recognition
       ↓
Visual Emotion

Text Input
       ↓
RoBERTa Transformer
       ↓
Text Sentiment

Audio Input
       ↓
Whisper Transformer
       ↓
Transcript
       ↓
RoBERTa Sentiment
       ↓
Audio Sentiment

All Modalities
       ↓
Fusion Layer
       ↓
Mismatch Detection
       ↓
Generative Summary
       ↓
Streamlit UI
```
---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/MoodSyncAI.git
cd MoodSyncAI
```

---

# Create Virtual Environment

## Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## Windows

```bash
python -m venv venv
venv\\Scripts\\activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Application

```bash
streamlit run app.py
```

Application runs at:

```text
http://localhost:8501
```

---

# Hugging Face Deployment

The application is deployed using Hugging Face Spaces with Docker support.

---

# Author

Tania Rose Jobi

---

# Academic Module

Data Analytics-3 — Deep Learning & Generative AI

Instructor:
Prof. Dr. Gayan de Silva
