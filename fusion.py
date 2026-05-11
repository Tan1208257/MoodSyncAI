import pandas as pd
import plotly.express as px


def map_emotion_to_sentiment(emotion):
    emotion = emotion.lower()

    positive_emotions = ["happy", "surprise"]
    negative_emotions = ["sad", "angry", "fear", "fearful", "disgust"]

    if emotion in positive_emotions:
        return "positive"
    elif emotion in negative_emotions:
        return "negative"
    return "neutral"


def fusion_logic(
    image_emotion,
    image_score,
    text_sentiment,
    text_score,
    audio_sentiment="not provided",
    audio_score=0.0
):
    image_sentiment = map_emotion_to_sentiment(image_emotion)

    modality_sentiments = [image_sentiment, text_sentiment]
    modality_scores = [image_score, text_score]

    if audio_sentiment != "not provided":
        modality_sentiments.append(audio_sentiment)
        modality_scores.append(audio_score)

    sentiment_counts = {}

    for sentiment in modality_sentiments:
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1

    majority_sentiment = max(sentiment_counts, key=sentiment_counts.get)
    unique_sentiments = set(modality_sentiments)

    if len(unique_sentiments) == 1:
        status = "ALIGNED"
        badge = "🟢 Aligned"
    elif len(unique_sentiments) == 2 and sentiment_counts[majority_sentiment] > 1:
        status = "PARTIAL MISMATCH"
        badge = "🟡 Partial Mismatch"
    else:
        status = "MISMATCH DETECTED"
        badge = "🟠 Mismatch Detected"

    confidence = round(sum(modality_scores) / len(modality_scores) * 100, 2)

    return {
        "image_sentiment": image_sentiment,
        "majority_sentiment": majority_sentiment,
        "status": status,
        "badge": badge,
        "confidence": confidence
    }


def generate_summary(
    image_emotion,
    image_sentiment,
    text_sentiment,
    audio_sentiment,
    fusion_status,
    audio_used=False
):
    if audio_used:
        modality_description = (
            f"The image suggests {image_sentiment} emotion through a facial expression "
            f"classified as {image_emotion}. The typed text shows {text_sentiment} sentiment, "
            f"while the audio transcript shows {audio_sentiment} sentiment."
        )
    else:
        modality_description = (
            f"The image suggests {image_sentiment} emotion through a facial expression "
            f"classified as {image_emotion}. The typed text shows {text_sentiment} sentiment."
        )

    if fusion_status == "ALIGNED":
        return (
            modality_description
            + " All available modalities are emotionally aligned, suggesting consistent emotional expression."
        )

    if fusion_status == "PARTIAL MISMATCH":
        return (
            modality_description
            + " The system detected a partial mismatch. Some emotional signals agree, but at least one modality differs."
        )

    return (
        modality_description
        + " The system detected a clear mismatch between the modalities."
    )


def create_bar_chart(predictions, title):
    df = pd.DataFrame(predictions)
    df["score"] = df["score"] * 100

    fig = px.bar(
        df,
        x="label",
        y="score",
        title=title,
        text=df["score"].round(2)
    )

    fig.update_layout(
        yaxis_title="Confidence (%)",
        xaxis_title="Class"
    )

    return fig


def create_timeline_chart(timeline_data):
    df = pd.DataFrame(timeline_data)

    fig = px.line(
        df,
        x="frame",
        y="confidence",
        color="emotion",
        markers=True,
        title="Webcam Emotion Changes Over Time"
    )

    fig.update_layout(
        xaxis_title="Captured Frame",
        yaxis_title="Confidence (%)"
    )

    return fig