import tempfile

from transformers import pipeline


def load_models():

    image_model = pipeline(
        "image-classification",
        model="dima806/facial_emotions_image_detection"
    )

    text_model = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        top_k=None
    )

    whisper_model = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny"
    )

    return image_model, text_model, whisper_model


def get_top_prediction(predictions):

    return max(
        predictions,
        key=lambda x: x["score"]
    )


def normalize_text_label(label):

    label = label.lower()

    if "positive" in label:
        return "positive"

    elif "negative" in label:
        return "negative"

    return "neutral"


def analyse_image(model, image):

    predictions = model(image)

    top_prediction = get_top_prediction(predictions)

    emotion = top_prediction["label"]
    score = top_prediction["score"]

    return predictions, emotion, score


def analyse_text(model, text):

    predictions = model(text)[0]

    top_prediction = get_top_prediction(predictions)

    sentiment = normalize_text_label(
        top_prediction["label"]
    )

    score = top_prediction["score"]

    return predictions, sentiment, score


def transcribe_audio(model, uploaded_audio):

    file_extension = uploaded_audio.name.split(".")[-1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=f".{file_extension}"
    ) as temp_audio:

        temp_audio.write(uploaded_audio.read())

        temp_audio_path = temp_audio.name

    result = model(temp_audio_path)

    transcript = result["text"]

    return transcript.strip()