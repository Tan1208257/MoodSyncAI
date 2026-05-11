import streamlit as st
from PIL import Image

from models import (
    load_models,
    analyse_image,
    analyse_text,
    transcribe_audio
)

from fusion import (
    fusion_logic,
    generate_summary,
    create_bar_chart,
    create_timeline_chart
)

st.set_page_config(
    page_title="MoodSyncAI",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 MoodSyncAI")

st.write(
    "Multi-modal emotion and sentiment analyser using image, text, audio, and webcam input."
)

image_model, text_model, whisper_model = load_models()

if "emotion_timeline" not in st.session_state:
    st.session_state.emotion_timeline = []

input_mode = st.radio(
    "Choose image input mode",
    ["Upload Image", "Use Webcam"]
)

uploaded_image = None
webcam_image = None

if input_mode == "Upload Image":
    uploaded_image = st.file_uploader(
        "Upload face image",
        type=["jpg", "jpeg", "png"]
    )

else:
    webcam_image = st.camera_input(
        "Capture face from webcam"
    )

uploaded_audio = st.file_uploader(
    "Upload audio clip (optional)",
    type=["wav", "mp3", "m4a"]
)

user_text = st.text_area(
    "Enter text"
)

if st.button("Analyse Emotion"):

    image_source = uploaded_image if input_mode == "Upload Image" else webcam_image

    if image_source is None:
        st.error("Please upload an image or capture from webcam.")

    elif user_text.strip() == "" and uploaded_audio is None:
        st.error("Please enter text or upload audio.")

    else:
        image = Image.open(image_source).convert("RGB")

        st.image(image, width=300)

        image_predictions, image_emotion, image_score = analyse_image(
            image_model,
            image
        )

        if input_mode == "Use Webcam":
            st.session_state.emotion_timeline.append(
                {
                    "frame": len(st.session_state.emotion_timeline) + 1,
                    "emotion": image_emotion,
                    "confidence": round(image_score * 100, 2)
                }
            )

        final_text = user_text.strip()
        audio_transcript = ""

        if uploaded_audio is not None:

            st.audio(uploaded_audio)

            audio_transcript = transcribe_audio(
                whisper_model,
                uploaded_audio
            )

            st.info(f"Audio Transcript: {audio_transcript}")

            if final_text == "":
                final_text = audio_transcript
            else:
                final_text = final_text + " " + audio_transcript

        text_predictions, text_sentiment, text_score = analyse_text(
            text_model,
            final_text
        )

        audio_sentiment = "not provided"
        audio_score = 0.0
        audio_predictions = None

        if audio_transcript.strip() != "":

            audio_predictions, audio_sentiment, audio_score = analyse_text(
                text_model,
                audio_transcript
            )

        fusion_result = fusion_logic(
            image_emotion=image_emotion,
            image_score=image_score,
            text_sentiment=text_sentiment,
            text_score=text_score,
            audio_sentiment=audio_sentiment,
            audio_score=audio_score
        )

        st.divider()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Visual Emotion",
                image_emotion,
                f"{round(image_score * 100, 2)}%"
            )

        with col2:
            st.metric(
                "Text Sentiment",
                text_sentiment.capitalize(),
                f"{round(text_score * 100, 2)}%"
            )

        with col3:
            if audio_transcript.strip() != "":
                st.metric(
                    "Audio Sentiment",
                    audio_sentiment.capitalize(),
                    f"{round(audio_score * 100, 2)}%"
                )
            else:
                st.metric(
                    "Audio Sentiment",
                    "Not Provided",
                    "Optional"
                )

        with col4:
            st.metric(
                "Fusion Result",
                fusion_result["badge"],
                f'{fusion_result["confidence"]}%'
            )

        st.divider()

        st.plotly_chart(
            create_bar_chart(
                image_predictions,
                "Visual Emotion Confidence"
            ),
            use_container_width=True
        )

        st.plotly_chart(
            create_bar_chart(
                text_predictions,
                "Text Sentiment Confidence"
            ),
            use_container_width=True
        )

        if audio_predictions is not None:
            st.plotly_chart(
                create_bar_chart(
                    audio_predictions,
                    "Audio Sentiment Confidence"
                ),
                use_container_width=True
            )

        if input_mode == "Use Webcam" and len(st.session_state.emotion_timeline) > 0:
            st.subheader("Webcam Emotion Timeline")

            st.plotly_chart(
                create_timeline_chart(st.session_state.emotion_timeline),
                use_container_width=True
            )

            if st.button("Clear Webcam Timeline"):
                st.session_state.emotion_timeline = []
                st.rerun()

        st.divider()

        summary = generate_summary(
            image_emotion=image_emotion,
            image_sentiment=fusion_result["image_sentiment"],
            text_sentiment=text_sentiment,
            audio_sentiment=audio_sentiment,
            fusion_status=fusion_result["status"],
            audio_used=audio_transcript.strip() != ""
        )

        st.subheader("Generative Summary")
        st.success(summary)