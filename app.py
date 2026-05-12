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
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
        color: white;
    }

    .stApp {
        background: linear-gradient(to bottom right, #0f172a, #1e293b);
    }

    h1, h2, h3 {
        color: #f8fafc;
    }

    .glass-box {
        background: rgba(255,255,255,0.08);
        padding: 20px;
        border-radius: 20px;
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
        border: 1px solid rgba(255,255,255,0.1);
    }

    .metric-card {
        background: rgba(255,255,255,0.08);
        padding: 15px;
        border-radius: 15px;
        text-align: center;
    }

    .big-font {
        font-size: 20px;
        font-weight: bold;
    }

    .small-font {
        color: #cbd5e1;
    }

    .stButton>button {
        background: linear-gradient(to right, #3b82f6, #8b5cf6);
        color: white;
        border-radius: 12px;
        border: none;
        height: 50px;
        width: 100%;
        font-size: 18px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- HEADER ----------------
st.markdown(
    """
    <div class="glass-box">
        <h1>MoodSyncAI</h1>
        <p class="small-font">
        Multi-Modal Emotion & Sentiment Analysis using Image, Audio, Text and Webcam.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Settings")

# st.sidebar.info(
#     "This application analyses emotions using multiple modalities."
# )

input_mode = st.sidebar.radio(
    "Choose Input Mode",
    ["Upload Image", "Use Webcam"]
)

show_charts = st.sidebar.checkbox(
    "Show Confidence Charts",
    value=True
)

show_timeline = st.sidebar.checkbox(
    "Show Webcam Timeline",
    value=True
)

# ---------------- LOAD MODELS ----------------
image_model, text_model, whisper_model = load_models()

if "emotion_timeline" not in st.session_state:
    st.session_state.emotion_timeline = []

# ---------------- INPUT SECTION ----------------
st.subheader("Input Section")

col1, col2 = st.columns(2)

uploaded_image = None
webcam_image = None

with col1:

    st.markdown("### Face Input")

    if input_mode == "Upload Image":

        uploaded_image = st.file_uploader(
            "Upload a face image",
            type=["jpg", "jpeg", "png"]
        )

    else:

        webcam_image = st.camera_input(
            "Capture image from webcam"
        )

with col2:

    st.markdown("### Audio & Text")

    uploaded_audio = st.file_uploader(
        "Upload audio clip (optional)",
        type=["wav", "mp3", "m4a"]
    )

    user_text = st.text_area(
        "Enter spoken sentence",
        placeholder="Example: No, everything is perfectly fine."
    )

# ---------------- ANALYSIS BUTTON ----------------
if st.button("Analyse Emotion"):

    image_source = uploaded_image if input_mode == "Upload Image" else webcam_image

    if image_source is None:
        st.error("Please upload or capture an image.")

    elif user_text.strip() == "" and uploaded_audio is None:
        st.error("Please enter text or upload audio.")

    else:

        image = Image.open(image_source).convert("RGB")

        st.subheader("Captured Face")
        st.image(image, width=350)

        # IMAGE ANALYSIS
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

        # AUDIO + TEXT
        final_text = user_text.strip()
        audio_transcript = ""

        if uploaded_audio is not None:

            st.audio(uploaded_audio)

            with st.spinner("Transcribing audio..."):

                audio_transcript = transcribe_audio(
                    whisper_model,
                    uploaded_audio
                )

            st.success(f"Transcript: {audio_transcript}")

            if final_text == "":
                final_text = audio_transcript
            else:
                final_text = final_text + " " + audio_transcript

        # TEXT SENTIMENT
        text_predictions, text_sentiment, text_score = analyse_text(
            text_model,
            final_text
        )

        # AUDIO SENTIMENT
        audio_sentiment = "not provided"
        audio_score = 0.0
        audio_predictions = None

        if audio_transcript.strip() != "":

            audio_predictions, audio_sentiment, audio_score = analyse_text(
                text_model,
                audio_transcript
            )

        # FUSION
        fusion_result = fusion_logic(
            image_emotion=image_emotion,
            image_score=image_score,
            text_sentiment=text_sentiment,
            text_score=text_score,
            audio_sentiment=audio_sentiment,
            audio_score=audio_score
        )

        # ---------------- RESULTS ----------------
        st.subheader("📊 Analysis Results")

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric(
                "Visual Emotion",
                image_emotion,
                f"{round(image_score * 100, 2)}%"
            )

        with m2:
            st.metric(
                "Text Sentiment",
                text_sentiment.capitalize(),
                f"{round(text_score * 100, 2)}%"
            )

        with m3:

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

        with m4:
            st.metric(
                "Fusion Result",
                fusion_result["badge"],
                f'{fusion_result["confidence"]}%'
            )

        # ---------------- CHARTS ----------------
        if show_charts:

            st.subheader("📈 Confidence Visualisations")

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

        # ---------------- TIMELINE ----------------
        if (
            input_mode == "Use Webcam"
            and len(st.session_state.emotion_timeline) > 0
            and show_timeline
        ):

            st.subheader("🎥 Webcam Emotion Timeline")

            st.plotly_chart(
                create_timeline_chart(
                    st.session_state.emotion_timeline
                ),
                use_container_width=True
            )

        # ---------------- SUMMARY ----------------
        st.subheader("📝 Generative Summary")

        summary = generate_summary(
            image_emotion=image_emotion,
            image_sentiment=fusion_result["image_sentiment"],
            text_sentiment=text_sentiment,
            audio_sentiment=audio_sentiment,
            fusion_status=fusion_result["status"],
            audio_used=audio_transcript.strip() != ""
        )

        st.success(summary)

        # ---------------- FOOTER ----------------
st.markdown("---")

st.markdown(
    unsafe_allow_html=True
)

