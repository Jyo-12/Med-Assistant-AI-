"""
app.py
 
MedAssist AI
============
 
Main Streamlit Application
 
Part 1
-------
✓ Imports
✓ Page Configuration
✓ Session State
✓ Sidebar Navigation
✓ Component Initialization
✓ Home Page
"""
 
import os
from pathlib import Path
import pandas as pd
import streamlit as st
from PIL import Image
 
# -----------------------------
# Chat / RAG
# -----------------------------
 
from chat_engine import ChatEngine
 
# -----------------------------
# Vision
# -----------------------------
 
from vision.vision_engine import VisionEngine
 
# -----------------------------
# Speech
# -----------------------------
 
from modules.speech import SpeechAssistant as SpeechEngine
 
# -----------------------------
# -----------------------------
 
from vision.report_generator import MedicalPDFReportAnalyzer as ReportGenerator
 
# -----------------------------
# Configuration
# -----------------------------
 
MODEL_PATH = "models/medicalcnn.pth"
 
DOCUMENT_FOLDER = "documents"
 
UPLOAD_FOLDER = "uploads"
UPLOAD_IMAGE_FOLDER = "uploads/images"
UPLOAD_REPORT_FOLDER = "uploads/reports"
GRADCAM_FOLDER = "uploads/gradcam"
GENERATED_REPORT_FOLDER = "reports"

for folder in [
    DOCUMENT_FOLDER,
    UPLOAD_FOLDER,
    UPLOAD_IMAGE_FOLDER,
    UPLOAD_REPORT_FOLDER,
    GRADCAM_FOLDER,
    GENERATED_REPORT_FOLDER,
]:
    Path(folder).mkdir(parents=True, exist_ok=True)
 
# -----------------------------
# Streamlit Configuration
# -----------------------------
 
st.set_page_config(
    page_title="MedAssist AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# -----------------------------
# Session State
# -----------------------------
 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
 
if "vision_engine" not in st.session_state:
    st.session_state.vision_engine = None
 
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None
 
if "speech_engine" not in st.session_state:
    st.session_state.speech_engine = None
 
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
 
if "report_history" not in st.session_state:
    st.session_state.report_history = []
 
# -----------------------------
# Load Engines
# -----------------------------
 
@st.cache_resource
def load_chat_engine():
    return ChatEngine()
 
 
@st.cache_resource
def load_vision_engine():
 
    if not os.path.exists(MODEL_PATH):
        return None
 
    return VisionEngine(
        model_path=MODEL_PATH
    )
 
 
@st.cache_resource
def load_speech_engine():
    return SpeechEngine()
 
 
# -----------------------------
# Initialize Components
# -----------------------------
 
if st.session_state.vision_engine is None:
    st.session_state.vision_engine = load_vision_engine()
 
if st.session_state.speech_engine is None:
    st.session_state.speech_engine = load_speech_engine()
 
# -----------------------------
# Sidebar
# -----------------------------
 
st.sidebar.title("🏥 MedAssist AI")
 
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🤖 AI Medical Chat",
        "🖼 Medical Image Analysis",
        "📄 Medical Report Analyzer",
        "📊 Health Dashboard",
        "🎤 Voice Assistant",
        "ℹ About",
    ]
)
 
st.sidebar.markdown("---")
 
st.sidebar.success("System Ready")
 
# -----------------------------
# Home Page
# -----------------------------
 
if page == "🏠 Home":
 
    st.title("🏥 MedAssist AI")
 
    st.markdown(
        """
### Intelligent Medical Assistant
 
MedAssist AI combines:
 
- 🤖 Gemini AI Chatbot
- 📄 Medical Report Retrieval (RAG)
- 🖼 Medical Image Classification
- 🔥 Grad-CAM Explainability
- 🎤 Voice Assistant
- 📚 FAISS Vector Search
 
Use the navigation menu on the left to access each module.
"""
    )
 
    col1, col2 = st.columns(2)
 
    with col1:
 
        st.info(
            """
### Vision AI
 
✔ Chest X-rays
 
✔ MRI
 
✔ CT
 
✔ Retina
 
✔ Skin Lesions
"""
        )
 
    with col2:
 
        st.info(
            """
### AI Assistant
 
✔ Medical Questions
 
✔ Document Search
 
✔ Voice Conversation
 
✔ Report Generation
"""
        )
 
    st.markdown("---")
 
    st.success("Welcome to MedAssist AI.")
 
# ============================================================
# PAGE 2: AI MEDICAL CHAT
# ============================================================
 
elif page == "🤖 AI Medical Chat":
 
    st.title("🤖 AI Medical Assistant")
    st.markdown(
        "Ask health-related questions powered by the MedAssist AI Medical Chat Engine."
    )
 
    # --------------------------------------------------------
    # Session State Initialization
    # --------------------------------------------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
 
    if "chat_engine" not in st.session_state or st.session_state.chat_engine is None:
        try:
            st.session_state.chat_engine = load_chat_engine()
        except Exception as e:
            st.error(
                "Unable to initialize AI Chat Engine.\n\n"
                f"{e}\n\n"
                "If the FAISS index is missing, run `python build_index.py` "
                "from the project folder first."
            )
            st.stop()
 
    chat_engine = st.session_state.chat_engine
 
    # --------------------------------------------------------
    # Chat Container
    # --------------------------------------------------------
    chat_container = st.container()
 
    with chat_container:
 
        if len(st.session_state.chat_history) == 0:
            st.info(
                "👋 Welcome to MedAssist AI.\n\n"
                "You can ask questions about diseases, symptoms, medications, "
                "healthy lifestyle, first aid, nutrition, or medical reports."
            )
 
        for message in st.session_state.chat_history:
 
            role = message.get("role", "assistant")
            content = message.get("content", "")
 
            with st.chat_message(role):
                st.markdown(content)
 
    # --------------------------------------------------------
    # User Input
    # --------------------------------------------------------
    user_prompt = st.chat_input(
        "Ask a medical question..."
    )
 
    # --------------------------------------------------------
    # Process User Prompt
    # --------------------------------------------------------
    if user_prompt:
 
        # Store user message
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_prompt
            }
        )
 
        with st.chat_message("user"):
            st.markdown(user_prompt)
 
        # Generate Response
        with st.chat_message("assistant"):
 
            placeholder = st.empty()
 
            try:
 
                with st.spinner("🩺 Thinking..."):
 
                    response = chat_engine.generate_response(
                        user_prompt
                    )
 
                    if response is None:
                        response = (
                            "I'm sorry, I couldn't generate a response."
                        )
 
                    if not isinstance(response, str):
                        response = str(response)
 
                placeholder.markdown(response)
 
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": response
                    }
                )
 
            except Exception as e:
 
                error_message = (
                    "⚠️ An unexpected error occurred while processing "
                    "your request.\n\n"
                    f"**Details:** {e}"
                )
 
                placeholder.error(error_message)
 
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": error_message
                    }
                )
 
    # --------------------------------------------------------
    # Sidebar Chat Controls
    # --------------------------------------------------------
    with st.sidebar:
 
        st.markdown("---")
        st.subheader("💬 Chat Controls")
 
        if st.button(
            "🗑️ Clear Chat",
            use_container_width=True
        ):
            st.session_state.chat_history = []
            st.rerun()
 
        st.caption(
            "Your conversation is stored only for the current "
            "Streamlit session."
        )
 
# ============================================================
# PAGE 3: MEDICAL IMAGE ANALYSIS
# ============================================================
 
elif page == "🖼 Medical Image Analysis":
 
    st.title("🖼 Medical Image Analysis")
    st.markdown(
        "Upload a medical image for analysis. Uploaded images are saved in `uploads/images`."
    )
 
    uploaded_image = st.file_uploader(
        "Upload Medical Image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        accept_multiple_files=False
    )
 
    if uploaded_image is not None:
 
        safe_name = Path(uploaded_image.name).name
        image_path = Path(UPLOAD_IMAGE_FOLDER) / safe_name
        image_path.write_bytes(uploaded_image.getbuffer())
 
        st.success(f"Saved image: `{image_path}`")
        st.image(
            Image.open(image_path),
            caption=safe_name,
            use_container_width=True
        )
 
        if st.session_state.vision_engine is None:
            st.warning(
                "Image upload is ready, but AI image analysis needs the model file "
                f"`{MODEL_PATH}`. Add or train that model to enable predictions."
            )
        else:
            if st.button(
                "Analyze Image",
                use_container_width=True,
                type="primary"
            ):
                gradcam_path = Path(GRADCAM_FOLDER) / f"{image_path.stem}_gradcam.png"
 
                with st.spinner("Analyzing image..."):
                    result = st.session_state.vision_engine.analyze(
                        str(image_path),
                        save_gradcam_path=str(gradcam_path)
                    )
 
                if not result.get("success"):
                    st.error(result.get("error", "Unable to analyze image."))
                else:
                    prediction = result.get("predicted_class", "Unknown")
                    confidence = result.get("confidence", 0.0)
 
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Prediction", prediction)
                    with col2:
                        st.metric("Confidence", f"{confidence * 100:.1f}%")
 
                    if gradcam_path.exists():
                        st.image(
                            str(gradcam_path),
                            caption="Grad-CAM",
                            use_container_width=True
                        )
 
                    st.session_state.prediction_history.append(
                        {
                            "filename": safe_name,
                            "prediction": prediction,
                            "confidence": round(confidence * 100, 2),
                            "saved_path": str(image_path),
                        }
                    )
 
# ============================================================
# PAGE 4: MEDICAL REPORT ANALYZER
# ============================================================
 
elif page == "📄 Medical Report Analyzer":
 
    st.title("📄 AI Medical Report Analyzer")
    st.markdown(
        """
        Upload a medical report in **PDF format** and let MedAssist AI
        extract, summarize, and explain the findings in simple language.
        """
    )
 
    # --------------------------------------------------------
    # Initialize Session State
    # --------------------------------------------------------
    if "report_history" not in st.session_state:
        st.session_state.report_history = []
 
    if "uploaded_report_name" not in st.session_state:
        st.session_state.uploaded_report_name = None
 
    if "report_summary" not in st.session_state:
        st.session_state.report_summary = None
 
    if "report_text" not in st.session_state:
        st.session_state.report_text = None
 
    if "report_analyzed" not in st.session_state:
        st.session_state.report_analyzed = False
 
    # --------------------------------------------------------
    # Upload PDF
    # --------------------------------------------------------
    uploaded_pdf = st.file_uploader(
        "Upload Medical Report",
        type=["pdf"],
        accept_multiple_files=False
    )
 
    if uploaded_pdf is not None:
 
        st.session_state.uploaded_report_name = uploaded_pdf.name
        safe_pdf_name = Path(uploaded_pdf.name).name
        report_path = Path(UPLOAD_REPORT_FOLDER) / safe_pdf_name
        report_path.write_bytes(uploaded_pdf.getbuffer())
        uploaded_pdf.seek(0)
 
        col1, col2 = st.columns([3, 1])
 
        with col1:
            st.success(f"Uploaded: **{uploaded_pdf.name}**")
            st.caption(f"Saved to `{report_path}`")
 
        with col2:
            st.metric(
                "Size",
                f"{uploaded_pdf.size / 1024:.1f} KB"
            )
 
        analyze_btn = st.button(
            "🔍 Analyze Report",
            use_container_width=True,
            type="primary"
        )
 
        if analyze_btn:
 
            try:
 
                with st.spinner("Reading PDF..."):
 
                    report_generator = ReportGenerator()
 
                    extracted_text = report_generator.extract_text(
                        uploaded_pdf
                    )
 
                    if extracted_text is None:
                        extracted_text = ""
 
                    extracted_text = extracted_text.strip()
 
                    if len(extracted_text) == 0:
                        st.error(
                            "No readable text found inside the uploaded PDF."
                        )
                        st.stop()
 
                    st.session_state.report_text = extracted_text
 
                with st.spinner("Analyzing medical report..."):
 
                    summary = report_generator.generate_summary(
                        extracted_text
                    )
 
                    if summary is None:
                        summary = (
                            "Summary could not be generated."
                        )
 
                    st.session_state.report_summary = summary
                    st.session_state.report_analyzed = True
 
                    st.session_state.report_history.append(
                        {
                            "filename": uploaded_pdf.name,
                            "saved_path": str(report_path),
                            "summary": summary
                        }
                    )
 
            except Exception as e:
 
                st.error(
                    f"Unable to analyze report.\n\n{e}"
                )
 
    # --------------------------------------------------------
    # Display Analysis
    # --------------------------------------------------------
    if st.session_state.report_analyzed:
 
        st.divider()
 
        st.subheader("📋 AI Summary")
 
        st.markdown(
            st.session_state.report_summary
        )
 
        with st.expander("📄 Extracted Report Text"):
 
            st.text_area(
                "",
                value=st.session_state.report_text,
                height=350,
                disabled=True
            )
 
        if st.button(
            "🗑️ Clear Analysis",
            use_container_width=True
        ):
 
            st.session_state.report_text = None
            st.session_state.report_summary = None
            st.session_state.report_analyzed = False
            st.session_state.uploaded_report_name = None
 
            st.rerun()
 
    # --------------------------------------------------------
    # Previous Reports
    # --------------------------------------------------------
    if len(st.session_state.report_history) > 0:
 
        st.divider()
 
        st.subheader("📚 Analysis History")
 
        for idx, report in enumerate(
            reversed(st.session_state.report_history),
            start=1
        ):
 
            with st.expander(
                f"{idx}. {report['filename']}"
            ):
 
                st.markdown(report["summary"])
 
# ============================================================
# PAGE 4: HEALTH DASHBOARD
# ============================================================
 
elif page == "📊 Health Dashboard":
 
    st.title("📊 MedAssist Health Dashboard")
    st.markdown(
        "Monitor your MedAssist usage statistics and recent activities."
    )
 
    # --------------------------------------------------------
    # Initialize Dashboard Session Variables
    # --------------------------------------------------------
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
 
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
 
    if "report_history" not in st.session_state:
        st.session_state.report_history = []
 
    total_predictions = len(st.session_state.prediction_history)
    total_messages = len(st.session_state.chat_history)
    total_reports = len(st.session_state.report_history)
 
    # --------------------------------------------------------
    # Overview Metrics
    # --------------------------------------------------------
    st.subheader("📈 Overview")
 
    col1, col2, col3 = st.columns(3)
 
    with col1:
        st.metric(
            "Disease Predictions",
            total_predictions
        )
 
    with col2:
        st.metric(
            "Chat Messages",
            total_messages
        )
 
    with col3:
        st.metric(
            "Reports Analyzed",
            total_reports
        )
 
    st.divider()
 
    # --------------------------------------------------------
    # Prediction History
    # --------------------------------------------------------
    st.subheader("🩺 Recent Disease Predictions")
 
    if total_predictions == 0:
 
        st.info(
            "No disease predictions have been made yet."
        )
 
    else:
 
        prediction_df = pd.DataFrame(
            st.session_state.prediction_history
        )
 
        st.dataframe(
            prediction_df,
            use_container_width=True,
            hide_index=True
        )
 
        csv_predictions = prediction_df.to_csv(
            index=False
        ).encode("utf-8")
 
        st.download_button(
            label="⬇ Download Prediction History",
            data=csv_predictions,
            file_name="prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )
 
    st.divider()
 
    # --------------------------------------------------------
    # Chat Statistics
    # --------------------------------------------------------
    st.subheader("💬 AI Chat Activity")
 
    user_messages = len(
        [
            x for x in st.session_state.chat_history
            if x["role"] == "user"
        ]
    )
 
    assistant_messages = len(
        [
            x for x in st.session_state.chat_history
            if x["role"] == "assistant"
        ]
    )
 
    col1, col2 = st.columns(2)
 
    with col1:
        st.metric(
            "User Messages",
            user_messages
        )
 
    with col2:
        st.metric(
            "Assistant Replies",
            assistant_messages
        )
 
    if total_messages:
 
        with st.expander("View Conversation"):
 
            for chat in st.session_state.chat_history:
 
                if chat["role"] == "user":
                    st.markdown(
                        f"**👤 You:** {chat['content']}"
                    )
                else:
                    st.markdown(
                        f"**🤖 AI:** {chat['content']}"
                    )
 
    st.divider()
 
    # --------------------------------------------------------
    # Report History
    # --------------------------------------------------------
    st.subheader("📄 Medical Reports")
 
    if total_reports == 0:
 
        st.info(
            "No medical reports analyzed."
        )
 
    else:
 
        history_df = pd.DataFrame(
            st.session_state.report_history
        )
 
        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )
 
        csv_reports = history_df.to_csv(
            index=False
        ).encode("utf-8")
 
        st.download_button(
            "⬇ Download Report History",
            csv_reports,
            "medical_report_history.csv",
            "text/csv",
            use_container_width=True
        )
 
    st.divider()
 
    # --------------------------------------------------------
    # Clear Application History
    # --------------------------------------------------------
    st.subheader("🗑 Reset Application")
 
    if st.button(
        "Clear All History",
        type="secondary",
        use_container_width=True
    ):
 
        st.session_state.prediction_history = []
        st.session_state.chat_history = []
        st.session_state.report_history = []
 
        st.success(
            "Application history has been cleared."
        )
 
        st.rerun()
 
# ============================================================
# PAGE 5: VOICE ASSISTANT
# ============================================================

elif page == "🎤 Voice Assistant":

    st.title("🎤 MedAssist AI Voice Assistant")

    st.markdown("""
Speak naturally with MedAssist AI.

Features:
- 🎙 Voice Input
- 🤖 AI Medical Chat
- 🔊 Voice Output
- 💬 Conversation History
""")

    # -----------------------------------------
    # Initialize Session State
    # -----------------------------------------

    if "voice_history" not in st.session_state:
        st.session_state.voice_history = []

    speech_engine = st.session_state.speech_engine

    chat_engine = load_chat_engine()

    col1, col2 = st.columns([3, 1])

    with col1:

        st.success("Microphone Ready")

    with col2:

        if st.button("🗑 Clear Voice Chat"):

            st.session_state.voice_history = []

            st.rerun()

    st.divider()

    # -----------------------------------------
    # Conversation History
    # -----------------------------------------

    for chat in st.session_state.voice_history:

        with st.chat_message(chat["role"]):

            st.markdown(chat["content"])

    st.divider()

    # -----------------------------------------
    # Record Button
    # -----------------------------------------

    if st.button("🎙 Start Recording", use_container_width=True):

        try:

            with st.spinner("Listening..."):

                user_text = speech_engine.speech_to_text()

            if user_text is None or len(user_text.strip()) == 0:

                st.warning("No speech detected.")

            else:

                st.session_state.voice_history.append(
                    {
                        "role": "user",
                        "content": user_text
                    }
                )

                with st.chat_message("user"):

                    st.markdown(user_text)

                with st.spinner("Thinking..."):

                    ai_response = chat_engine.generate_response(
                        user_text
                    )

                st.session_state.voice_history.append(
                    {
                        "role": "assistant",
                        "content": ai_response
                    }
                )

                with st.chat_message("assistant"):

                    st.markdown(ai_response)

                with st.spinner("Speaking..."):

                    speech_engine.text_to_speech(ai_response)

        except Exception as e:

            st.error(f"Voice Assistant Error\n\n{e}")

    st.divider()

    st.info(
        """
Tips

• Speak clearly

• Ask one question at a time

• Internet connection required for Gemini

• Audio output uses your default speaker
        """
    )