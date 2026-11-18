
import validators
import streamlit as st

from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi

from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredURLLoader


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Content Summarizer From YouTube or Website",
    page_icon="🦜",
    layout="wide"
)


# --------------------------------------------------
# YOUTUBE TRANSCRIPT FUNCTION
# --------------------------------------------------

def get_youtube_transcript(url):

    parsed_url = urlparse(url)
    hostname = (parsed_url.hostname or "").lower()

    if hostname in ["youtu.be", "www.youtu.be"]:
        video_id = parsed_url.path.strip("/").split("/")[0]

    elif hostname in [
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com"
    ]:
        if parsed_url.path == "/watch":
            video_id = parse_qs(parsed_url.query).get("v", [None])[0]

        elif parsed_url.path.startswith(("/shorts/", "/embed/", "/live/")):
            video_id = parsed_url.path.split("/")[2]

        else:
            video_id = None

    else:
        raise ValueError("Invalid YouTube URL.")

    if not video_id:
        raise ValueError("Could not extract YouTube video ID.")

    # Fetch transcript directly
    transcript = YouTubeTranscriptApi().fetch(
        video_id,
        languages=["en"]
    )

    transcript_text = " ".join(
        snippet.text for snippet in transcript
    )

    if not transcript_text.strip():
        raise ValueError("The video transcript is empty.")

    return [
        Document(
            page_content=transcript_text,
            metadata={"source": url}
        )
    ]


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0b1120;
    color: #f8fafc;
}

.block-container {
    max-width: 1000px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}

.main-title {
    font-size: 40px;
    font-weight: 800;
    text-align: center;
    color: #ffffff !important;
    margin-bottom: 8px;
}

.subtitle {
    text-align: center;
    color: #a5b4fc !important;
    font-size: 16px;
    margin-bottom: 35px;
}

.search-label {
    color: #ffffff !important;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 12px;
}

div[data-testid="stTextInput"] input {
    background-color: #172033 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: 1px solid #64748b !important;
    border-radius: 12px !important;
    min-height: 55px !important;
    font-size: 16px !important;
    caret-color: #60a5fa !important;
}

div[data-testid="stTextInput"] input::placeholder {
    color: #cbd5e1 !important;
    -webkit-text-fill-color: #cbd5e1 !important;
    opacity: 1 !important;
}

div[data-testid="stTextInput"] input:focus {
    border: 2px solid #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.2) !important;
}

div.stButton > button {
    width: 100%;
    min-height: 52px;
    background: #2563eb !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px;
    font-size: 16px;
    font-weight: 700;
}

div.stButton > button:hover {
    background: #1d4ed8 !important;
    color: #ffffff !important;
}

section[data-testid="stSidebar"] {
    background-color: #111827 !important;
}

section[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}

.model-badge {
    background: #1e3a8a;
    border: 1px solid #3b82f6;
    color: #ffffff !important;
    padding: 12px;
    border-radius: 10px;
    font-weight: 700;
    font-size: 15px;
    text-align: center;
}

div[data-testid="stCaptionContainer"] {
    color: #cbd5e1 !important;
}

#MainMenu, footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🦜 AI Content Summarizer From YouTube or Website</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Summarize YouTube videos and website articles instantly using AI'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ Settings")

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="Enter your Groq API key"
    )

    st.divider()

    st.markdown("### 🤖 AI Model")

    st.markdown(
        '<div class="model-badge">OpenAI GPT-OSS 20B</div>',
        unsafe_allow_html=True
    )

    st.caption("Powered by Groq API and LangChain")


# --------------------------------------------------
# URL INPUT
# --------------------------------------------------

st.markdown(
    '<div class="search-label">'
    '🔗 Enter YouTube or Website URL'
    '</div>',
    unsafe_allow_html=True
)

generic_url = st.text_input(
    "URL",
    placeholder="Paste your YouTube or website link here...",
    label_visibility="collapsed"
)

st.caption(
    "Supports YouTube videos with accessible transcripts "
    "and publicly accessible website articles."
)


# --------------------------------------------------
# PROMPT
# --------------------------------------------------

prompt_template = """
You are an expert AI summarization assistant.

Summarize the following content in approximately 300 words.

Include:
1. Main topic
2. Important points
3. Key information
4. Main conclusions

Avoid unnecessary repetition.
Do not invent facts that are not present in the content.

Content:
{text}

Summary:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["text"]
)


# --------------------------------------------------
# SUMMARIZATION
# --------------------------------------------------

if st.button("✨ Summarize Content", use_container_width=True):

    if not groq_api_key.strip():
        st.error("Please enter your Groq API key.")

    elif not generic_url.strip():
        st.error("Please enter a YouTube or website URL.")

    elif not validators.url(generic_url.strip()):
        st.error("Please enter a valid URL.")

    else:

        try:

            with st.spinner(
                "🔍 Fetching content and generating summary..."
            ):

                url = generic_url.strip()

                # Fetch YouTube transcript or website content

                if "youtube.com" in url or "youtu.be" in url:

                    st.info("Fetching YouTube transcript...")

                    docs = get_youtube_transcript(url)

                else:

                    loader = UnstructuredURLLoader(
                        urls=[url],
                        ssl_verify=True,
                        headers={
                            "User-Agent": "Mozilla/5.0"
                        }
                    )

                    docs = loader.load()

                if not docs:
                    raise ValueError(
                        "No content could be extracted from this URL."
                    )

                # Initialize Groq LLM

                llm = ChatGroq(
                    model="openai/gpt-oss-20b",
                    groq_api_key=groq_api_key,
                    temperature=0
                )

                # Summarization chain

                chain = load_summarize_chain(
                    llm,
                    chain_type="stuff",
                    prompt=prompt
                )

                result = chain.invoke({
                    "input_documents": docs
                })

                output_summary = result["output_text"]

            st.success("Summary generated successfully!")

            st.markdown("### 📝 Generated Summary")

            st.markdown(output_summary)

        except Exception as e:

            error = str(e)

            if "transcript" in error.lower():
                st.error(
                    "YouTube transcript unavailable. "
                    "Check whether captions are enabled."
                )

            elif "429" in error:
                st.error(
                    "API rate limit reached. Please try again later."
                )

            elif "401" in error or "403" in error:
                st.error(
                    "Authentication or access error. "
                    "Check your API key and video permissions."
                )

            else:
                st.error(f"Unable to summarize content: {error}")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <p style="text-align:center;color:#94a3b8;">
        Built with ❤️ using LangChain, Groq & Streamlit
    </p>
    """,
    unsafe_allow_html=True
)
