import streamlit as st
from pathlib import Path
from datetime import datetime

from src.rag.pipeline import RAGPipeline
from src.nlp.analyzer import NLPAnalyzer
from src.report.generator import ReportGenerator
from src.utils.file_handler import FileHandler



st.set_page_config(
    page_title="BriefBot",
    page_icon="🧠",
    layout="wide"
)



css_path = Path("assets/style.css")
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



def init_state():
    defaults = {
        "rag_pipeline": None,
        "chat_history": [],
        "documents_loaded": False,
        "nlp_results": None,
        "uploaded_filenames": []
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()



def sidebar():
    with st.sidebar:

        # Logo
        if Path("assets/logo.png").exists():
            st.image("assets/logo.png", width=60)
        else:
            st.markdown("### 🧠")

        st.title("BriefBot")
        st.caption("Upload → Analyze → Query → Report")

        st.divider()

        # API + Model
        st.subheader("⚙️ Settings")

        groq_key = st.text_input("Groq API Key", type="password")

        model = st.selectbox(
            "Model",
            ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
        )

        st.divider()

        # Upload
        st.subheader("📁 Upload")
        files = st.file_uploader(
            "Upload files",
            type=["pdf", "txt", "csv", "md", "json"],
            accept_multiple_files=True
        )

        process_clicked = st.button("🚀 Process", use_container_width=True)

        return groq_key, model, files, process_clicked



def process_documents(files, key, model):
    try:
        handler = FileHandler()
        texts = handler.process_files(files)

        pipeline = RAGPipeline(groq_key=key, model=model)
        pipeline.build_index(texts)

        analyzer = NLPAnalyzer()
        full_text = " ".join([t["content"] for t in texts])
        nlp = analyzer.analyze(full_text)

        st.session_state.rag_pipeline = pipeline
        st.session_state.nlp_results = nlp
        st.session_state.documents_loaded = True
        st.session_state.chat_history = []
        st.session_state.uploaded_filenames = [f.name for f in files]

        st.success(f"{len(files)} files processed!")

    except Exception as e:
        st.error(f"Error: {str(e)}")



def chat_ui():
    st.subheader("💬 Ask Questions")

    # show history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask something...")

    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            try:
                result = st.session_state.rag_pipeline.query(prompt)

                st.markdown(result["answer"])

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["answer"]
                })

            except Exception as e:
                st.error(str(e))



def nlp_ui():
    st.subheader("📊 NLP Insights")

    res = st.session_state.nlp_results
    if not res:
        st.warning("No data")
        return

    col1, col2 = st.columns(2)

    col1.metric("Words", res["word_count"])
    col2.metric("Unique", res["unique_terms"])

    st.divider()

    st.subheader("Keywords")
    for k, s in res["keywords"][:10]:
        st.write(f"{k} ({s:.2f})")

    st.subheader("Summary")
    st.info(res["summary"])



def report_ui():
    st.subheader("📝 Generate Report")

    name = st.text_input("Project Name")
    author = st.text_input("Author")

    if st.button("Generate"):
        try:
            gen = ReportGenerator(
                rag_pipeline=st.session_state.rag_pipeline,
                nlp_results=st.session_state.nlp_results
            )

            report = gen.generate(
                project_name=name,
                author=author,
                date=str(datetime.today()),
                filenames=st.session_state.uploaded_filenames
            )

            st.download_button("Download", report["markdown"])

        except Exception as e:
            st.error(str(e))



def main():
    key, model, files, process = sidebar()

    if process:
        if not key:
            st.warning("Enter API key")
        elif not files:
            st.warning("Upload files")
        else:
            with st.spinner("Processing..."):
                process_documents(files, key, model)

    if not st.session_state.documents_loaded:
        st.title("🧠 AI Project Intelligence System")
        st.info("Upload documents to begin")
    else:
        tab1, tab2, tab3 = st.tabs(["Chat", "NLP", "Report"])

        with tab1:
            chat_ui()
        with tab2:
            nlp_ui()
        with tab3:
            report_ui()


if __name__ == "__main__":
    main()