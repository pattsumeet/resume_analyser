import os
from dotenv import load_dotenv
import gradio as gr
from resume_reader import extract_text
from rag_utils import create_vector_store, create_conversational_rag, answer_question

# Load .env file
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Caches
vectordb_cache = {}
conv_chain_cache = {}
chat_histories = {}

def analyze_resume(file, message, session_id, chat_history):
    if not file:
        return chat_history, ""  # Keep chat history as is

    # Save uploaded file
    uploaded_path = file.name
    file_name = os.path.basename(uploaded_path)
    file_path = os.path.join(UPLOAD_DIR, file_name)
    with open(uploaded_path, "rb") as src, open(file_path, "wb") as dst:
        dst.write(src.read())

    # Create or reuse vector store & conversational chain
    if file_name not in vectordb_cache:
        try:
            text = extract_text(file_path)
            vectordb = create_vector_store(text)
            conv_chain = create_conversational_rag(vectordb)
            vectordb_cache[file_name] = vectordb
            conv_chain_cache[file_name] = conv_chain
        except ValueError as e:
            chat_history.append([message, f"Error processing resume: {str(e)}"])
            return chat_history, ""
    else:
        conv_chain = conv_chain_cache[file_name]

    # Use passed chat_history (from Gradio) instead of session dict
    # Convert Gradio chat format [[user, bot], ...] to list of tuples
    llm_history = [(q, a) for q, a in chat_history]

    # Get answer
    answer = answer_question(conv_chain, message, llm_history)

    # Append new message to chat history
    chat_history.append([message, answer])

    # Return updated chat history and clear input box
    return chat_history, ""


# Gradio Chat UI
with gr.Blocks() as demo:
    gr.Markdown("## Resume Analyzer Chat Assistant")
    session_id = gr.Textbox(label="Session ID (unique per user)", placeholder="Enter a unique ID", value="default_session")
    file_input = gr.File(file_types=[".pdf", ".docx"], label="Upload Resume")
    chatbot = gr.Chatbot(label="Resume Analyzer")
    msg = gr.Textbox(label="Your Question", placeholder="Ask anything about your resume...")
    submit = gr.Button("Send")

    submit.click(
        fn=analyze_resume,
        inputs=[file_input, msg, session_id, chatbot],
        outputs=[chatbot, msg]
    )

if __name__ == "__main__":
    demo.launch()
