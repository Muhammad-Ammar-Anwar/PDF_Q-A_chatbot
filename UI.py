import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize embedding and LLM only once
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model_name="Llama3-8b-8192", groq_api_key=groq_api_key)

# Define system prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Act as a Q/A chatbot. "
    "Answer concisely and with enough detail.\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

def process_pdf(pdf_file):
    """Loads and processes the uploaded PDF into a retriever-ready vector store."""
    loader = PyPDFLoader(pdf_file.name)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
    split_docs = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever()

    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    return rag_chain

# Store chain globally after PDF upload
rag_chain = None

# Upload handler
def handle_upload(pdf_file):
    global rag_chain
    rag_chain = process_pdf(pdf_file)
    return "✅ PDF processed! You can now ask your questions."

# Q&A handler
def ask_question(user_input):
    if not rag_chain:
        return "⚠️ Please upload a PDF first."
    response = rag_chain.invoke({"input": user_input})
    return response["answer"]

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(), title="PDF Q & A ChatBot") as demo:
    gr.Markdown("## 🤖 PDF ChatBot")
    gr.Markdown("Upload your research PDF and ask questions about it.")

    with gr.Row():
        with gr.Column(scale=1):
            pdf_upload = gr.File(label="📄 Upload PDF", file_types=[".pdf"])
            upload_status = gr.Textbox(label="Status", interactive=False)
            pdf_upload.change(fn=handle_upload, inputs=pdf_upload, outputs=upload_status)
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat", show_label=True)
            question_input = gr.Textbox(placeholder="Ask a question...", label="Your Question")
            send_button = gr.Button("Ask")

            def respond(user_input, chat_history):
                answer = ask_question(user_input)
                chat_history.append((user_input, answer))
                return chat_history, ""

            send_button.click(respond, inputs=[question_input, chatbot], outputs=[chatbot, question_input])
            question_input.submit(respond, inputs=[question_input, chatbot], outputs=[chatbot, question_input])

    gr.Markdown("— Powered by LangChain, FAISS, HuggingFace & Groq")

# Launch with shareable link
demo.launch(share=True)
