from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import uuid
import uvicorn

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

# ── Environment ──────────────────────────────────────────────────────────────
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY not found in environment / .env file")

UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Shared heavy objects (loaded once at startup) ────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=groq_api_key)

system_prompt = (
    "You are an assistant for question-answering tasks about a PDF document. "
    "You are given relevant excerpts from the document as context. "
    "Use ONLY the provided context to answer the question. "
    "Do not say the PDF is incomplete or that you only have a snippet — "
    "you have been given the most relevant parts of the document for this question. "
    "If the answer is not found in the context, say 'I could not find that information in the document.'\n\n"
    "{context}"
)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# ── In-memory session store  { session_id -> rag_chain } ────────────────────
sessions: dict = {}

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="PDF Q&A API",
    description="Upload a PDF and ask questions about it using RAG + Groq LLM.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response schemas ────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    session_id: str
    question: str

class UploadResponse(BaseModel):
    session_id: str
    message: str
    filename: str
    pages: int

class AnswerResponse(BaseModel):
    session_id: str
    question: str
    answer: str

class SessionsResponse(BaseModel):
    active_sessions: list[str]
    count: int

# ── Helper ────────────────────────────────────────────────────────────────────
def build_rag_chain(pdf_path: str):
    """Load a PDF, chunk it, embed into FAISS, return a RAG chain."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
    split_docs = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever()

    qa_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    return rag_chain, len(docs)   # also return page count for the response

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "PDF Q&A API is running 🚀"}


@app.post("/upload", response_model=UploadResponse, tags=["PDF"])
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file.
    Returns a **session_id** that must be passed with every /ask request.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Save the file with a unique name to avoid collisions
    session_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{session_id}.pdf")

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        rag_chain, page_count = build_rag_chain(save_path)
    except Exception as e:
        os.remove(save_path)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

    sessions[session_id] = rag_chain

    return UploadResponse(
        session_id=session_id,
        message="PDF processed successfully. You can now ask questions.",
        filename=file.filename,
        pages=page_count,
    )


@app.post("/ask", response_model=AnswerResponse, tags=["Q&A"])
async def ask_question(body: QuestionRequest):
    """
    Ask a question about a previously uploaded PDF.
    Requires the **session_id** returned by /upload.
    """
    rag_chain = sessions.get(body.session_id)
    if not rag_chain:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please upload a PDF first.",
        )

    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        response = rag_chain.invoke({"input": body.question})
        answer = response["answer"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    return AnswerResponse(
        session_id=body.session_id,
        question=body.question,
        answer=answer,
    )


@app.delete("/session/{session_id}", tags=["Session"])
def delete_session(session_id: str):
    """
    Delete a session and its associated PDF to free memory.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    del sessions[session_id]

    pdf_path = os.path.join(UPLOAD_DIR, f"{session_id}.pdf")
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    return {"message": f"Session {session_id} deleted successfully."}


@app.get("/sessions", response_model=SessionsResponse, tags=["Session"])
def list_sessions():
    """List all active session IDs."""
    return SessionsResponse(
        active_sessions=list(sessions.keys()),
        count=len(sessions),
    )

if __name__ == "__main__":
    
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)