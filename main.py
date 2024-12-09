from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import tempfile
import os
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.embeddings import HuggingFaceEmbeddings
from model import llm
import datetime
import uuid
import PyPDF2


# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = "./uploaded_files"
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize models and chains
embeddings = HuggingFaceEmbeddings()

# Setup prompts
contextualize_q_system_prompt = """Given a chat history and the latest user question 
which might reference context in the chat history, formulate a standalone question which can be understood 
without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""

qa_system_prompt = """You are an assistant for question-answering tasks. Use 
the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you 
don't know. Use three sentences maximum and keep the answer concise.\n\n{context}"""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

def extract_text_from_pdf(pdf_path):
    documents = []
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    page_text = "\n".join([line for line in page_text.splitlines() if line.strip()])
                    doc = Document(
                        page_content=page_text,
                        metadata={"source": pdf_path, "page": page_num + 1}
                    )
                    documents.append(doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing {pdf_path}: {str(e)}")
    return documents

def process_multiple_pdfs(files):
    all_documents = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.file.read())
            pdf_path = tmp_file.name
            original_filename = file.filename
        documents = extract_text_from_pdf(pdf_path)
        for doc in documents:
            doc.metadata["source"] = original_filename
        all_documents.extend(documents)
        os.unlink(pdf_path)
    return all_documents

@app.get("/")
async def serve_homepage():
    return FileResponse("templates/index.html")

@app.post("/process")
async def process_pdfs(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    all_documents = process_multiple_pdfs(files)

    # Create vector store
    db = Chroma.from_documents(
        all_documents,
        embeddings,
        persist_directory=persistent_directory
    )
    
    # Create retriever
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    # Create chains
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    # Store the chain in app state
    app.state.chain = rag_chain
    app.state.chat_history = []

    return {"status": "success", "documents_processed": len(all_documents)}

@app.post("/query")
async def query_documents(body: dict = Body(...)):
    if not hasattr(app.state, "chain"):
        raise HTTPException(status_code=400, detail="No documents have been processed yet.")

    question = body.get("question")
    if not question:
        raise HTTPException(status_code=422, detail="Question field is required")

    try:
        result = app.state.chain.invoke({
            "input": question,
            "chat_history": app.state.chat_history
        })
        
        # Update chat history
        app.state.chat_history.append(HumanMessage(content=question))
        app.state.chat_history.append(SystemMessage(content=result["answer"]))

        return JSONResponse(content={
            "messages": [{
                "role": "assistant",
                "content": result["answer"],
                "timestamp": str(datetime.datetime.now())
            }],
            "conversation_id": str(uuid.uuid4())
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")
