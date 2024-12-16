from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from dotenv import load_dotenv
import PyPDF2
from io import StringIO
import csv
import pandas as pd
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import datetime
import uuid

# Load environment variables and initialize FastAPI
load_dotenv()
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
model = SentenceTransformer('all-MiniLM-L6-v2')
genai.configure(api_key="Here is your API KEY ") #----------------------------------------------attention ------------------------------------------

def pdf_to_csv_in_memory(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    all_text = []
    
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            lines = text.split("\n")
            all_text.append(lines)
    
    csv_output = StringIO()
    csv_writer = csv.writer(csv_output)
    for page_text in all_text:
        for line in page_text:
            csv_writer.writerow([line])
    
    csv_output.seek(0)
    df = pd.read_csv(csv_output)
    return df

@app.get("/")
async def serve_homepage():
    return FileResponse("templates/index.html")

@app.post("/process")
async def process_pdfs(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    
    try:
        # Process the first PDF file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(await files[0].read())
            tmp_file.seek(0)
            
            # Convert PDF to CSV in memory
            df = pdf_to_csv_in_memory(tmp_file)
            
            # Detect text column
            text_column = None
            for col in df.columns:
                if df[col].dtype == object:
                    text_column = col
                    break
            
            if text_column is None:
                raise ValueError("No text column found in the CSV data.")
            
            # Combine text and create documents
            response = " ".join(df[text_column].dropna().tolist())
            text_splitter = SemanticChunker(
                HuggingFaceEmbeddings(),
                breakpoint_threshold_type="percentile"
            )
            documents = text_splitter.create_documents([response])
            
            # Create vector store
            vectorstore = Chroma.from_documents(documents=documents, embedding=embedding_model)
            
            # Create retriever
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})
            
            # Store in app state
            app.state.retriever = retriever
            
            return {"status": "success", "message": "PDF processed successfully"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_documents(body: dict = Body(...)):
    if not hasattr(app.state, "retriever"):
        raise HTTPException(status_code=400, detail="No documents have been processed yet.")
    
    question = body.get("question")
    if not question:
        raise HTTPException(status_code=422, detail="Question field is required")
    
    try:
        # Get relevant documents
        relevant_chunks = app.state.retriever.get_relevant_documents(question)
        
        # Check if question needs tabular response
        table_keywords = ['table', 'data', 'numbers', 'statistics', 'figures', 'comparison', 'metrics']
        needs_table = any(keyword in question.lower() for keyword in table_keywords)
        
        # Define base prompt structure
        if needs_table:
            prompt_structure = {
                "query": "the original question",
                "response": {
                    "summary": "Clear and concise summary",
                    "key_points": {
                        "name of key point1": "First key finding",
                        "name of key point2": "Second key finding"
                    },
                    "tables": [{
                        "title": "Relevant table name",
                        "headers": ["Column1", "Column2"],
                        "rows": [["Value1", "Value2"]]
                    }]
                }
            }
        else:
            prompt_structure = {
                "query": "the original question",
                "response": {
                    "summary": "Clear and concise summary",
                    "key_points": {
                        "name of key point1": "First key finding",
                        "name of key point2": "Second key finding"
                    }
                }
            }

        structured_prompt = f"""
        -YOU ARE HELP FUL ASSISTANT WHICH PERFORM MATHS WITH THE DATA ACCORDING TO THE QUESTION.
        Generate a response in this exact JSON structure:
        {prompt_structure}

        Context:
        {relevant_chunks}

        Question:
        {question}

        Provide only the JSON response with named key points.
        """
        
        # Generate response using Gemini
        gemini_model = genai.GenerativeModel("gemini-2.0-flash-exp")
        response = gemini_model.generate_content(structured_prompt)
        
        return JSONResponse(content={
            "messages": [{
                "role": "assistant",
                "content": response.text,
                "timestamp": str(datetime.datetime.now())
            }],
            "conversation_id": str(uuid.uuid4())
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
