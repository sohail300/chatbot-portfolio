from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
import json
import asyncio

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.environ.get("AI_API_KEY"))

model = genai.GenerativeModel('gemini-2.5-flash')

app = FastAPI(
    title="Portfolio Chatbot API",
    description="An intelligent chatbot that answers questions about Sohail's portfolio using RAG",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    msg: str

# Environment variables
pinecone_api_key = os.environ.get('PINECONE_API_KEY')
pinecone_environment = os.environ.get('PINECONE_ENVIRONMENT')

# Initialize Pinecone client and index at startup
print(f"Initializing Pinecone with API key: {bool(pinecone_api_key)}")
try:
    pc = Pinecone(api_key=pinecone_api_key)
    print(f"Pinecone client created: {pc}")
    
    # Check if index exists, if not create it
    index_name = "portfolio"
    if index_name not in pc.list_indexes().names():
        print(f"Index {index_name} not found. Creating new index...")
        pc.create_index(
            name=index_name,
            dimension=768,  # all-mpnet-base-v2 embedding dimension
            metric="cosine"
        )
        print(f"Created new index: {index_name}")
    
    index = pc.Index(index_name)
    print(f"Connected to Pinecone index: {index_name}, index: {index}")
except Exception as e:
    print(f"Error connecting to Pinecone: {e}")
    import traceback
    traceback.print_exc()
    pc = None
    index = None

# Encoder
encoder = SentenceTransformer("all-mpnet-base-v2")

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <body>
            <h1>Healthy Server!</h1>
        </body>
    </html>
    """

@app.post("/api/chat", response_model=ChatResponse)
async def chat_handler(request: ChatRequest):
    try:
        # Get the latest message content
        latest_message = request.messages[-1].content
        print(f"Latest message: {latest_message}")

        # Create embedding
        search_vector = encoder.encode(latest_message).tolist()
        print('Vector created for search')

        # Search in Pinecone
        if index is None:
            print("Index is None, returning fallback response")
            return ChatResponse(msg="I'm sorry, I'm having trouble connecting to my knowledge base right now. Please try again later.")
        
        try:
            results = index.query(
                vector=search_vector,
                top_k=5,
                include_metadata=True
            )
            print("Vector search results:")
            print(f"Results: {results}")
            
            if results and 'matches' in results and results['matches']:
                doc_context = (
                    "\nSTART CONTEXT\n"
                    + "\n".join(match['metadata']['title'] + ": " + match['metadata']['description'] for match in results['matches'])
                    + "\nEND CONTEXT"
                )
            else:
                doc_context = "\nSTART CONTEXT\nNo relevant information found.\nEND CONTEXT"
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            return ChatResponse(msg="I'm sorry, I'm having trouble searching my knowledge base right now. Please try again later.")

        print('Document context:')
        print(doc_context)

        # Create the RAG prompt
        rag_prompt = (
            "You are an AI assistant answering questions as Sohail in his Portfolio Web App. Talk like a human. "
            "Format responses using markdown where applicable.\n"
            "Just give the required answers and to the point. If the answer is not provided in the context, the AI assistant will say, "
            "\"I'm sorry, I do not know the answer\". "
            "The context is provided below\n"
            f"{doc_context}\n"
            "The Question is below\n"
            f"{latest_message}\n"
        )

        print('RAG prompt created')

        try:
            response = model.generate_content(rag_prompt)
            result = response.text
            print(f"Generated response: {result}")
            return ChatResponse(msg=result)
        except Exception as e:
            print(f"Error generating response with Gemini: {e}")
            return ChatResponse(msg="I'm sorry, I'm having trouble generating a response right now. Please try again later.")

    except Exception as e:
        print(f"Error in chat handler: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/stream")
async def chat_stream_handler(request: ChatRequest):
    async def generate_stream():
        try:
            # Get the latest message content
            latest_message = request.messages[-1].content
            print(f"Latest message: {latest_message}")

            # Create embedding
            search_vector = encoder.encode(latest_message).tolist()
            print('Vector created for search')

            # Search in Pinecone
            if index is None:
                print("Index is None, returning fallback response")
                yield f"data: {json.dumps({'content': 'I\'m sorry, I\'m having trouble connecting to my knowledge base right now. Please try again later.'})}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            try:
                results = index.query(
                    vector=search_vector,
                    top_k=5,
                    include_metadata=True
                )
                print("Vector search results:")
                print(f"Results: {results}")
                
                if results and 'matches' in results and results['matches']:
                    doc_context = (
                        "\nSTART CONTEXT\n"
                        + "\n".join(match['metadata']['title'] + ": " + match['metadata']['description'] for match in results['matches'])
                        + "\nEND CONTEXT"
                    )
                else:
                    doc_context = "\nSTART CONTEXT\nNo relevant information found.\nEND CONTEXT"
            except Exception as e:
                print(f"Error querying Pinecone: {e}")
                yield f"data: {json.dumps({'content': 'I\'m sorry, I\'m having trouble searching my knowledge base right now. Please try again later.'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            print('Document context:')
            print(doc_context)

            # Create the RAG prompt
            rag_prompt = (
                "You are an AI assistant answering questions as Sohail in his Portfolio Web App. Talk like a human. "
                "Format responses using markdown where applicable.\n"
                "Just give the required answers and to the point. If the answer is not provided in the context, the AI assistant will say, "
                "\"I'm sorry, I do not know the answer\". "
                "The context is provided below\n"
                f"{doc_context}\n"
                "The Question is below\n"
                f"{latest_message}\n"
            )

            print('RAG prompt created')

            try:
                response = model.generate_content(rag_prompt)
                result = response.text
                print(f"Generated response: {result}")
                
                # Stream the response word by word
                words = result.split(' ')
                for i, word in enumerate(words):
                    chunk = word + (' ' if i < len(words) - 1 else '')
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                    await asyncio.sleep(0.03)  # 30ms delay between words
                
                yield "data: [DONE]\n\n"
            except Exception as e:
                print(f"Error generating response with Gemini: {e}")
                yield f"data: {json.dumps({'content': 'I\'m sorry, I\'m having trouble generating a response right now. Please try again later.'})}\n\n"
                yield "data: [DONE]\n\n"

        except Exception as e:
            print(f"Error in chat stream handler: {str(e)}")
            yield f"data: {json.dumps({'content': 'Sorry, something went wrong. Please try again.'})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
