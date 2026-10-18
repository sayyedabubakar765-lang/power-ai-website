import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import ollama

app = FastAPI(title="Abubakar Ai Fast Core")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "llama3.2:1b"

class ChatRequest(BaseModel):
    prompt: str

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>Error: index.html not found</h3>"

@app.post("/chat")
def handle_chat(request: ChatRequest):
    try:
        # Optimized token limits and thread configuration for lightning-fast local response
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": request.prompt}],
            options={
                "num_predict": 300,
                "temperature": 0.7,
                "num_thread": 4
            }
        )
        return {"response": response['message']['content']}
    except Exception as e:
        return {"response": f"System error: {str(e)}"}

@app.get("/download/{target}")
def download_target(target: str):
    return JSONResponse({
        "status": "success",
        "message": f"Binary target [{target}] build complete."
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
