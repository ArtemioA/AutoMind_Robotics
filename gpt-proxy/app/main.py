import os 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import uvicorn

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY no está configurada en el entorno.")

MODEL = os.getenv("MODEL", "gpt-4.1-mini")  # Este modelo SÍ funciona con Responses API
client = OpenAI()

app = FastAPI(title="GPT Proxy", version="2.0")

class InferenceIn(BaseModel):
    text: str
    image_b64: str
    mime: str = "image/png"

class InferenceOut(BaseModel):
    model: str
    output: str

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}

@app.post("/infer", response_model=InferenceOut)
def infer(payload: InferenceIn):
    try:
        # Crear data URL igual que en tu código que funciona
        data_url = f"data:{payload.mime};base64,{payload.image_b64}"
        
        # Usar Responses API (Assistants API) en lugar de Chat Completions
        resp = client.responses.create(
            model=MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": payload.text},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
        )
        
        out = resp.output_text
        return {"model": MODEL, "output": out}
        
    except Exception as e:
        print(f"Error en inferencia: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
