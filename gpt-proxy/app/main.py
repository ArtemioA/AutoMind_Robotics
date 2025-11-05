import os 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import uvicorn
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY no está configurada en el entorno.")

MODEL = os.getenv("MODEL", "gpt-4.1-mini")
client = OpenAI()

app = FastAPI(title="GPT Proxy Vision", version="3.0")

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

@app.post("/debug")
def debug(payload: InferenceIn):
    """Endpoint para debug"""
    return {
        "text_received": payload.text,
        "image_b64_length": len(payload.image_b64),
        "mime_received": payload.mime,
        "message": "Datos recibidos correctamente"
    }

@app.post("/infer", response_model=InferenceOut)
def infer(payload: InferenceIn):
    try:
        logger.info(f"Iniciando inferencia con modelo {MODEL}")
        logger.info(f"Texto: {payload.text[:100]}...")
        logger.info(f"Tamaño imagen base64: {len(payload.image_b64)}")
        logger.info(f"MIME: {payload.mime}")
        
        # Crear data URL
        data_url = f"data:{payload.mime};base64,{payload.image_b64}"
        
        logger.info("Llamando a OpenAI Responses API...")
        
        # Usar Responses API
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
        
        output_text = resp.output_text
        logger.info(f"Inferencia completada. Longitud respuesta: {len(output_text)}")
        
        return {"model": MODEL, "output": output_text}
        
    except Exception as e:
        logger.error(f"Error en inferencia: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
