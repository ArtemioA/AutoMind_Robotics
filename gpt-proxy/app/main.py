import os
import logging
from typing import Any, List, Optional, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import uvicorn

# Logs visibles en Cloud Run
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")

# Clave API
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY no está configurada en el entorno.")

# Modelo por defecto (puedes forzarlo vía variable MODEL)
MODEL_DEFAULT = os.getenv("MODEL", "gpt-4.1-mini")

client = OpenAI()

app = FastAPI(title="GPT Proxy", version="3.0")

# -------- Esquemas --------
class InferenceSimpleIn(BaseModel):
    text: str
    image_b64: Optional[str] = None
    mime: Optional[str] = None  # p.ej. "image/jpeg"
    model: Optional[str] = None # override opcional

class InferenceOut(BaseModel):
    model: str
    output: str
    branch: Optional[str] = None

class InferRawIn(BaseModel):
    # payload crudo para OpenAI responses.create
    # Debe contener "input": [...]
    input: List[Any]
    model: Optional[str] = None  # override opcional

# -------- Rutas --------
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_DEFAULT}

@app.post("/infer_simple", response_model=InferenceOut)
def infer_simple(payload: InferenceSimpleIn):
    """
    Ruta cómoda: acepta text + image_b64 y arma data:URL internamente.
    Útil para clientes simples; pero si hay dudas, usa /infer_raw.
    """
    try:
        model = payload.model or MODEL_DEFAULT

        if payload.image_b64:
            mime = payload.mime or "image/jpeg"
            data_url = f"data:{mime};base64,{payload.image_b64}"
            content = [
                {"type": "input_text", "text": payload.text},
                {"type": "input_image", "image_url": data_url},
            ]
            branch = "vision-b64"
            log.info(f"[VISION-B64] model={model}, mime={mime}, b64_len={len(payload.image_b64)}")
        else:
            content = [{"type": "input_text", "text": payload.text}]
            branch = "text"
            log.info(f"[TEXT] model={model}")

        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": content}],
        )
        return {"model": model, "output": resp.output_text, "branch": branch}

    except Exception:
        log.exception("Inference error (/infer_simple)")
        raise HTTPException(status_code=500, detail="Inference error")

@app.post("/infer_raw")
def infer_raw(payload: InferRawIn):
    """
    Ruta PASSTHROUGH: reenvía EXACTAMENTE lo que te funciona en Colab.
    Envía el mismo objeto 'input=[{...}]' que usas con responses.create.
    """
    try:
        model = payload.model or MODEL_DEFAULT
        # Log de tamaños para depurar
        try:
            # calcular tamaños aproximados del bloque de imagen si viene data:
            total_len = 0
            b64_len = 0
            for blk in payload.input:
                if isinstance(blk, dict) and "content" in blk:
                    for part in blk.get("content", []):
                        if isinstance(part, dict):
                            if part.get("type") in ("input_text", "text"):
                                txt = part.get("text") or ""
                                total_len += len(txt)
                            if part.get("type") in ("input_image", "image_url"):
                                url = part.get("image_url")
                                if isinstance(url, dict):
                                    url = url.get("url")
                                if isinstance(url, str) and url.startswith("data:"):
                                    # longitud aproximada del base64
                                    if ";base64," in url:
                                        b64 = url.split(";base64,")[-1]
                                        b64_len += len(b64)
            log.info(f"[RAW] model={model}, approx_text_len={total_len}, approx_b64_len={b64_len}")
        except Exception:
            pass

        resp = client.responses.create(
            model=model,
            input=payload.input,
        )
        return {"model": model, "output": resp.output_text}

    except Exception:
        log.exception("Inference error (/infer_raw)")
        raise HTTPException(status_code=500, detail="Inference error")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
