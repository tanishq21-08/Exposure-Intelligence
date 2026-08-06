from fastapi import FastAPI
from pydantic import BaseModel
from ingestion import sheet_to_text
from confidence import extract_with_consistency

app = FastAPI(title="Exposure Intelligence API")

# ---- request/response shapes ----
class ExtractRequest(BaseModel):
    text: str                       # the raw Statement-of-Values text
    n: int = 5                      # samples for the confidence layer (optional, defaults to 5)
    use_cache: bool = True          # replay cached result if available

# ---- endpoints ----
@app.get("/health")
def health():
    """Simple check that the service is alive."""
    return {"status": "ok"}

@app.post("/extract")
def extract_endpoint(req: ExtractRequest):
    """Take raw SOV text, return structured exposure data with confidence."""
    result = extract_with_consistency(req.text, n=req.n, use_cache=req.use_cache)
    return {"locations": result}