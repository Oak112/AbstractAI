"""Context Compiler - Main FastAPI Application"""

import os
import re
import json
import zipfile
import io
from datetime import datetime
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

from prompt import ULTIMATE_PROMPT, DOCUMENT_SEPARATORS, DOCUMENT_NAMES

# Load environment variables
load_dotenv()

# Initialize OpenAI client with BuilderSpace API
client = OpenAI(
    base_url="https://space.ai-builders.com/backend/v1",
    api_key=os.getenv("AI_BUILDER_TOKEN")
)

app = FastAPI(
    title="Context Compiler",
    description="将长对话/会议记录编译成一套完整的产品规格文档",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    """Request model for document generation"""
    context: str
    project_name: Optional[str] = "未命名项目"
    model: Optional[str] = "gpt-5"


class DocumentResponse(BaseModel):
    """Response model for a single document"""
    name: str
    content: str


class GenerateResponse(BaseModel):
    """Response model for document generation"""
    success: bool
    project_name: str
    documents: list[DocumentResponse]
    generated_at: str
    raw_response: Optional[str] = None


def parse_documents(raw_text: str) -> list[dict]:
    """Parse the AI response into individual documents"""
    documents = []
    
    # Split by document separators
    for i, separator in enumerate(DOCUMENT_SEPARATORS):
        start_idx = raw_text.find(separator)
        if start_idx == -1:
            # Try alternative formats
            alt_separator = separator.replace("===== ", "=====").replace(" =====", "=====")
            start_idx = raw_text.find(alt_separator)
        
        if start_idx != -1:
            # Find the end (next separator or end of text)
            end_idx = len(raw_text)
            for j in range(i + 1, len(DOCUMENT_SEPARATORS)):
                next_sep_idx = raw_text.find(DOCUMENT_SEPARATORS[j])
                if next_sep_idx == -1:
                    alt_next = DOCUMENT_SEPARATORS[j].replace("===== ", "=====").replace(" =====", "=====")
                    next_sep_idx = raw_text.find(alt_next)
                if next_sep_idx != -1:
                    end_idx = next_sep_idx
                    break
            
            # Extract content
            content_start = start_idx + len(separator)
            content = raw_text[content_start:end_idx].strip()
            
            documents.append({
                "name": DOCUMENT_NAMES[i],
                "content": content
            })
    
    return documents


@app.get("/")
async def root():
    """Serve the main page"""
    static_path = Path(__file__).parent / "static" / "index.html"
    if static_path.exists():
        return FileResponse(static_path)
    return {"message": "Context Compiler API", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/api/generate", response_model=GenerateResponse)
async def generate_documents(request: GenerateRequest):
    """Generate specification documents from context (non-streaming)."""
    # Allow very short inputs as requested (only reject empty/whitespace)
    if not request.context or not request.context.strip():
        raise HTTPException(
            status_code=400,
            detail="Context is empty. 请至少输入一些内容。",
        )

    # Build the full prompt
    full_prompt = ULTIMATE_PROMPT + request.context

    # Determine target model (default to gpt-5, allow gemini-2.5-pro)
    model_name = (request.model or "gpt-5").strip()

    try:
        # Call BuilderSpace API
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": full_prompt},
            ],
            max_tokens=32000,
            temperature=1.0,
        )

        raw_response = response.choices[0].message.content

        # Parse documents from response
        documents = parse_documents(raw_response)

        if not documents:
            # If parsing failed, return raw response as a single document
            documents = [{
                "name": "完整输出.md",
                "content": raw_response,
            }]

        return GenerateResponse(
            success=True,
            project_name=request.project_name,
            documents=[DocumentResponse(**doc) for doc in documents],
            generated_at=datetime.now().isoformat(),
            raw_response=raw_response if len(documents) < 5 else None,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI generation failed: {str(e)}",
        )


@app.post("/api/generate-stream")
def generate_documents_stream(request: GenerateRequest):
    """Streaming version of document generation.

    Returns a JSON Lines (NDJSON) stream with events:
    - meta:           {"type": "meta", "project_name", "document_names", "generated_at"}
    - doc_started:    {"type": "doc_started", "doc_index"}
    - chunk:          {"type": "chunk", "doc_index", "delta"}
    - doc_complete:   {"type": "doc_complete", "doc_index"}
    - done:           {"type": "done"}
    - error:          {"type": "error", "message"}
    """

    if not request.context or not request.context.strip():
        raise HTTPException(
            status_code=400,
            detail="Context is empty. 请至少输入一些内容。",
        )

    project_name = request.project_name or "未命名项目"
    full_prompt = ULTIMATE_PROMPT + request.context

    # Determine target model (default to gpt-5, allow gemini-2.5-pro)
    model_name = (request.model or "gpt-5").strip()

    def event_stream():
        """Inner generator that yields NDJSON events."""
        try:
            # Send initial metadata so the frontend can prepare document slots
            meta_event = {
                "type": "meta",
                "project_name": project_name,
                "document_names": DOCUMENT_NAMES,
                "generated_at": datetime.now().isoformat(),
            }
            yield json.dumps(meta_event, ensure_ascii=False) + "\n"

            # Start streaming completion from BuilderSpace (OpenAI-compatible)
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=32000,
                temperature=1.0,
                stream=True,
            )

            buffer = ""
            current_doc_index = None
            max_sep_len = max(len(s) for s in DOCUMENT_SEPARATORS)

            def emit_doc_started(idx: int) -> str:
                return json.dumps({"type": "doc_started", "doc_index": idx}, ensure_ascii=False) + "\n"

            def emit_chunk(idx: int, text: str) -> str:
                return json.dumps({
                    "type": "chunk",
                    "doc_index": idx,
                    "delta": text,
                }, ensure_ascii=False) + "\n"

            def emit_doc_complete(idx: int) -> str:
                return json.dumps({"type": "doc_complete", "doc_index": idx}, ensure_ascii=False) + "\n"

            for chunk in completion:
                choice = chunk.choices[0]
                # In streaming mode, content arrives in delta
                if not hasattr(choice, "delta") or choice.delta is None:
                    continue
                delta_text = choice.delta.content
                if not delta_text:
                    continue

                buffer += delta_text

                while True:
                    # Until we see the first separator, don't emit any user-visible text
                    if current_doc_index is None:
                        first_sep = DOCUMENT_SEPARATORS[0]
                        sep_idx = buffer.find(first_sep)
                        if sep_idx == -1:
                            # Avoid unbounded buffer if, for some reason, the model
                            # does not respect the protocol; in that case, fall back
                            # to treating everything as the first doc after a while.
                            if len(buffer) > 4000:
                                current_doc_index = 0
                                yield emit_doc_started(current_doc_index)
                                safe_len = max(0, len(buffer) - (max_sep_len - 1))
                                if safe_len:
                                    text_to_send = buffer[:safe_len]
                                    buffer = buffer[safe_len:]
                                    yield emit_chunk(current_doc_index, text_to_send)
                            break

                        # Drop everything before the first separator and start DOC 0
                        buffer = buffer[sep_idx + len(first_sep) :]
                        current_doc_index = 0
                        yield emit_doc_started(current_doc_index)
                        continue

                    # We already started a document; look for the next separator
                    next_index = current_doc_index + 1
                    next_sep_idx = -1
                    if next_index < len(DOCUMENT_SEPARATORS):
                        next_sep = DOCUMENT_SEPARATORS[next_index]
                        next_sep_idx = buffer.find(next_sep)

                    if next_sep_idx == -1:
                        # No full next separator yet. Flush a "safe" prefix of the
                        # buffer so we don't cut a separator in half.
                        safe_len = len(buffer) - (max_sep_len - 1)
                        if safe_len > 0:
                            text_to_send = buffer[:safe_len]
                            buffer = buffer[safe_len:]
                            if text_to_send:
                                yield emit_chunk(current_doc_index, text_to_send)
                        break

                    # Found the next separator in the buffer
                    content = buffer[:next_sep_idx]
                    if content:
                        yield emit_chunk(current_doc_index, content)
                    yield emit_doc_complete(current_doc_index)

                    buffer = buffer[next_sep_idx + len(DOCUMENT_SEPARATORS[next_index]) :]
                    current_doc_index = next_index
                    yield emit_doc_started(current_doc_index)
                    # Loop again in case multiple separators are already in buffer

            # Stream has finished; flush any remaining content
            if current_doc_index is None:
                # We never saw a separator; treat everything as a single document 0
                residual = buffer.strip()
                if residual:
                    yield emit_doc_started(0)
                    yield emit_chunk(0, residual)
                    yield emit_doc_complete(0)
            else:
                residual = buffer.strip()
                if residual:
                    yield emit_chunk(current_doc_index, residual)
                yield emit_doc_complete(current_doc_index)

            yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"

        except Exception as e:
            error_event = {"type": "error", "message": str(e)}
            yield json.dumps(error_event, ensure_ascii=False) + "\n"

    return StreamingResponse(event_stream(), media_type="application/jsonl")


@app.post("/api/generate-from-file")
async def generate_from_file(
    file: UploadFile = File(...),
    project_name: str = Form(default="未命名项目"),
    model: str = Form(default="gpt-5"),
):
    """Generate documents from uploaded file"""
    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content.decode("gbk")
        except:
            raise HTTPException(status_code=400, detail="Could not decode file.")
    request = GenerateRequest(context=text, project_name=project_name, model=model)
    return await generate_documents(request)


@app.post("/api/download-zip")
async def download_zip(documents: list[DocumentResponse]):
    """Download all documents as a ZIP file"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for doc in documents:
            zf.writestr(doc.name, doc.content)
    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=context_compiler_output.zip"}
    )


# Mount static files (frontend)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

