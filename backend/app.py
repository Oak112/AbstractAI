"""Abstraction AI - Main FastAPI Application"""

import os
import re
import json
import zipfile
import io
import time
import threading
import queue
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

from prompt import get_prompt_bundle

# Load environment variables
load_dotenv()

# Initialize OpenAI client with BuilderSpace API
client = OpenAI(
    base_url="https://space.ai-builders.com/backend/v1",
    api_key=os.getenv("AI_BUILDER_TOKEN")
)

app = FastAPI(
    title="Abstraction AI",
    description="Compile long conversations into executable product specification documents",
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
    project_name: Optional[str] = None
    model: Optional[str] = "gpt-5"
    lang: Optional[str] = "en"


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
    """Parse the AI response into individual documents (legacy wrapper)."""
    return parse_documents_with_bundle(raw_text, lang="en")


def parse_documents_with_bundle(raw_text: str, *, lang: str | None) -> list[dict]:
    bundle = get_prompt_bundle(lang)
    documents = []

    for i, separator in enumerate(bundle.document_separators):
        matched_separator = separator
        start_idx = raw_text.find(matched_separator)
        if start_idx == -1:
            alt_separator = separator.replace("===== ", "=====").replace(" =====", "=====")
            start_idx = raw_text.find(alt_separator)
            if start_idx != -1:
                matched_separator = alt_separator

        if start_idx != -1:
            end_idx = len(raw_text)
            for j in range(i + 1, len(bundle.document_separators)):
                next_sep = bundle.document_separators[j]
                next_sep_idx = raw_text.find(next_sep)
                if next_sep_idx == -1:
                    alt_next = next_sep.replace("===== ", "=====").replace(" =====", "=====")
                    next_sep_idx = raw_text.find(alt_next)
                if next_sep_idx != -1:
                    end_idx = next_sep_idx
                    break

            content_start = start_idx + len(matched_separator)
            content = raw_text[content_start:end_idx].strip()

            documents.append({
                "name": bundle.document_names[i],
                "content": content,
            })

    return documents


@app.get("/")
async def root():
    """Serve the main page (default: English)."""
    static_dir = Path(__file__).parent / "static"
    en_path = static_dir / "index.en.html"
    zh_path = static_dir / "index.html"
    if en_path.exists():
        return FileResponse(en_path)
    if zh_path.exists():
        return FileResponse(zh_path)
    return {"message": "Abstraction AI API", "status": "running"}


@app.get("/en")
async def root_en():
    """Serve the English page."""
    static_path = Path(__file__).parent / "static" / "index.en.html"
    if static_path.exists():
        return FileResponse(static_path)
    return await root()


@app.get("/zh")
async def root_zh():
    """Serve the Chinese page."""
    static_path = Path(__file__).parent / "static" / "index.html"
    if static_path.exists():
        return FileResponse(static_path)
    return await root()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/api/generate", response_model=GenerateResponse)
async def generate_documents(request: GenerateRequest):
    """Generate specification documents from context (non-streaming)."""
    bundle = get_prompt_bundle(request.lang)
    # Allow very short inputs as requested (only reject empty/whitespace)
    if not request.context or not request.context.strip():
        raise HTTPException(
            status_code=400,
            detail="Context is empty. Please provide some text." if bundle.lang == "en" else "上下文为空，请至少输入一些内容。",
        )

    # Build the full prompt
    full_prompt = bundle.ultimate_prompt + request.context

    # Determine target model (default to gpt-5, allow gemini-2.5-pro)
    model_name = (request.model or "gpt-5").strip()
    project_name = (request.project_name or "").strip() or bundle.default_project_name

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
        documents = parse_documents_with_bundle(raw_response, lang=bundle.lang)

        if not documents:
            # If parsing failed, return raw response as a single document
            documents = [{
                "name": bundle.full_output_name,
                "content": raw_response,
            }]

        return GenerateResponse(
            success=True,
            project_name=project_name,
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

    bundle = get_prompt_bundle(request.lang)

    if not request.context or not request.context.strip():
        raise HTTPException(
            status_code=400,
            detail="Context is empty. Please provide some text." if bundle.lang == "en" else "上下文为空，请至少输入一些内容。",
        )

    project_name = (request.project_name or "").strip() or bundle.default_project_name
    full_prompt = bundle.ultimate_prompt + request.context

    # Determine target model (default to gpt-5, allow gemini-2.5-pro)
    model_name = (request.model or "gpt-5").strip()

    # Gemini models don't support streaming in BuilderSpace API
    # Fall back to non-streaming mode for gemini
    is_gemini = "gemini" in model_name.lower()

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

    def event_stream_gemini():
        """Non-streaming mode for Gemini with heartbeat to prevent connection timeout.

        Uses a background thread for the API call while the main generator
        sends heartbeat events every 5 seconds to keep the connection alive.
        """
        result_queue = queue.Queue()

        def api_call_thread():
            """Background thread to make the API call."""
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": full_prompt}],
                    max_tokens=32000,
                    temperature=1.0,
                )
                raw_text = response.choices[0].message.content or ""
                result_queue.put(("success", raw_text))
            except Exception as e:
                result_queue.put(("error", str(e)))

        try:
            # Send initial metadata
            meta_event = {
                "type": "meta",
                "project_name": project_name,
                "document_names": bundle.document_names,
                "generated_at": datetime.now().isoformat(),
            }
            yield json.dumps(meta_event, ensure_ascii=False) + "\n"

            # Start API call in background thread
            thread = threading.Thread(target=api_call_thread, daemon=True)
            thread.start()

            # Send heartbeats while waiting for API response
            heartbeat_interval = 5  # seconds
            max_wait_time = 180  # 3 minutes max
            elapsed = 0

            while thread.is_alive() and elapsed < max_wait_time:
                try:
                    # Check if result is ready (non-blocking with short timeout)
                    result = result_queue.get(timeout=heartbeat_interval)
                    break  # Got result, exit loop
                except queue.Empty:
                    # No result yet, send heartbeat to keep connection alive
                    heartbeat_event = {
                        "type": "heartbeat",
                        "elapsed_seconds": elapsed + heartbeat_interval,
                        "message": "Gemini is thinking… please wait." if bundle.lang == "en" else "Gemini 正在思考中，请稍候..."
                    }
                    yield json.dumps(heartbeat_event, ensure_ascii=False) + "\n"
                    elapsed += heartbeat_interval
            else:
                # Loop ended without break - either timeout or thread died
                if elapsed >= max_wait_time:
                    raise Exception("Gemini API timed out. Please try again." if bundle.lang == "en" else "Gemini API 响应超时，请稍后重试")
                # Try to get result one more time
                try:
                    result = result_queue.get(timeout=1)
                except queue.Empty:
                    raise Exception("Gemini API call failed." if bundle.lang == "en" else "Gemini API 调用失败")

            # Process result
            status, data = result
            if status == "error":
                raise Exception(data)

            raw_text = data

            # Parse documents from raw response
            documents = parse_documents_with_bundle(raw_text, lang=bundle.lang)
            if not documents:
                documents = [{"name": bundle.full_output_name, "content": raw_text}]

            # Emit events for each document
            for idx, doc in enumerate(documents):
                yield emit_doc_started(idx)
                yield emit_chunk(idx, doc["content"])
                yield emit_doc_complete(idx)

            yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"

        except Exception as e:
            error_event = {"type": "error", "message": str(e)}
            yield json.dumps(error_event, ensure_ascii=False) + "\n"

    def event_stream():
        """Streaming mode for GPT models with heartbeat to prevent connection timeout."""
        try:
            meta_event = {
                "type": "meta",
                "project_name": project_name,
                "document_names": bundle.document_names,
                "generated_at": datetime.now().isoformat(),
            }
            yield json.dumps(meta_event, ensure_ascii=False) + "\n"

            # Use a queue and thread to enable heartbeat during slow API response
            chunk_queue = queue.Queue()
            api_error = {"error": None}

            def stream_api_call():
                """Background thread to stream chunks from API."""
                try:
                    completion = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": full_prompt}],
                        max_tokens=32000,
                        temperature=1.0,
                        stream=True,
                    )
                    for chunk in completion:
                        chunk_queue.put(("chunk", chunk))
                    chunk_queue.put(("done", None))
                except Exception as e:
                    api_error["error"] = str(e)
                    chunk_queue.put(("error", str(e)))

            # Start API call in background thread
            api_thread = threading.Thread(target=stream_api_call, daemon=True)
            api_thread.start()

            buffer = ""
            current_doc_index = None
            max_sep_len = max(len(s) for s in bundle.document_separators)
            last_chunk_time = time.time()
            heartbeat_interval = 10  # Send heartbeat every 10 seconds of no data
            total_elapsed = 0
            max_wait = 300  # 5 minutes max total wait

            while True:
                try:
                    # Try to get next chunk with timeout
                    msg_type, msg_data = chunk_queue.get(timeout=heartbeat_interval)
                    last_chunk_time = time.time()

                    if msg_type == "error":
                        raise Exception(msg_data)
                    elif msg_type == "done":
                        break  # Exit main loop, proceed to flush remaining

                    # Process the chunk
                    chunk = msg_data
                    choice = chunk.choices[0]
                    if not hasattr(choice, "delta") or choice.delta is None:
                        continue
                    delta_text = choice.delta.content
                    if not delta_text:
                        continue

                    buffer += delta_text

                    # Parse and emit document chunks
                    while True:
                        if current_doc_index is None:
                            first_sep = bundle.document_separators[0]
                            sep_idx = buffer.find(first_sep)
                            if sep_idx == -1:
                                if len(buffer) > 4000:
                                    current_doc_index = 0
                                    yield emit_doc_started(current_doc_index)
                                    safe_len = max(0, len(buffer) - (max_sep_len - 1))
                                    if safe_len:
                                        text_to_send = buffer[:safe_len]
                                        buffer = buffer[safe_len:]
                                        yield emit_chunk(current_doc_index, text_to_send)
                                break

                            buffer = buffer[sep_idx + len(first_sep):]
                            current_doc_index = 0
                            yield emit_doc_started(current_doc_index)
                            continue

                        next_index = current_doc_index + 1
                        next_sep_idx = -1
                        if next_index < len(bundle.document_separators):
                            next_sep = bundle.document_separators[next_index]
                            next_sep_idx = buffer.find(next_sep)

                        if next_sep_idx == -1:
                            safe_len = len(buffer) - (max_sep_len - 1)
                            if safe_len > 0:
                                text_to_send = buffer[:safe_len]
                                buffer = buffer[safe_len:]
                                if text_to_send:
                                    yield emit_chunk(current_doc_index, text_to_send)
                            break

                        content = buffer[:next_sep_idx]
                        if content:
                            yield emit_chunk(current_doc_index, content)
                        yield emit_doc_complete(current_doc_index)

                        buffer = buffer[next_sep_idx + len(bundle.document_separators[next_index]):]
                        current_doc_index = next_index
                        yield emit_doc_started(current_doc_index)

                except queue.Empty:
                    # No chunk received within heartbeat_interval, send heartbeat
                    total_elapsed += heartbeat_interval
                    if total_elapsed >= max_wait:
                        raise Exception("GPT API timed out. Please try again." if bundle.lang == "en" else "GPT API 响应超时，请稍后重试")
                    heartbeat_event = {
                        "type": "heartbeat",
                        "elapsed_seconds": total_elapsed,
                        "message": "GPT is thinking… please wait." if bundle.lang == "en" else "GPT 正在思考中，请稍候..."
                    }
                    yield json.dumps(heartbeat_event, ensure_ascii=False) + "\n"
                    continue

            # Stream finished; flush remaining
            if current_doc_index is None:
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

    # Use appropriate generator based on model
    if is_gemini:
        return StreamingResponse(event_stream_gemini(), media_type="application/jsonl")
    else:
        return StreamingResponse(event_stream(), media_type="application/jsonl")


@app.post("/api/generate-from-file")
async def generate_from_file(
    file: UploadFile = File(...),
    project_name: str = Form(default=""),
    model: str = Form(default="gpt-5"),
    lang: str = Form(default="en"),
):
    """Generate documents from uploaded file"""
    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content.decode("gbk")
        except:
            bundle = get_prompt_bundle(lang)
            raise HTTPException(
                status_code=400,
                detail="Could not decode file." if bundle.lang == "en" else "文件解码失败，请上传 UTF-8 或 GBK 编码的文本文件。",
            )
    request = GenerateRequest(context=text, project_name=project_name, model=model, lang=lang)
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
