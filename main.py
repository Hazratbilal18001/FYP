"""
FastAPI API server for the Healthcare AI Platform.
"""

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from database import (
    init_database,
    create_chat,
    get_all_chats,
    get_messages,
    update_chat_title,
    update_chat_domain,
    delete_chat,
)
from memory import load_chat_history, save_user_message, save_ai_message
from agent import stream_response


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="Healthcare AI Platform")
init_database()

if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


class CreateChatRequest(BaseModel):
    title: str = "New Chat"
    domain: str = "medical"


class UpdateChatRequest(BaseModel):
    title: str | None = None
    domain: str | None = None


class ChatRequest(BaseModel):
    message: str
    chat_id: int | None = None
    domain: str = "medical"


@app.get("/")
def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="frontend/index.html not found")


@app.get("/api/chats")
def api_get_chats():
    return get_all_chats()


@app.post("/api/chats")
def api_create_chat(payload: CreateChatRequest):
    chat_id = create_chat(payload.title, payload.domain)
    return {"chat_id": chat_id, "title": payload.title, "domain": payload.domain}


@app.delete("/api/chats/{chat_id}")
def api_delete_chat(chat_id: int):
    delete_chat(chat_id)
    return {"success": True}


@app.patch("/api/chats/{chat_id}")
def api_update_chat(chat_id: int, payload: UpdateChatRequest):
    if payload.title:
        update_chat_title(chat_id, payload.title)
    if payload.domain:
        update_chat_domain(chat_id, payload.domain)
    return {"success": True}


@app.get("/api/chats/{chat_id}/messages")
def api_get_messages(chat_id: int):
    return get_messages(chat_id)


@app.post("/api/chat")
def api_chat(payload: ChatRequest):
    user_message = payload.message.strip()
    chat_id = payload.chat_id
    domain = payload.domain

    if not user_message:
        raise HTTPException(status_code=400, detail="Message is required")

    if chat_id is None:
        title = user_message[:40] + ("..." if len(user_message) > 40 else "")
        chat_id = create_chat(title, domain)

    save_user_message(chat_id, user_message)
    chat_history = load_chat_history(chat_id)

    def generate():
        full_response = ""
        yield f"data: {json.dumps({'type': 'meta', 'chat_id': chat_id, 'domain': domain})}\n\n"
        try:
            for token in stream_response(chat_history, domain_hint=domain):
                full_response += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"
            return
        save_ai_message(chat_id, full_response)
        yield f"data: {json.dumps({'type': 'done', 'full_response': full_response})}\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(generate(), media_type="text/event-stream", headers=headers)


@app.get("/{path:path}")
def serve_static(path: str):
    file_path = FRONTEND_DIR / path
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
    return JSONResponse(status_code=404, content={"detail": "Not Found"})
