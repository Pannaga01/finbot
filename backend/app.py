# app.py

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from fastapi.middleware.cors import CORSMiddleware

from router import route_query
from rag import answer
from config import ROLE_COLLECTIONS

app = FastAPI(title="FinBot API", version="0.1")
# -------------------------------
# CORS Middleware
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # allow all origins
    allow_credentials=True,
    allow_methods=["*"],        # allow all HTTP methods
    allow_headers=["*"],        # allow all headers
)

# -------------------------------
# Request Schema
# -------------------------------
class QueryRequest(BaseModel):
    user_id: str
    role: str
    query: str


# -------------------------------
# Helpers
# -------------------------------
def get_session_id(request: Request) -> str:
    """
    Extract session identifier from headers or client info.
    """
    return (
        request.headers.get("x-session-id")
        or request.headers.get("x-forwarded-for", "").split(",")[0]
        or request.client.host
    )


def route_query_by_intent(body: QueryRequest, request: Request) -> str:
    """
    Route query using semantic router.
    Raises ValueError if blocked by guardrails.
    """
    session_id = get_session_id(request)
    return route_query(body.query, session_id)


def enforce_rbac(route: str, role: str) -> tuple:
    """
    Enforce role-based access control.
    """
    if route in ROLE_COLLECTIONS[role]:
        return True, None
    return False, f"Role '{role}' does not have access to route '{route}'"


def retrieve_answer(query: str, route: str, role: str):
    """
    Wrapper around RAG answer pipeline.
    """
    return answer(query, route, role)


# -------------------------------
# API Endpoint
# -------------------------------
@app.post("/query")
def query_endpoint(body: QueryRequest, request: Request):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    # -------------------------------
    # Semantic Routing + Guardrails
    # -------------------------------
    try:
        route = route_query_by_intent(body, request)
    except ValueError as guardrail_err:
        return {
            "role": body.role,
            "route": "blocked",
            "allowed_collections": [],
            "blocked": True,
            "block_reason": str(guardrail_err),
        }

    # -------------------------------
    # RBAC Enforcement
    # -------------------------------
    allowed, block_reason = enforce_rbac(route, body.role)
    if not allowed:
        return {
            "role": body.role,
            "route": route,
            "allowed_collections": [],
            "blocked": True,
            "block_reason": block_reason,
        }

    # -------------------------------
    # RAG Pipeline
    # -------------------------------
    ans, context = retrieve_answer(body.query, route, body.role)

    return {"answer": ans}