import os
import re
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from starlette.background import BackgroundTask
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
from typing import List, Tuple, Dict, Any, Set

API_KEY = os.getenv("API_KEY")
API_HOST = os.getenv("API_HOST")

# --- Access Control Configuration ---
ALLOWED_PATHS_AND_METHODS: List[Tuple[str, str]] = [
    ("GET", r"models$"),
    ("GET", r"models/[^/]+/[^/]+/endpoints$"),
    ("POST", r"chat/completions$"),
    ("POST", r"completions$"),
    ("POST", r"embeddings$"),
    ("GET", r"generation$"),
    ("POST", r"moderations$"),
]
DENIED_PATHS_AND_METHODS: List[Tuple[str, str]] = [
    ("GET", r"credits$"),
    ("GET", r"key$"),
    ("GET", r"keys$"),
    ("POST", r"keys$"),
    ("POST", r"credits/coinbase$"),
    ("POST", r"auth/keys$"),
]
COMPILED_ALLOWED_PATHS = [
    (method, re.compile(pattern)) for method, pattern in ALLOWED_PATHS_AND_METHODS
]
COMPILED_DENIED_PATHS = [
    (method, re.compile(pattern)) for method, pattern in DENIED_PATHS_AND_METHODS
]

# --- Parameter Configuration ---
# Constant 1: "supported_parameters" (parameters generally supported by completion-like models)
_SUPPORTED_PARAMETERS_LIST: List[Dict[str, Any]] = [
    {"label": "max_tokens", "type": "number", "min": 1, "max": 4096, "placeholder": "256"},
    {"label": "temperature", "type": "number", "step": 0.01, "min": 0, "max": 2, "placeholder": "0.7"},
    {"label": "stop", "type": "textarea", "placeholder": "\\n, ###, [END]", "rows": 1},
    {"label": "reasoning", "type": "checkbox", "checked": True},
    {"label": "include_reasoning", "type": "checkbox"},
    {"label": "tools", "type": "textarea", "placeholder": '[{"type": "calculator"}, {"type": "search"}]', "rows": 1},
    {"label": "tool_choice", "type": "text", "placeholder": "calculator"},
    {"label": "top_p", "type": "number", "step": 0.01, "min": 0, "max": 1, "placeholder": "1.0"},
    {"label": "top_k", "type": "number", "min": 1, "max": 100, "placeholder": "1"},
    {"label": "min_p", "type": "number", "step": 0.01, "min": 0, "max": 1, "placeholder": "0.0"},
    {"label": "top_a", "type": "number", "step": 0.01, "min": 0, "max": 1, "placeholder": "0.0"},
    {"label": "seed", "type": "number", "placeholder": "42"},
    {"label": "presence_penalty", "type": "number", "step": 0.01, "min": -2, "max": 2, "placeholder": "0.0"},
    {"label": "frequency_penalty", "type": "number", "step": 0.01, "min": -2, "max": 2, "placeholder": "0.0"},
    {"label": "repetition_penalty", "type": "number", "step": 0.01, "min": 1, "max": 2, "placeholder": "1.0"},
    {"label": "logit_bias", "type": "textarea", "placeholder": '{"50256": -100, "198": 5}', "rows": 1},
    {"label": "logprobs", "type": "number", "min": 1, "max": 5, "placeholder": "1"},
    {"label": "top_logprobs", "type": "number", "min": 1, "max": 5, "placeholder": "1"},
    {"label": "response_format", "type": "text", "placeholder": "json_object"},
    {"label": "structured_outputs", "type": "checkbox"},
    {"label": "web_search_options", "type": "textarea", "placeholder": '{"region": "us", "num_results": 5}', "rows": 1},
]

# Constant 2: "second is ... it is requaired parameters" (parameters specific or essential for chat completions)
_CHAT_COMPLETIONS_SPECIFIC_PARAMETERS_LIST: List[Dict[str, Any]] = [
    {"label": "messages", "type": "textarea", "placeholder": '[{"role": "user", "content": "AHOY!"}]', "rows": 1},
    {"label": "model", "type": "text", "placeholder": "gpt-3.5-turbo"},
]

# VALID_PARAMETERS_CONFIG is the combination of these two lists, specifically for chat/completions validation
VALID_PARAMETERS_CONFIG: List[Dict[str, Any]] = _SUPPORTED_PARAMETERS_LIST + _CHAT_COMPLETIONS_SPECIFIC_PARAMETERS_LIST
VALID_PARAMETER_NAMES: Set[str] = {param["label"] for param in VALID_PARAMETERS_CONFIG}


def get_lifespan():
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient(timeout=None) as client:
            app.state.http = client
            yield
    return lifespan


app = FastAPI(lifespan=get_lifespan())

# --- 2. ADD CORS MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- END OF CORS MIDDLEWARE ADDITION ---


async def check_access_and_parameters(
    method: str,
    path: str,
    query_params: Dict[str, Any],
    request_body_bytes: bytes,
    content_type: str | None,
):
    # 1. Check if explicitly denied
    for denied_method, denied_pattern in COMPILED_DENIED_PATHS:
        if method == denied_method and denied_pattern.fullmatch(path):
            raise HTTPException(
                status_code=403,
                detail=f"Access to {method} /{path} is explicitly forbidden.",
            )

    # 2. Check if allowed
    is_path_allowed = False
    for allowed_method, allowed_pattern in COMPILED_ALLOWED_PATHS:
        if method == allowed_method and allowed_pattern.fullmatch(path):
            is_path_allowed = True
            break

    if not is_path_allowed:
        raise HTTPException(
            status_code=403, detail=f"Access to {method} /{path} is not allowed."
        )

    # 3. Validate parameters (query and body) *only* for "chat/completions"
    #    The `path` variable here corresponds to `full_path` from the proxy route.
    if path == "chat/completions":
        all_request_params = set(query_params.keys())

        # Chat completions typically use POST with a JSON body
        if method == "POST" and request_body_bytes:
            if content_type and "application/json" in content_type.lower():
                try:
                    body_json = json.loads(request_body_bytes.decode() or "{}")
                    if isinstance(body_json, dict):
                        all_request_params.update(body_json.keys())
                except json.JSONDecodeError:
                    raise HTTPException(status_code=400, detail="Invalid JSON body for chat/completions.")
            # If not JSON, but body exists for POST to chat/completions, it's likely an issue,
            # but original code didn't strictly enforce JSON here if body_json failed to load keys.
            # Upstream will likely reject non-JSON for chat/completions.

        for param_name in all_request_params:
            if param_name == "stream": # 'stream' is generally handled by client, not a model param.
                continue
            if param_name not in VALID_PARAMETER_NAMES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid parameter for {path}: '{param_name}'. Allowed parameters for this endpoint: {', '.join(sorted(list(VALID_PARAMETER_NAMES)))} or 'stream'.",
                )
    # For other allowed paths (e.g., "completions", "embeddings"),
    # parameter validation against VALID_PARAMETER_NAMES is skipped.
    # The upstream API will handle validation for those.


@app.api_route(
    "/api-v1/{full_path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    response_class=StreamingResponse,
)
async def proxy(full_path: str, request: Request):
    request_body_bytes = await request.body()

    await check_access_and_parameters(
        method=request.method,
        path=full_path,
        query_params=dict(request.query_params),
        request_body_bytes=request_body_bytes,
        content_type=request.headers.get("content-type"),
    )

    target_url = f"https://{API_HOST}/api/v1/{full_path}"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    if request.headers.get("content-type"):
        headers["Content-Type"] = request.headers.get("content-type")

    stream_ctx = app.state.http.stream(
        method=request.method,
        url=target_url,
        headers=headers,
        params=request.query_params,
        content=request_body_bytes,
        timeout=None,
    )

    try:
        resp = await stream_ctx.__aenter__()
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        error_body = await exc.response.aread()
        await resp.aclose()
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=error_body.decode(errors="replace"),
        )
    except httpx.RequestError as exc:
        print(f"RequestError connecting to target: {exc}") # Keep for debugging
        await stream_ctx.__aexit__(None, None, None) # Ensure context is exited on RequestError too
        raise HTTPException(
            status_code=502,
            detail=f"Error connecting to upstream service: {exc!r}",
        )

    background = BackgroundTask(stream_ctx.__aexit__, None, None, None)
    return StreamingResponse(
        resp.aiter_raw(),
        status_code=resp.status_code,
        headers={
            k: v
            for k, v in resp.headers.items()
            if k.lower() not in ("transfer-encoding", "connection")
        },
        media_type=resp.headers.get("content-type"),
        background=background,
    )


if __name__ == "__main__":
    import uvicorn

    if not API_KEY or not API_HOST:
        print("Warning: API_KEY and API_HOST environment variables are not set.")
        print("Using placeholders for testing.")
        os.environ.setdefault("API_KEY", "test_api_key")
        os.environ.setdefault("API_HOST", "dummy.api.host")

    uvicorn.run(app, host="0.0.0.0", port=8000)