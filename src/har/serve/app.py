"""FastAPI app: health, labels, and one-window predict."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from har.constants import ACTIVITY_CODES, CODE_TO_NAME, GROUP_OF
from har.models.export import ModelBundle, bundle_from_env, predict_window
from har.serve.schema import (
    MAX_BODY_BYTES,
    HealthResponse,
    LabelsResponse,
    PredictRequest,
    PredictResponse,
)


def create_app(bundle: ModelBundle | None = None) -> FastAPI:
    """Build an app. Tests pass a bundle; uvicorn loads HAR_MODEL_PATH on startup."""
    holder: dict[str, ModelBundle | None] = {"bundle": bundle}

    @asynccontextmanager
    async def lifespan(_application: FastAPI) -> AsyncIterator[None]:
        if holder["bundle"] is None:
            holder["bundle"] = bundle_from_env()
        yield

    application = FastAPI(title="WISDM HAR", version="0.1.0", lifespan=lifespan)

    @application.middleware("http")
    async def cap_body(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        length = request.headers.get("content-length")
        if length is not None:
            try:
                n = int(length)
            except ValueError:
                n = MAX_BODY_BYTES + 1
            if n > MAX_BODY_BYTES:
                return JSONResponse({"detail": "request too large"}, status_code=413)
        return await call_next(request)

    def _bundle() -> ModelBundle:
        loaded = holder["bundle"]
        if loaded is None:
            loaded = bundle_from_env()
            holder["bundle"] = loaded
        return loaded

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", model_id=_bundle().model_id)

    @application.get("/labels", response_model=LabelsResponse)
    def labels() -> LabelsResponse:
        return LabelsResponse(
            codes=list(ACTIVITY_CODES),
            names=dict(CODE_TO_NAME),
            groups=dict(GROUP_OF),
        )

    @application.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        loaded = _bundle()
        err = _shape_error(req, loaded)
        if err is not None:
            raise HTTPException(status_code=422, detail=err)
        try:
            payload = predict_window(loaded, np.asarray(req.samples, dtype=np.float32))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return PredictResponse(**payload)

    return application


def _shape_error(req: PredictRequest, bundle: ModelBundle) -> str | None:
    t = len(req.samples)
    if t != bundle.n_timesteps:
        return f"expected T={bundle.n_timesteps}, got {t}"
    width = len(req.samples[0])
    if width != bundle.n_channels:
        return f"expected C={bundle.n_channels}, got {width}"
    if len(req.channels) != bundle.n_channels:
        return f"expected C={bundle.n_channels} channel names, got {len(req.channels)}"
    expected_names = tuple(bundle.channel_names[: bundle.n_channels])
    if tuple(req.channels) != expected_names:
        return f"channel names must be {list(expected_names)} in that order"
    if abs(float(req.hz) - float(bundle.hz)) > 1e-6:
        return f"expected hz={bundle.hz}, got {req.hz}"
    if req.device != bundle.device:
        return f"expected device={bundle.device}, got {req.device}"
    return None


app = create_app()
