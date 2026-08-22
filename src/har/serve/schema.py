"""HTTP request and response models for the HAR API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from har.types import Device

MAX_SAMPLES = 512
MAX_CHANNELS = 16
MAX_BODY_BYTES = 1_048_576


class PredictRequest(BaseModel):
    device: Device
    hz: float
    channels: list[str] = Field(min_length=1, max_length=MAX_CHANNELS)
    samples: list[list[float]] = Field(min_length=1, max_length=MAX_SAMPLES)

    @model_validator(mode="after")
    def samples_are_rectangular(self) -> PredictRequest:
        widths = {len(row) for row in self.samples}
        if len(widths) != 1:
            raise ValueError("samples rows must have equal length")
        width = next(iter(widths))
        if width > MAX_CHANNELS:
            raise ValueError(f"samples row length must be <= {MAX_CHANNELS}")
        return self


class PredictResponse(BaseModel):
    activity_code: str
    activity_name: str
    group: str
    proba: dict[str, float]
    confidence: float
    abstained: bool


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_id: str


class LabelsResponse(BaseModel):
    codes: list[str]
    names: dict[str, str]
    groups: dict[str, str]
