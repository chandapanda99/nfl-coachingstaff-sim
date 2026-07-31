"""Provider contracts and shared model configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, Protocol
from urllib.parse import urlparse

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator

APPROVED_LICENSES = {
    "Apache-2.0",
    "MIT",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "MPL-2.0",
}


class ModelProvider(StrEnum):
    OLLAMA = "ollama"
    AZURE_FOUNDRY = "azure_foundry"


class ModelConfiguration(BaseModel):
    """Provider-neutral model selection and generation settings."""

    provider: str = ModelProvider.OLLAMA.value
    model: str = Field(min_length=1)
    base_url: str = "http://127.0.0.1:11434"
    upstream_url: HttpUrl | None = None
    license: str
    temperature: float = Field(default=0.0, ge=0, le=2)
    seed: int = 2026
    provider_options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider", mode="before")
    @classmethod
    def normalize_provider(cls, value: Any) -> str:
        normalized = str(value).strip().lower()
        if not normalized:
            raise ValueError("provider must not be empty")
        return normalized

    @model_validator(mode="after")
    def validate_common_settings(self) -> ModelConfiguration:
        if self.license not in APPROVED_LICENSES:
            raise ValueError(f"model license must be one of {sorted(APPROVED_LICENSES)}")
        endpoint = urlparse(self.base_url)
        if endpoint.scheme not in {"http", "https"} or not endpoint.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) endpoint")
        return self

    @property
    def model_id(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass(frozen=True)
class ProviderCapabilities:
    """Behavior a provider adapter explicitly promises to support."""

    generation_parameters: frozenset[str]
    api_mode: Literal["native", "chat_completions", "responses"]
    structured_output: bool = True

    def select_generation_parameters(self, configuration: ModelConfiguration) -> dict[str, Any]:
        configured = {
            "temperature": configuration.temperature,
            "seed": configuration.seed,
        }
        unknown = self.generation_parameters.difference(configured)
        if unknown:
            raise ValueError(f"provider declares unknown generation parameters: {sorted(unknown)}")
        return {name: configured[name] for name in self.generation_parameters}


@dataclass(frozen=True)
class ProviderModel:
    chat_model: Any
    authentication: str
    effective_generation_parameters: dict[str, Any]


class ProviderAdapter(Protocol):
    """Self-contained integration point for one LangChain model provider."""

    provider_id: str
    display_name: str
    capabilities: ProviderCapabilities

    def validate_configuration(self, configuration: ModelConfiguration) -> None: ...

    def build(self, configuration: ModelConfiguration) -> ProviderModel: ...
