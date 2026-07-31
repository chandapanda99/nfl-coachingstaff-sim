"""Registry and backwards-compatible registration for model providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from nfl_coaching_sim.providers.base import (
    ModelConfiguration,
    ProviderAdapter,
    ProviderCapabilities,
    ProviderModel,
)

ProviderBuilder = Callable[[ModelConfiguration], tuple[object, str]]
ProviderValidator = Callable[[ModelConfiguration], None]
_PROVIDERS: dict[str, ProviderAdapter] = {}


def register_provider(adapter: ProviderAdapter) -> None:
    """Register or replace a complete provider adapter."""

    provider_id = adapter.provider_id.strip().lower()
    if not provider_id:
        raise ValueError("provider must not be empty")
    _PROVIDERS[provider_id] = adapter


def get_provider(provider_id: str) -> ProviderAdapter:
    try:
        return _PROVIDERS[provider_id]
    except KeyError as error:
        raise ValueError(f"unsupported model provider: {provider_id}; registered providers: {sorted(_PROVIDERS)}") from error


def model_provider_choices() -> list[tuple[str, str]]:
    return [(adapter.display_name, provider_id) for provider_id, adapter in _PROVIDERS.items()]


@dataclass(frozen=True)
class FunctionProviderAdapter:
    """Compatibility adapter for lightweight third-party registrations."""

    provider_id: str
    display_name: str
    builder: ProviderBuilder
    capabilities: ProviderCapabilities
    validator: ProviderValidator | None = None

    def validate_configuration(self, configuration: ModelConfiguration) -> None:
        if self.validator is not None:
            self.validator(configuration)

    def build(self, configuration: ModelConfiguration) -> ProviderModel:
        self.validate_configuration(configuration)
        chat_model, authentication = self.builder(configuration)
        return ProviderModel(
            chat_model=chat_model,
            authentication=authentication,
            effective_generation_parameters=self.capabilities.select_generation_parameters(configuration),
        )


def register_model_provider(
    provider: str,
    builder: ProviderBuilder,
    label: str | None = None,
    *,
    capabilities: ProviderCapabilities | None = None,
    validator: ProviderValidator | None = None,
) -> None:
    """Register a function-based provider without changing orchestration code."""

    provider_id = str(provider).strip().lower()
    register_provider(
        FunctionProviderAdapter(
            provider_id=provider_id,
            display_name=label or provider_id.replace("_", " ").title(),
            builder=builder,
            capabilities=capabilities
            or ProviderCapabilities(
                generation_parameters=frozenset(),
                api_mode="native",
            ),
            validator=validator,
        )
    )
