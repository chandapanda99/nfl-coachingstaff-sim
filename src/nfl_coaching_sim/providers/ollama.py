"""Ollama provider adapter."""

from __future__ import annotations

from nfl_coaching_sim.providers.base import (
    ModelConfiguration,
    ModelProvider,
    ProviderCapabilities,
    ProviderModel,
)


class OllamaProvider:
    provider_id = ModelProvider.OLLAMA.value
    display_name = "Local sideline (Ollama)"
    capabilities = ProviderCapabilities(
        generation_parameters=frozenset({"temperature", "seed"}),
        api_mode="native",
    )

    def validate_configuration(self, configuration: ModelConfiguration) -> None:
        if configuration.provider_options:
            raise ValueError("Ollama does not support provider_options")

    def build(self, configuration: ModelConfiguration) -> ProviderModel:
        from langchain_ollama import ChatOllama

        self.validate_configuration(configuration)
        generation_parameters = self.capabilities.select_generation_parameters(configuration)
        model = ChatOllama(
            model=configuration.model,
            base_url=configuration.base_url,
            validate_model_on_init=False,
            **generation_parameters,
        )
        return ProviderModel(
            chat_model=model,
            authentication="local",
            effective_generation_parameters=generation_parameters,
        )
