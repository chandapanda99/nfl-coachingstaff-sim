"""Azure AI Foundry adapter for the Responses API."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from nfl_coaching_sim.providers.base import (
    ModelConfiguration,
    ModelProvider,
    ProviderCapabilities,
    ProviderModel,
)


class AzureFoundryProvider:
    provider_id = ModelProvider.AZURE_FOUNDRY.value
    display_name = "Azure AI Foundry"
    capabilities = ProviderCapabilities(
        generation_parameters=frozenset({"reasoning_effort"}),
        api_mode="responses",
    )

    def validate_configuration(self, configuration: ModelConfiguration) -> None:
        endpoint = urlparse(configuration.base_url)
        if endpoint.scheme != "https":
            raise ValueError("Azure Foundry endpoints must use HTTPS")
        if not endpoint.path.rstrip("/").endswith("/openai/v1"):
            raise ValueError("Azure Foundry base_url must end with /openai/v1/")
        if configuration.provider_options:
            raise ValueError("Azure Foundry does not support provider_options")

    def build(self, configuration: ModelConfiguration) -> ProviderModel:
        from langchain_openai import ChatOpenAI

        self.validate_configuration(configuration)
        from nfl_coaching_sim.settings import get_application_settings

        api_key = get_application_settings(self.provider_id).foundry_api_key
        if api_key:
            credential: Any = api_key
            authentication = "api_key_environment"
        else:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider

            credential = get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default",
            )
            authentication = "default_azure_credential"

        generation_parameters = self.capabilities.select_generation_parameters(configuration)
        model_parameters = dict(generation_parameters)
        reasoning_effort = model_parameters.pop("reasoning_effort", None)
        if reasoning_effort is not None:
            # The Responses API accepts reasoning controls as a nested object.
            model_parameters["reasoning"] = {"effort": reasoning_effort}
        model = ChatOpenAI(
            model=configuration.model,
            base_url=configuration.base_url,
            api_key=credential,
            use_responses_api=True,
            **model_parameters,
        )
        if not model.use_responses_api:
            raise RuntimeError("Azure Foundry must use LangChain's Responses API mode")
        return ProviderModel(
            chat_model=model,
            authentication=authentication,
            effective_generation_parameters=generation_parameters,
        )
