"""Built-in model provider adapters."""

from nfl_coaching_sim.providers.azure_foundry import AzureFoundryProvider
from nfl_coaching_sim.providers.base import (
    ModelConfiguration,
    ModelProvider,
    ProviderAdapter,
    ProviderCapabilities,
    ProviderModel,
)
from nfl_coaching_sim.providers.ollama import OllamaProvider
from nfl_coaching_sim.providers.registry import (
    get_provider,
    model_provider_choices,
    register_model_provider,
    register_provider,
)

register_provider(OllamaProvider())
register_provider(AzureFoundryProvider())

__all__ = [
    "ModelConfiguration",
    "ModelProvider",
    "ProviderAdapter",
    "ProviderCapabilities",
    "ProviderModel",
    "get_provider",
    "model_provider_choices",
    "register_model_provider",
    "register_provider",
]
