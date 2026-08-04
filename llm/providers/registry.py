from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from llm.providers.base import OpenAICompatibleProvider, Provider
from llm.providers.mock import MockProvider
from llm.types import ProviderConfigError

DEFAULT_QUOTAS_PATH = Path(__file__).parent / "quotas.yaml"


class ProviderQuotaConfig(BaseModel):
    base_url: str
    api_key_env: str
    default_model: str
    reset_window: str
    rpm_limit: int | None = None
    input_price_per_1m: float | None = None
    output_price_per_1m: float | None = None
    last_verified: str


def load_providers(path: Path = DEFAULT_QUOTAS_PATH) -> dict[str, Provider]:
    """Build the provider map from a quotas.yaml file. `mock` is always included.

    Fails fast with a clear ProviderConfigError on a malformed entry - a broken
    config should crash at import time, not on the first real request.
    """
    raw = yaml.safe_load(path.read_text()) or {}
    entries = raw.get("providers", {}) or {}

    providers: dict[str, Provider] = {"mock": MockProvider()}
    for name, entry in entries.items():
        try:
            config = ProviderQuotaConfig.model_validate(entry)
        except ValidationError as exc:
            raise ProviderConfigError(
                f"{path}: invalid entry for provider '{name}': {exc}"
            ) from exc
        providers[name] = OpenAICompatibleProvider(
            name=name,
            base_url=config.base_url,
            api_key_env=config.api_key_env,
            default_model=config.default_model,
            input_price_per_1m=config.input_price_per_1m,
            output_price_per_1m=config.output_price_per_1m,
        )
    return providers


_PROVIDERS = load_providers()


def get_provider(name: str) -> Provider:
    try:
        return _PROVIDERS[name]
    except KeyError:
        valid = ", ".join(sorted(_PROVIDERS))
        raise ProviderConfigError(
            f"unknown llm provider '{name}'. Valid providers: {valid}"
        ) from None
