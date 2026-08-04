from __future__ import annotations

import textwrap

import pytest

from llm.providers.registry import get_provider, load_providers
from llm.types import ProviderConfigError


def test_get_provider_unknown_name_lists_valid_names():
    with pytest.raises(ProviderConfigError) as exc_info:
        get_provider("not-a-real-provider")
    message = str(exc_info.value)
    assert "not-a-real-provider" in message
    assert "mock" in message


def test_mock_provider_always_registered():
    assert get_provider("mock").name == "mock"


def test_load_providers_rejects_malformed_entry(tmp_path):
    bad_yaml = tmp_path / "quotas.yaml"
    bad_yaml.write_text(
        textwrap.dedent(
            """
            providers:
              broken:
                base_url: "https://example.com"
            """
        )
    )
    with pytest.raises(ProviderConfigError) as exc_info:
        load_providers(bad_yaml)
    assert "broken" in str(exc_info.value)


def test_load_providers_registers_all_configured_providers(tmp_path):
    path = tmp_path / "quotas.yaml"
    path.write_text(
        textwrap.dedent(
            """
            providers:
              testprov:
                base_url: "https://example.com/v1"
                api_key_env: "TESTPROV_API_KEY"
                default_model: "test-model"
                reset_window: "per minute"
                last_verified: "2026-08-04"
            """
        )
    )
    providers = load_providers(path)
    assert set(providers) == {"mock", "testprov"}
    assert providers["testprov"].name == "testprov"


def test_load_providers_always_includes_mock_even_with_empty_file(tmp_path):
    path = tmp_path / "quotas.yaml"
    path.write_text("providers: {}\n")
    providers = load_providers(path)
    assert set(providers) == {"mock"}
