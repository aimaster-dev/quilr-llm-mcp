"""Configuration: defaults, flag/env precedence, validation and the multi-worker env round trip."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_router.keys import fingerprint_api_key
from llm_router.settings import Settings, load_settings

BASE_ENV = {
    "LLM_ROUTER_PRIMARY_URL": "http://127.0.0.1:9001/v1",
    "LLM_ROUTER_PRIMARY_MODEL": "primary-model",
    "LLM_ROUTER_TENANT_KEYS": "key-one, key-two",
}


def test_defaults_match_the_spec() -> None:
    settings = load_settings([], BASE_ENV)
    assert (settings.db_busy_timeout, settings.db_fail_fast) == (1.0, 1.0)
    assert settings.primary_timeout == 3.0  # 3000 ms
    assert settings.tokens_per_window == 50_000
    assert settings.window_seconds == 60.0
    assert settings.failover_on_connection_error and settings.failover_on_server_error
    assert settings.secondary_upstream() is None
    assert settings.primary_upstream().timeout == 3.0


def test_flags_override_environment() -> None:
    env = {
        **BASE_ENV,
        "LLM_ROUTER_PRIMARY_TIMEOUT_MS": "2500",
        "LLM_ROUTER_TOKENS_PER_WINDOW": "1000",
        "LLM_ROUTER_FAILOVER_ON_SERVER_ERROR": "false",
    }
    argv = ["--primary-timeout-ms", "1500", "--failover-on-server-error"]
    argv += ["--secondary-url", "http://b/v1", "--secondary-model", "backup"]
    settings = load_settings(argv, env)
    assert settings.primary_timeout == 1.5
    assert settings.tokens_per_window == 1_000
    assert settings.failover_on_server_error is True
    assert settings.secondary_upstream() is not None


def test_failover_switches() -> None:
    defaults = load_settings([], BASE_ENV).failover_policy()
    assert (defaults.on_connection_error, defaults.on_server_error, defaults.on_refused) == (True, True, True)
    argv = ["--no-failover-on-connection-error", "--no-failover-on-server-error", "--no-failover-on-refused"]
    policy = load_settings(argv, BASE_ENV).failover_policy()
    assert (policy.on_connection_error, policy.on_server_error, policy.on_refused) == (False, False, False)
    env = {**BASE_ENV, "LLM_ROUTER_FAILOVER_ON_REFUSED": "off"}
    assert load_settings([], env).failover_policy().on_refused is False


def test_tenant_keys_are_kept_only_as_fingerprints() -> None:
    fingerprint = fingerprint_api_key("key-three")
    settings = load_settings([], {**BASE_ENV, "LLM_ROUTER_TENANT_KEY_SHA256": fingerprint.upper()})
    assert settings.tenant_key_fingerprints == {
        fingerprint_api_key("key-one"),
        fingerprint_api_key("key-two"),
        fingerprint,
    }
    assert "key-one" not in repr(settings)


def test_secrets_stay_out_of_repr() -> None:
    env = {**BASE_ENV, "LLM_ROUTER_PRIMARY_API_KEY": "sk-primary-secret"}
    settings = load_settings([], env)
    assert settings.primary_api_key == "sk-primary-secret"
    assert "sk-primary-secret" not in repr(settings)
    assert "sk-primary-secret" not in repr(settings.primary_upstream())


@pytest.mark.parametrize(
    ("argv", "env"),
    [
        ([], {k: v for k, v in BASE_ENV.items() if k != "LLM_ROUTER_TENANT_KEYS"}),
        ([], {k: v for k, v in BASE_ENV.items() if k != "LLM_ROUTER_PRIMARY_URL"}),
        (["--secondary-url", "http://b/v1"], BASE_ENV),
        (["--primary-timeout-ms", "0"], BASE_ENV),
        (["--tokens-per-window", "0"], BASE_ENV),
        (["--max-response-bytes", "100"], BASE_ENV),
        (["--db-busy-timeout-ms", "-1"], BASE_ENV),
        ([], {**BASE_ENV, "LLM_ROUTER_DB_FAIL_FAST_MS": "-5"}),
        ([], {**BASE_ENV, "LLM_ROUTER_FAILOVER_ON_SERVER_ERROR": "maybe"}),
        ([], {**BASE_ENV, "LLM_ROUTER_TENANT_KEY_SHA256": "not-a-digest"}),
    ],
)
def test_invalid_configuration_is_a_usage_error(argv: list[str], env: dict[str, str], capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        load_settings(argv, env)
    assert excinfo.value.code == 2


def test_environment_round_trip_for_worker_processes(tmp_path: Path) -> None:
    env = {**BASE_ENV, "LLM_ROUTER_PRIMARY_API_KEY": "sk-p", "LLM_ROUTER_SECONDARY_API_KEY": "sk-s"}
    argv = [
        "--secondary-url",
        "http://b/v1",
        "--secondary-model",
        "backup",
        "--secondary-timeout-ms",
        "12500",
        "--db-path",
        str(tmp_path / "x.db"),
        "--no-failover-on-connection-error",
        "--no-failover-on-refused",
        "--max-response-bytes",
        "65536",
        "--db-busy-timeout-ms",
        "250",
        "--db-fail-fast-ms",
        "1500",
        "--workers",
        "4",
    ]
    settings = load_settings(argv, env)
    exported = settings.to_env()
    assert "key-one" not in str(exported)  # workers receive fingerprints, not raw tenant keys
    assert load_settings([], exported) == settings


def test_settings_validate_directly() -> None:
    with pytest.raises(ValueError, match="default_max_tokens"):
        Settings(
            primary_url="http://a",
            primary_model="m",
            tenant_key_fingerprints=frozenset({fingerprint_api_key("k")}),
            default_max_tokens=0,
        )
