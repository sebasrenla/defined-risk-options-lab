"""Secret loading via the OS keyring, with an opt-in environment fallback.

Credentials (broker OAuth secrets, refresh tokens, account numbers) must never
live in source or config. This loader reads them from the operating system's
**keyring** (Windows Credential Manager / macOS Keychain / Secret Service), with
an optional environment-variable fallback for local development.

Posture:

* By default, an environment variable is tried first (convenient for dev), then
  the keyring.
* Set ``OPTVOL_REQUIRE_KEYRING=true`` to disable the env fallback entirely and
  require the keyring, the right setting for any shared or production-like
  environment, so secrets can only come from the OS secret store.
* A missing *required* secret raises with a source hint (never printing the
  value); optional secrets return ``None``.

Nothing here contains a secret, only the *mechanism* for fetching one.

Provenance
----------
Ported from the program's ``secret_loader.py`` (keyring service and example
credential names genericized).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

try:
    from dotenv import load_dotenv
except Exception:  # noqa: BLE001 - dotenv is optional
    def load_dotenv(*args, **kwargs):  # type: ignore[no-redef]
        return False


DEFAULT_KEYRING_SERVICE = "optvol"
_TRUE_VALUES = {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def _load_env_once() -> None:
    load_dotenv(dotenv_path=".env")


def _as_nonempty(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@lru_cache(maxsize=1)
def _keyring_service() -> str:
    _load_env_once()
    return _as_nonempty(os.getenv("OPTVOL_KEYRING_SERVICE")) or DEFAULT_KEYRING_SERVICE


@lru_cache(maxsize=1)
def _require_keyring() -> bool:
    _load_env_once()
    raw = _as_nonempty(os.getenv("OPTVOL_REQUIRE_KEYRING")) or "false"
    return raw.lower() in _TRUE_VALUES


@lru_cache(maxsize=1)
def _keyring_module():
    try:
        import keyring  # type: ignore
    except Exception:  # noqa: BLE001 - keyring is optional
        return None
    return keyring


@lru_cache(maxsize=None)
def get_secret(env_name: str, keyring_key: str, required: bool = True) -> Optional[str]:
    """Fetch a secret from the environment (unless disabled) and/or the keyring.

    Parameters
    ----------
    env_name : str
        Environment-variable name to try when the env fallback is enabled.
    keyring_key : str
        Key to look up in the keyring service.
    required : bool
        If ``True``, raise when the secret cannot be found; otherwise return
        ``None``.
    """
    _load_env_once()
    require_keyring = _require_keyring()

    if not require_keyring:
        env_value = _as_nonempty(os.getenv(env_name))
        if env_value is not None:
            return env_value

    service = _keyring_service()
    keyring_mod = _keyring_module()
    if keyring_mod is not None:
        try:
            keyring_value = _as_nonempty(keyring_mod.get_password(service, keyring_key))
        except Exception as exc:  # noqa: BLE001
            if required:
                raise RuntimeError(
                    f"Unable to read secret '{env_name}' from keyring service '{service}'."
                ) from exc
            return None
        if keyring_value is not None:
            return keyring_value
    elif require_keyring and required:
        raise RuntimeError(
            "OPTVOL_REQUIRE_KEYRING=true but 'keyring' is not available in this Python environment."
        )

    if required:
        source_hint = (
            f"keyring:{service}/{keyring_key}"
            if require_keyring
            else f"env:{env_name} or keyring:{service}/{keyring_key}"
        )
        raise RuntimeError(f"Missing required secret '{env_name}' ({source_hint}).")
    return None


def get_broker_oauth_credentials(
    *, require_account: bool = False
) -> tuple[str, str, Optional[str]]:
    """Example: fetch a broker's OAuth client secret, refresh token, and account
    number using the pattern above. Env names and keyring keys are illustrative.
    """
    client_secret = get_secret("BROKER_CLIENT_SECRET", "broker_client_secret", required=True)
    refresh_token = get_secret("BROKER_REFRESH_TOKEN", "broker_refresh_token", required=True)
    account_number = get_secret("BROKER_ACCOUNT_NUMBER", "broker_account_number", required=require_account)
    return client_secret, refresh_token, account_number
