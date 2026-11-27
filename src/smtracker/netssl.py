"""HTTPS from the packaged program: the list of certificates a website is checked against.

WHY (27 Nov 2026, the segmentation pack on a lab Mac: "CERTIFICATE_VERIFY_FAILED ... unable to get local issuer
certificate"). Python's OpenSSL looks for its certificate list where the BUILD machine's Python kept it; inside the
packaged app on another Mac that file is not there, and OpenSSL never reads macOS's Keychain. So every HTTPS download
failed there: the segmentation pack, Cellpose-SAM's model, the updater's question to GitHub. Windows is not affected:
Python reads the Windows certificate store.

`use_bundled_certificates`, called once at start (`app.main`), points OpenSSL at a list where nothing names one
(``SSL_CERT_FILE`` / ``SSL_CERT_DIR``) and the default one is missing: certifi's (Mozilla's list, shipped in the
bundle), else macOS's own ``/etc/ssl/cert.pem``. Through the environment, so every HTTPS user in the process follows,
Cellpose's own downloader included. A list somebody set, or a default that exists, is left alone. Never raises: a
program that cannot set this still opens, and a download then says why it failed.
"""
from __future__ import annotations

import os
import ssl
import sys

__all__ = ["use_bundled_certificates", "MAC_SYSTEM_LIST"]

#: macOS's own certificate list (LibreSSL's), present on every macOS: the fallback where certifi is missing.
MAC_SYSTEM_LIST = "/etc/ssl/cert.pem"


def _certifi_list() -> str | None:
    try:
        import certifi

        path = certifi.where()
    except Exception:                                      # noqa: BLE001 - no certifi: the next choice
        return None
    return path if os.path.isfile(path) else None


def use_bundled_certificates(environ=None, *, platform: str | None = None, defaults=None) -> str | None:
    """Point OpenSSL at a certificate list where none is set and the default is missing (not on Windows). Returns the
    list now used, or None where nothing was changed."""
    environ = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    if platform.startswith("win"):
        return None
    if environ.get("SSL_CERT_FILE") or environ.get("SSL_CERT_DIR"):
        return None
    try:
        paths = defaults or ssl.get_default_verify_paths()
    except Exception:                                      # noqa: BLE001
        return None
    if paths.cafile or paths.capath:                       # OpenSSL's default list exists here: left alone
        return None
    candidates = [_certifi_list()] + ([MAC_SYSTEM_LIST] if platform == "darwin" else [])
    chosen = next((c for c in candidates if c and os.path.isfile(c)), None)
    if chosen:
        environ["SSL_CERT_FILE"] = chosen
    return chosen
