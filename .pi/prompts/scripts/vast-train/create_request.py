#!/usr/bin/env python3
"""Single-shot Vast create transport with durable, credential-free diagnostics.

The upstream API uses PUT /api/v0/asks/<offer>/ with Bearer authentication.
Unlike the CLI, this adapter retains HTTP status and never retries a mutation.
Unknown results require exact-label reconciliation, not another create request.
"""

import argparse
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from workflow_common import IMAGE, now, write_json

MAX_RESPONSE_BYTES = 1024 * 1024
SAFE_ERROR_CODES = {"offer_unavailable", "offer_not_found", "insufficient_credit",
                    "invalid_args", "invalid_api_key", "too_many_requests"}


class NoRedirects(HTTPRedirectHandler):
    """Do not resend a mutation or forward credentials to a redirect target."""

    def redirect_request(self, req, fp, code, msg, headers, newurl, /) -> None:
        return None


def unique_fields(pairs: list[tuple[str, object]]) -> dict:
    fields = {}
    for key, value in pairs:
        if key in fields:
            raise ValueError("Duplicate response fields")
        fields[key] = value
    return fields


def classify_response(*, status: int, body: bytes) -> dict:
    result = {"http_status": status, "response_bytes": len(body),
              "response_sha256": hashlib.sha256(body).hexdigest(),
              "outcome": "unknown", "reason": "invalid_json"}
    try:
        response = json.loads(body, object_pairs_hook=unique_fields)
    except (ValueError, UnicodeError):
        return result
    if not isinstance(response, dict):
        result["reason"] = "invalid_response_shape"
        return result
    # Persist only allowlisted fields. Never persist free-form API messages,
    # headers, URLs or raw bodies: they can contain credentials/instance records.
    code = response.get("error")
    if isinstance(code, str) and code in SAFE_ERROR_CODES:
        result["provider_error_code"] = code
    identity = response.get("new_contract")
    if not 200 <= status < 300:
        result["reason"] = "http_error"
    elif response.get("success") is True and type(identity) is int and identity > 0:
        result.update(outcome="created", reason="confirmed_creation", instance_id=identity)
    elif response.get("success") is False and (identity is None or type(identity) is int and identity == 0):
        result.update(outcome="rejected", reason="explicit_rejection")
    else:
        result["reason"] = "unrecognized_response"
    return result


def send_create(*, offer_id: int, label: str) -> dict:
    key = os.environ.get("VAST_API_KEY", "")
    if not key or any(character.isspace() for character in key):
        return {"outcome": "unknown", "reason": "missing_credentials"}
    payload = {"client_id": "me", "image": IMAGE, "disk": 100, "env": {},
               "label": label, "runtype": "ssh_direc ssh_proxy", "cancel_unavail": True}
    request = Request(f"https://console.vast.ai/api/v0/asks/{offer_id}/",
                      data=json.dumps(payload).encode(), method="PUT",
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        response = build_opener(NoRedirects()).open(request, timeout=90)
    except HTTPError as error:
        response = error
    except (URLError, OSError, ValueError, HTTPException):
        # Never serialize exceptions: they may contain request URLs or headers.
        return {"outcome": "unknown", "reason": "transport_error"}
    try:
        with response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
            if len(body) > MAX_RESPONSE_BYTES:
                return {"outcome": "unknown", "reason": "response_too_large", "http_status": response.code}
            return classify_response(status=response.code, body=body)
    except (OSError, ValueError, HTTPException):
        return {"outcome": "unknown", "reason": "response_read_error"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offer-id", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.offer_id <= 0 or not re.fullmatch(r"toy-act-train-actv2-[a-zA-Z0-9_-]+", args.label):
        parser.error("Invalid offer ID or iteration-owned label")
    if args.result.exists():
        parser.error("Result already exists; do not resend a create request")
    result = {"version": 1, "offer_id": args.offer_id, "label": args.label,
              "recorded_at": now(), **send_create(offer_id=args.offer_id, label=args.label)}
    # Persist before stdout/parent state updates, so resume can recover the result.
    write_json(path=args.result, value=result)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
