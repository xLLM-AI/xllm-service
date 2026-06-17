# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm-service/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Minimal etcd v3 client over the gRPC-gateway JSON/HTTP API.

We deliberately avoid `python-etcd3` (grpcio/protobuf native deps are brittle
inside the shared GPU container). etcd exposes the full v3 KV/Lease API as JSON
over the same client port (default 2379), verified against etcd 3.4.30:

    POST /v3/lease/grant      {"TTL": <s>, "ID": 0}      -> {"ID","TTL"}
    POST /v3/kv/put           {"key","value","lease"}    (key/value base64)
    POST /v3/lease/keepalive  {"ID": <lease>}            -> {"result":{"TTL"}}
    POST /v3/lease/revoke     {"ID": <lease>}
    POST /v3/kv/range         {"key": <b64>}             -> {"kvs":[...]} | {}

Lease keep-alive refreshes the TTL WITHOUT touching the key, so it generates no
watch event -- the master's watcher only sees the initial PUT and the final
DELETE (on lease expiry/revoke). Only `requests` is required.
"""

import base64
import logging

import requests

logger = logging.getLogger("vllm_sidecar.etcd")


class EtcdError(RuntimeError):
    pass


def _b64(s: str) -> str:
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


class EtcdGatewayClient:
    """Lease-oriented etcd v3 client. Endpoints are tried in order per call."""

    def __init__(
        self,
        endpoints: str | list[str],
        username: str = "",
        password: str = "",
        timeout: float = 3.0,
    ) -> None:
        # `endpoints` may be a comma-separated string or a list of host:port.
        if isinstance(endpoints, str):
            endpoints = [e.strip() for e in endpoints.split(",") if e.strip()]
        # Keep an explicit scheme if present; only default to http:// otherwise,
        # so endpoints like "https://host:2379" are not turned into
        # "http://https://host:2379".
        self._bases = [
            e if e.startswith(("http://", "https://")) else "http://" + e
            for e in (endpoint.rstrip("/") for endpoint in endpoints)
        ]
        if not self._bases:
            raise ValueError("at least one etcd endpoint is required")
        self._timeout = timeout
        self._session = requests.Session()
        if username:
            self._authenticate(username, password)

    def _authenticate(self, username: str, password: str) -> None:
        resp = self._post(
            "/v3/auth/authenticate", {"name": username, "password": password}
        )
        token = resp.get("token")
        if not token:
            raise EtcdError("etcd authenticate returned no token")
        # etcd v3 gateway expects the token in the Authorization header.
        self._session.headers["Authorization"] = token

    def _post(self, path: str, body: dict) -> dict:
        last_err = None
        for base in self._bases:
            try:
                r = self._session.post(base + path, json=body, timeout=self._timeout)
                if r.status_code == 200:
                    try:
                        data = r.json()
                    except ValueError as e:
                        last_err = EtcdError(f"invalid JSON response from {path}: {e}")
                        continue
                    # Every v3 gateway endpoint returns a JSON object; coerce any
                    # other shape (null/list) to {} so callers can rely on .get().
                    return data if isinstance(data, dict) else {}
                last_err = EtcdError(f"{path} -> HTTP {r.status_code}: {r.text[:200]}")
            except requests.RequestException as e:  # connection/timeout
                last_err = e
        raise EtcdError(f"all etcd endpoints failed for {path}: {last_err}")

    # --- lease lifecycle ---------------------------------------------------

    def lease_grant(self, ttl_seconds: int) -> str:
        """Grant a lease; returns the lease id (int64 as a decimal string)."""
        resp = self._post("/v3/lease/grant", {"TTL": ttl_seconds, "ID": 0})
        lease_id = resp.get("ID")
        if not lease_id:
            raise EtcdError(f"lease grant returned no ID: {resp}")
        return lease_id

    def lease_keepalive(self, lease_id: str) -> int:
        """Refresh a lease once; returns the remaining TTL (0 == lease gone)."""
        resp = self._post("/v3/lease/keepalive", {"ID": lease_id})
        # "result" may be present but null (proto3 JSON for an empty message),
        # and "TTL" itself may be missing or null; treat all of these as 0.
        result = resp.get("result") or {}
        return int(result.get("TTL") or 0)

    def lease_revoke(self, lease_id: str) -> None:
        """Revoke a lease -> its key is deleted immediately."""
        self._post("/v3/lease/revoke", {"ID": lease_id})

    # --- kv ----------------------------------------------------------------

    def put(self, key: str, value: str, lease_id: str) -> None:
        self._post(
            "/v3/kv/put", {"key": _b64(key), "value": _b64(value), "lease": lease_id}
        )

    def get(self, key: str) -> str | None:
        """Return the string value at ``key`` or None if absent."""
        resp = self._post("/v3/kv/range", {"key": _b64(key)})
        kvs = resp.get("kvs")
        if not kvs:
            return None
        # proto3 JSON omits empty values, so "value" may be absent for an
        # empty string; treat that as "".
        return base64.b64decode(kvs[0].get("value", "")).decode("utf-8")
