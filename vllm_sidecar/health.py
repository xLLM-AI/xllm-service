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
"""vLLM health probing.

The lease only proves the sidecar is alive; it says nothing about vLLM. We gate
registration and lease refresh on vLLM's own `/health` endpoint so that a dead
vLLM (with the sidecar still running) does not leave a phantom instance routable.
"""

import logging

import requests

logger = logging.getLogger("vllm_sidecar.health")


class VllmHealthProbe:
    def __init__(self, vllm_url: str, timeout: float = 3.0) -> None:
        self._base = vllm_url.rstrip("/")
        self._timeout = timeout
        # Reuse one connection across the periodic probes (HTTP keep-alive) to
        # avoid piling up TIME_WAIT sockets on the host.
        self._session = requests.Session()

    def is_healthy(self) -> bool:
        """True iff vLLM `/health` returns 2xx within the timeout."""
        try:
            r = self._session.get(self._base + "/health", timeout=self._timeout)
            return 200 <= r.status_code < 300
        except requests.RequestException as e:
            logger.debug("vLLM health probe failed: %s", e)
            return False

    def served_model(self) -> str | None:
        """Best-effort first model id from `/v1/models`, for logging only."""
        try:
            r = self._session.get(self._base + "/v1/models", timeout=self._timeout)
            data = r.json().get("data", [])
            return data[0]["id"] if data else None
        except (requests.RequestException, ValueError, KeyError, IndexError):
            return None
