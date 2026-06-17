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
"""vLLM sidecar entrypoint: health-gated etcd lease registration loop.

Run alongside a vLLM server:

    python -m vllm_sidecar.sidecar \
        --etcd-endpoints 127.0.0.1:2379 \
        --vllm-url http://127.0.0.1:18000 \
        --register-addr 127.0.0.1:18000

See README.md for the full flag list and the lifecycle contract.
"""

import argparse
import json
import logging
import os
import signal
import threading
import uuid
from types import FrameType

from .etcd_registry import EtcdGatewayClient, EtcdError
from .health import VllmHealthProbe
from .meta import InstanceType, build_instance_key, build_instance_meta

logger = logging.getLogger("vllm_sidecar")


class Sidecar:
    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._etcd = EtcdGatewayClient(
            args.etcd_endpoints,
            username=args.etcd_username,
            password=args.etcd_password,
            timeout=args.etcd_timeout,
        )
        self._health = VllmHealthProbe(args.vllm_url, timeout=args.health_timeout)
        self._instance_type = InstanceType[args.instance_type]
        self._key = build_instance_key(
            args.register_addr, self._instance_type, args.etcd_namespace
        )
        self._stop = threading.Event()
        self._lease_id = None
        self._incarnation_id = None

    # --- registration primitives ------------------------------------------

    def _new_incarnation(self) -> str:
        token = uuid.uuid4().hex[:12]
        return (
            f"{self._args.instance_name}-{token}" if self._args.instance_name else token
        )

    def _register(self) -> bool:
        """Grant a fresh lease and put the instance key. Returns success."""
        try:
            incarnation = self._new_incarnation()
            lease_id = self._etcd.lease_grant(self._args.lease_ttl)
            meta = build_instance_meta(
                self._args.register_addr,
                incarnation,
                self._instance_type,
                self._args.backend_type,
            )
            self._etcd.put(self._key, json.dumps(meta), lease_id)
            self._lease_id = lease_id
            self._incarnation_id = incarnation
            logger.info(
                "registered %s (incarnation=%s, lease=%s, ttl=%ds)",
                self._key,
                incarnation,
                lease_id,
                self._args.lease_ttl,
            )
            return True
        except EtcdError as e:
            logger.warning("register failed, will retry: %s", e)
            self._lease_id = None
            return False

    def _deregister(self) -> None:
        """Revoke the lease so the master removes the instance immediately."""
        if self._lease_id is None:
            return
        try:
            self._etcd.lease_revoke(self._lease_id)
            logger.info("deregistered %s (lease=%s revoked)", self._key, self._lease_id)
        except EtcdError as e:
            # On failure the lease still expires within ttl; not fatal.
            logger.warning(
                "revoke failed (lease expires in <=%ds): %s", self._args.lease_ttl, e
            )
        finally:
            self._lease_id = None
            self._incarnation_id = None

    @property
    def _registered(self) -> bool:
        return self._lease_id is not None

    # --- main loop ---------------------------------------------------------

    def run(self) -> None:
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)

        self._wait_until_healthy()
        if self._stop.is_set():
            return
        model = self._health.served_model()
        logger.info("vLLM healthy (model=%s), registering ...", model or "?")
        self._register()

        fail = 0
        while not self._stop.wait(self._args.keepalive_interval):
            if self._health.is_healthy():
                fail = 0
                self._keepalive_or_reregister()
            else:
                fail += 1
                logger.warning(
                    "vLLM health probe failed (%d/%d)",
                    fail,
                    self._args.health_fail_threshold,
                )
                if self._registered and fail >= self._args.health_fail_threshold:
                    logger.error("vLLM unhealthy, deregistering")
                    self._deregister()

        self._deregister()
        logger.info("sidecar stopped")

    def _wait_until_healthy(self) -> None:
        backoff = 1.0
        while not self._stop.is_set() and not self._health.is_healthy():
            logger.info(
                "waiting for vLLM at %s to become healthy ...", self._args.vllm_url
            )
            self._stop.wait(backoff)
            backoff = min(backoff * 2, self._args.lease_ttl)

    def _keepalive_or_reregister(self) -> None:
        if not self._registered:
            self._register()  # recovered after a previous deregister
            return
        try:
            ttl = self._etcd.lease_keepalive(self._lease_id)
            if ttl <= 0:
                logger.warning("lease %s lost (ttl=0), re-registering", self._lease_id)
                self._lease_id = None
                self._register()
        except EtcdError as e:
            logger.warning("keepalive failed, re-registering: %s", e)
            self._lease_id = None
            self._register()

    def _on_signal(self, signum: int, _frame: FrameType | None) -> None:
        logger.info("received signal %d, shutting down", signum)
        self._stop.set()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vllm_sidecar",
        description="Auto-register a vLLM instance into xllm-service via etcd.",
    )
    p.add_argument(
        "--etcd-endpoints",
        default=os.environ.get("ETCD_ENDPOINTS", "127.0.0.1:2379"),
        help="comma-separated host:port list (default 127.0.0.1:2379)",
    )
    p.add_argument(
        "--etcd-namespace",
        default=os.environ.get("ETCD_NAMESPACE", ""),
        help="must match master's --etcd_namespace (default empty)",
    )
    p.add_argument("--etcd-username", default=os.environ.get("ETCD_USERNAME", ""))
    p.add_argument("--etcd-password", default=os.environ.get("ETCD_PASSWORD", ""))
    p.add_argument("--etcd-timeout", type=float, default=3.0)
    p.add_argument(
        "--vllm-url",
        default="http://127.0.0.1:18000",
        help="base URL of the local vLLM server",
    )
    p.add_argument(
        "--register-addr",
        default=None,
        help="host:port the master uses to reach vLLM, NO scheme "
        "(default: derived from --vllm-url)",
    )
    p.add_argument("--backend-type", default="vllm")
    p.add_argument(
        "--instance-type",
        default="DEFAULT",
        choices=[t.name for t in InstanceType],
        help="single vLLM instance must be DEFAULT to be routable",
    )
    p.add_argument(
        "--instance-name",
        default="vllm",
        help="human-readable prefix for the incarnation id / logs",
    )
    p.add_argument(
        "--lease-ttl",
        type=int,
        default=6,
        help="etcd lease TTL in seconds (~2x keepalive interval)",
    )
    p.add_argument(
        "--keepalive-interval",
        type=float,
        default=2.0,
        help="seconds between health probe + lease refresh",
    )
    p.add_argument("--health-timeout", type=float, default=3.0)
    p.add_argument(
        "--health-fail-threshold",
        type=int,
        default=3,
        help="consecutive failed probes before deregistering",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def _derive_addr(vllm_url: str) -> str:
    # strip scheme and any trailing path -> host:port
    no_scheme = vllm_url.split("://", 1)[-1]
    return no_scheme.split("/", 1)[0]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.register_addr:
        args.register_addr = _derive_addr(args.vllm_url)
    logger.info(
        "sidecar starting: register %s as backend_type=%s type=%s",
        args.register_addr,
        args.backend_type,
        args.instance_type,
    )
    Sidecar(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
