# Copyright 2025 The xLLM Authors. All Rights Reserved.
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
"""Runtime unit tests for the vLLM sidecar without live etcd/vLLM."""

import argparse
import base64
import json

import pytest
import requests

from xllm_service.vllm_sidecar import sidecar as sidecar_mod
from xllm_service.vllm_sidecar.etcd_registry import EtcdError, EtcdGatewayClient
from xllm_service.vllm_sidecar.health import VllmHealthProbe
from xllm_service.vllm_sidecar.meta import InstanceType


class _Response:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.headers = {}

    def post(self, url, json, timeout):
        self.calls.append((url, json, timeout))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _Etcd:
    def __init__(self, *_args, **_kwargs):
        self.puts = []
        self.keepalives = []
        self.revoked = []
        self.keepalive_ttl = 5
        self.fail_grant = False
        self.fail_keepalive = False

    def lease_grant(self, ttl_seconds):
        if self.fail_grant:
            raise EtcdError("grant failed")
        return "lease-1"

    def put(self, key, value, lease_id):
        self.puts.append((key, json.loads(value), lease_id))

    def lease_keepalive(self, lease_id):
        if self.fail_keepalive:
            raise EtcdError("keepalive failed")
        self.keepalives.append(lease_id)
        return self.keepalive_ttl

    def lease_revoke(self, lease_id):
        self.revoked.append(lease_id)


class _Health:
    def is_healthy(self):
        return True

    def served_model(self):
        return "demo-model"


def _args(**kwargs):
    values = dict(
        etcd_endpoints="127.0.0.1:2379",
        etcd_username="",
        etcd_password="",
        etcd_timeout=1.0,
        vllm_url="http://127.0.0.1:18000",
        register_addr="127.0.0.1:18000",
        backend_type="vllm",
        instance_type="DEFAULT",
        instance_name="vllm",
        etcd_namespace="",
        lease_ttl=6,
        keepalive_interval=2.0,
        health_timeout=1.0,
        health_fail_threshold=3,
        xllm_service_url="http://127.0.0.1:9998",
        internal_token="",
        heartbeat_interval=3.0,
        metrics_url="http://127.0.0.1:18000/metrics",
        log_level="INFO",
    )
    values.update(kwargs)
    return argparse.Namespace(**values)


def _install_fakes(monkeypatch):
    etcd = _Etcd()
    monkeypatch.setattr(sidecar_mod, "EtcdGatewayClient", lambda *a, **k: etcd)
    monkeypatch.setattr(sidecar_mod, "VllmHealthProbe", lambda *a, **k: _Health())
    return etcd


def test_health_probe_success_failure_and_model(monkeypatch):
    responses = iter(
        [
            _Response(204),
            _Response(503),
            requests.Timeout("timeout"),
            _Response(200, {"data": [{"id": "model-a"}]}),
            _Response(200, {"data": []}),
        ]
    )

    def fake_get(*_args, **_kwargs):
        item = next(responses)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(requests, "get", fake_get)
    probe = VllmHealthProbe("http://vllm/")
    assert probe.is_healthy()
    assert not probe.is_healthy()
    assert not probe.is_healthy()
    assert probe.served_model() == "model-a"
    assert probe.served_model() is None


def test_etcd_gateway_lease_kv_auth_and_errors(monkeypatch):
    value = base64.b64encode(b"payload").decode("ascii")
    session = _Session(
        [
            _Response(payload={"token": "tok"}),
            _Response(payload={"ID": "lease-1"}),
            _Response(payload={"result": {"TTL": "4"}}),
            _Response(),
            _Response(),
            _Response(payload={"kvs": [{"value": value}]}),
            _Response(payload={}),
        ]
    )
    monkeypatch.setattr(requests, "Session", lambda: session)
    client = EtcdGatewayClient("127.0.0.1:2379", username="u", password="p")

    assert client._session.headers["Authorization"] == "tok"
    assert client.lease_grant(6) == "lease-1"
    assert client.lease_keepalive("lease-1") == 4
    client.lease_revoke("lease-1")
    client.put("key", "payload", "lease-1")
    assert client.get("key") == "payload"
    assert client.get("missing") is None

    monkeypatch.setattr(
        requests, "Session", lambda: _Session([_Response(500, text="bad")])
    )
    with pytest.raises(EtcdError):
        EtcdGatewayClient("127.0.0.1:2379").lease_grant(6)
    with pytest.raises(ValueError):
        EtcdGatewayClient("")


def test_sidecar_registration_keepalive_and_deregister(monkeypatch):
    etcd = _install_fakes(monkeypatch)
    sc = sidecar_mod.Sidecar(_args())

    assert sc._register()
    assert sc._registered
    key, meta, lease_id = etcd.puts[0]
    assert key == "XLLM:DEFAULT:127.0.0.1:18000"
    assert meta["backend_type"] == "vllm"
    assert lease_id == "lease-1"

    sc._keepalive_or_reregister()
    assert etcd.keepalives == ["lease-1"]

    sc._deregister()
    assert not sc._registered
    assert etcd.revoked == ["lease-1"]


def test_sidecar_reregisters_on_lost_or_failed_lease(monkeypatch):
    etcd = _install_fakes(monkeypatch)
    sc = sidecar_mod.Sidecar(_args())
    assert sc._register()

    etcd.keepalive_ttl = 0
    sc._keepalive_or_reregister()
    assert len(etcd.puts) == 2

    etcd.fail_keepalive = True
    sc._keepalive_or_reregister()
    assert len(etcd.puts) == 3

    etcd.fail_grant = True
    sc._lease_id = None
    assert not sc._register()
    assert not sc._registered


def test_sidecar_parser_and_main(monkeypatch):
    ran = []

    class FakeSidecar:
        def __init__(self, args):
            self.args = args

        def run(self):
            ran.append((self.args.register_addr, self.args.log_level))

    monkeypatch.setattr(sidecar_mod, "Sidecar", FakeSidecar)
    assert sidecar_mod._derive_addr("http://host:18000/v1/models") == "host:18000"
    assert (
        sidecar_mod.main(["--vllm-url", "http://host:18000", "--log-level", "DEBUG"])
        == 0
    )
    assert ran == [("host:18000", "DEBUG")]

    args = sidecar_mod.build_parser().parse_args(["--instance-type", "DEFAULT"])
    assert args.instance_type == InstanceType.DEFAULT.name
