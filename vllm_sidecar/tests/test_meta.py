# Copyright 2025-2026 The xLLM Authors.
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
"""Unit tests for the InstanceMetaInfo / etcd-key Python mirror (no etcd needed).

Guards against drift from the C++ source of truth:
  * key layout / prefixes  -- instance_mgr.cpp:45 ETCD_KEYS_PREFIX_MAP
  * namespace normalization -- utils.cpp:105 normalize_etcd_namespace
  * required JSON fields    -- types.h:248 parse_from_json
"""

import pytest

from vllm_sidecar.meta import (
    InstanceType,
    build_instance_key,
    build_instance_meta,
    normalize_etcd_namespace,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("", ""),
        ("foo", "/foo/"),
        ("/foo", "/foo/"),
        ("foo/", "/foo/"),
        ("/a/b/", "/a/b/"),
        ("///", ""),
    ],
)
def test_normalize_namespace_matches_cpp(raw, expected):
    assert normalize_etcd_namespace(raw) == expected


def test_key_default_prefix_no_namespace():
    assert build_instance_key("127.0.0.1:18000") == "XLLM:DEFAULT:127.0.0.1:18000"


def test_key_with_namespace():
    key = build_instance_key("127.0.0.1:18000", InstanceType.DEFAULT, "ns")
    assert key == "/ns/XLLM:DEFAULT:127.0.0.1:18000"


def test_meta_has_required_fields_and_values():
    meta = build_instance_meta("127.0.0.1:18000", "vllm-abc123")
    # types.h parse_from_json requires these three (.at()).
    for required in ("name", "rpc_address", "type"):
        assert required in meta
    assert meta["name"] == "127.0.0.1:18000"
    assert meta["rpc_address"] == "127.0.0.1:18000"
    assert meta["type"] == 0  # DEFAULT, the routable type for a lone instance
    assert meta["backend_type"] == "vllm"
    assert meta["incarnation_id"] == "vllm-abc123"
    assert isinstance(meta["register_ts_ms"], int) and meta["register_ts_ms"] > 0


def test_addr_has_no_scheme():
    # init_brpc_channel prepends http://; the registered addr must be bare.
    meta = build_instance_meta("127.0.0.1:18000", "x")
    assert "://" not in meta["name"]
