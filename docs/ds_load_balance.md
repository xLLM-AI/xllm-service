# DeepSeek load balancing

The `ds_load_balance` policy adapts the instance-level Prefill/Decode routing
described in DeepSeek's deployment article. Select
`--load_balance_policy=ds_load_balance` to route each generation request to
the healthy prefill engine with the smallest assigned prefill workload:

```text
score(P) = sum(compute estimates of unfinished prefill requests assigned to P)
selected = argmin(score(P))
score(selected) += predicted_compute(selected, input_tokens, reusable_prefix_tokens)
```

This is a time-based policy without CAR weights, SLO thresholds, or P/D role
changes. The new request's estimate is reserved after selection; it does not
participate in the current choice. Equal loads rotate. A DEFAULT engine can serve
alone. Requests to `/v1/models` use ordinary instance discovery without reserving
compute.

Decode adapts the within-group rule in
[DualPath, section 6.1](https://arxiv.org/html/2602.21548v1#S6.SS1)
to the current Service's one-request-at-a-time admission. For healthy D engines,
let `T(D)` be assigned prompt plus generated tokens, `S(D)` unfinished request
count, and `N` the incoming prompt length. Set
`Z = 1.05 * (sum(T(D)) + N) / number_of_healthy_D`.
Among engines with `T(D)+N <= Z`, choose minimum `(S(D), T(D))`.
If that set is empty, choose minimum `(T(D), S(D))`. Exact ties rotate.
This favors low concurrency for small additions that fit the token threshold,
and low KV load when all candidates would exceed it. The Service has no exact
per-engine free-KV capacity telemetry, so this is a load heuristic; engine-side
HBM admission remains responsible for capacity. It is not DualPath's group
queues or HBM-aware batch admission.

Selection, incarnation binding, and request reservations are atomic. The P
reservation is released on the first handoff/token notification, or on
cancellation/final completion before handoff. Decode accounting lasts until
completion/cancellation. Generation accounting uses actual token deltas for
streaming output, cumulative counts for non-streaming output, and cumulative
usage when supplied. MTP multi-token output is counted in full, and empty or
repeated cumulative frames add no tokens. Streaming delta frames without usage
have no sequence number in the current protocol and cannot be deduplicated.
A Store query runs outside topology/metrics locks.
Hints from an engine's previous incarnation are discarded.

## Time profiles

The existing `TimePredictor` fits a quadratic `F(N) = a + b*N + c*N²` in
milliseconds. Its new routing method fits nonnegative coefficients using
normalized lengths, preserving the existing SLO methods. This keeps short-input
and high-hit estimates positive even with noisy/concave profile samples.
The compute estimate with a reusable prefix H is
`F(N) - F(H) + F(0)`; H=0 uses F(N). This retains the suffix attention cost
against the cached prefix and the fixed per-request cost. It is an approximation,
not a separate fit of Host/Store transfer latency.

An engine needs at least three distinct, positive token lengths with positive,
finite measured times. Missing/invalid profiles are ineligible, rather than
being interpreted as zero cost. Fixed PREFILL instances may not run the
engine's MIX-only automatic profiling. Supply measurements using
`--prefill_time_profile_file=/path/prefill_profiles.json`:

```json
{
  "prefill-0": [[128, 12.0], [1024, 45.0], [4096, 180.0]],
  "prefill-1": [[128, 13.0], [1024, 47.0], [4096, 190.0]]
}
```

These numbers are illustrative; replace them with uncached prefill measurements
for the actual model, hardware and engine options. Names must equal registered
instance names. A configured profile overrides the engine's registration
profile and persists across re-registration. Engines without an override use
their existing registered TTFT profile. Restart Service after editing profiles.

## Mooncake metadata support

The optional dependency uses Mooncake's official
`WrappedMasterService::BatchExistKey` RPC declaration and its YLT TCP client.
It compiles only the metadata adapter as C++20; Service remains C++17. It does
not allocate a Store segment, register Host memory, instantiate RealClient,
or move KV data. Existing engines continue performing prefetch/offload.

Mooncake is a pinned Git submodule at `third_party/Mooncake`, matching xLLM's
third-party dependency layout. `third_party/mooncake.cmake` provides the
`mooncake::metadata` target, keeping SDK headers and C++20 private to the
adapter. This does not add or modify vcpkg dependencies.

Initialize the submodule and use a compatible running Store revision. Complete
Boost development headers (including Lockfree and Parameter) must be available
as a system/preinstalled dependency. If outside the default search paths, use
the standard `CMAKE_INCLUDE_PATH` to locate their include directory:

```bash
git submodule update --init --recursive third_party/Mooncake
cmake -S . -B build \
  -DBUILD_TESTING=ON \
  -DUSE_MOONCAKE_PREFIX_LOOKUP=ON
cmake --build build -j8
```

When building against a paired engine checkout with newer public schemas, also
set the existing `-DXLLM_PROTO_DIR=/path/to/paired-engine/xllm/proto` option.

The normal build defaults to `USE_MOONCAKE_PREFIX_LOOKUP=OFF` and has no
Mooncake dependency. ds_load_balance without a Mooncake configuration estimates
uncached prefill. Supplying a Mooncake configuration to a build without the
adapter fails at startup.

Example `mooncake_prefix.json`:

```json
{
  "master_address": "192.0.2.10:50051",
  "tenant_id": "default",
  "timeout_ms": 20,
  "max_keys": 4096,
  "layouts": [{
    "instances": ["prefill-0", "prefill-1"],
    "block_size": 128,
    "hash_seed": 1024,
    "hash_lookahead": 0,
    "key_prefixes_hex": ["<complete binary StoreEntry.key_prefix as lowercase hex>"]
  }]
}
```

The prefix placeholder must be replaced before launch. Each layout lists all
required cache components and TP rank keys, excluding only the final 16-byte
token hash. MLA uses its shared rank-0 Store key. Distinct engine KV layouts
need separate entries. Instance names must not occur in multiple entries.

The existing engine registration does **not** export Store key namespaces.
Therefore this first implementation requires explicit prefixes matching
`KVCacheStore::build_key_prefix` and `build_schema_hash` in the engine:
`xllm-kv-v3:`, length-prefixed model ID and key component, parallelism,
block type, optional TP rank, and the raw 16-byte schema hash. Do not derive
prefixes solely from the HTTP model alias or use the KV event index's canonical
hash encoding. Export the exact `StoreEntry.key_prefix` bytes from the engine
configuration/debugger, or construct them from its exact tensor schema.
Re-export/reconfigure after changing model identity, TP size, tensor shapes,
dtypes, cache domains or hash settings. Operators must include every required
shard/component; the router cannot discover omitted components.

Supported hashes are the engine's little-endian causal text XXH3-128 chain,
including `hash_lookahead=1` for MTP text hashing. Multimodal/stateful or other
hash protocols must leave Store lookup disabled. Only full contiguous blocks
whose **every** component/shard exists count as hits. At least one input token
is left for logits. `max_keys` caps total query keys and conservatively truncates
the prefix horizon, dividing the budget between layouts.

One TCP connection and one batch RPC share the configured timeout budget,
without retries. The deadline governs network IO, not CPU scheduling or local
hash construction. Metadata errors/timeouts/response-size mismatches explicitly
produce a zero-hit routing estimate. Engine-side reload remains authoritative:
a routing hint does not reserve KV until engine GET. This SDK's Exist API
refreshes Mooncake's configured read leases for matched objects; it is not a
side-effect-free probe.

## Launch and observe

Keep the usual etcd and engine settings; add the new policy/configuration flags:

```bash
./build/xllm_service/xllm_master_serving \
  --tokenizer_path=/mnt/models/your-model \
  --etcd_addr=127.0.0.1:2379 \
  --load_balance_policy=ds_load_balance \
  --prefill_time_profile_file=/path/prefill_profiles.json \
  --mooncake_prefix_config_file=/path/mooncake_prefix.json \
  --v=1
```

VLOG(1) records selected P/D, Store hit tokens, estimated compute milliseconds
and already-queued compute milliseconds, plus selected D request/token counts.
Run the focused tests after building:

```bash
ctest --test-dir build --output-on-failure \
  -R 'DsLoadBalanceTest|PrefillTimePredictorTest|MooncakePrefixTest|RequestMetricsTest|RequestAdmissionTest'
```

## Scope of the estimate

The queue ledger belongs to this Service process. Multiple independent routers
do not share this ledger. Running requests retain their whole reservation until
handoff; this version does not subtract elapsed execution time or model engine
batching/chunking/overlap. Store/SSD promotion, Host/NPU copies, network
contention and decode admission limits are not fitted. HBM/Host KV event hits
are not used by this policy. It provides a routing estimate, without promising
large-concurrency throughput or lower TTFT in every workload.

The routing design follows
[the DeepSeek deployment article](https://zhuanlan.zhihu.com/p/2095560295930508953).
The current implementation covers instance-level
minimum-assigned-prefill-time routing and a token-threshold Decode heuristic.
It does not implement the article's
dynamic length buckets, cross-group quota pulling, per-DP-rank assignment, or
per-step chunk sizing from a time budget. The current routing protocol carries
P/D instance names, without a target DP rank. Rank assignment and chunk budgets
need engine scheduler/protocol changes.

With q=N-H and c=H, the current prefix estimate is
`F(q+c)-F(c)+F(0) = a + b*q + d*q² + 2*d*q*c`.
It is a constrained approximation obtained from the existing univariate fit,
not an independently profiled two-variable `cost(q, c)` model.

## Functional validation

The focused policy, predictor, prefix lookup, request metrics and admission
tests passed all 27 cases with the Mooncake adapter enabled. An isolated real
Mooncake Master also verified metadata hits, missing keys and bounded RPC
failure handling.

A Qwen3-8B BF16 run used two TP1 Prefill engines and two TP1 Decode engines
on Ascend A3, with the Python model implementation, ACL graph and task pipeline.
Profiles were measured at 128, 512 and 1536 uncached input tokens per P. These
samples calibrated the functional test; they are not a production load model.
Mooncake used a 2 GiB DRAM segment and a 2 GiB bucket-backend disk quota.

| Check | Observed result |
| --- | --- |
| Cross-P multi-turn reuse | First input: 1905 tokens; first output: 512 tokens. The client appended returned text to the history, and the second turn used the other P. Both the router and engine reported 2304 cached tokens, including 399 tokens generated by the previous D. |
| Concurrent routing | Four concurrent requests completed; both P and both D engines were selected, with existing reservations visible to later admissions. |
| Cancellation | One stream was deliberately cancelled. Four subsequent probes completed, and P compute reservations and D request/token counters returned to zero. |
| Disk offload | Store metrics and cache files confirmed disk writes. This run did not force a cold SSD read. |

Ten requests using this policy completed, in addition to the deliberately cancelled
request. Second-turn client TTFT in the multi-turn case was about 669 ms. There
was no Mooncake-disabled comparison, and these results do not establish a TTFT
improvement or large-concurrency performance. The first turn was non-streaming;
its full-response latency must not be interpreted as TTFT.

An initial engine request combining `min_tokens` and `ignore_eos` failed task
pipeline sampling validation on D, before switching to this policy. Final test
requests used `max_tokens` and `ignore_eos`, omitting `min_tokens`. This engine
limitation is not fixed by the routing change. A private CPUinfo snapshot mount
was also needed to work around a host-specific `/proc/cpuinfo` read stall during
model initialization.
