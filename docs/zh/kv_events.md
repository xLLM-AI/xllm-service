# xLLM KV events 自动订阅、分层索引与 CAR 路由

## 目录维护的内容

service 在后台订阅 xLLM 的 ZMQ KV events，维护按实例、DP、Device/Host
区分的前缀目录。目录只保存元数据，不存储或搬运 KV tensor。
`kv_directory` 管理订阅与发现，`kv_index` 维护分层前缀索引，
`kv_event_decoder` 解析事件。

CAR 结合目录中的 Device/Host 连续前缀命中和实例负载选点，内部查询接口与 CAR
共用该目录。KV 元数据来自引擎事件；etcd 负责实例发现、健康状态和负载同步，
不存储 KV 元数据。RR/SLO_AWARE 保持各自的选点逻辑。

## 构建和回归

ZeroMQ 由 service 自身的 vcpkg manifest 管理。
在已准备好项目依赖的开发环境执行：

```bash
mkdir -p build
cd build
cmake .. -DBUILD_TESTING=ON
make -j 8
ctest --output-on-failure -R '^(KV(Index|EventDecoder|Directory|CacheQuery)|CacheAwareRouting)'
```

回归使用真实 TCP XPUB/ROUTER 模拟 xLLM 发布端，不需要模型、NPU 或 etcd。
覆盖晚加入订阅、分层副本、连续前缀、DP 视图隔离、序号断档、快照与实时事件合并、
断线/进程重启、容量边界、自动发现的加入/退出/进程替换/DP 变化/端口冲突，
以及内部查询的鉴权和输入校验。

## 引擎和 service 配置

在当前已支持 KV events 的 xLLM 实例上，保留模型及推理参数，并设置：

```bash
--port=8000 \
--enable_prefix_cache=true \
--block_size=128 \
--kv_events_config='{"publisher":"zmq","endpoint":"tcp://*:18000","replay_endpoint":"tcp://*:38000","topic":"xllm-kv"}'
```

Host 事件仍要求引擎本身启用了本地 Host offload，且拷贝已经成功确认。
PD 模式先订阅 P；当前引擎的 D 侧预留块尚不能直接视为可用 KV。
现有引擎对 flat text prefix cache 的支持边界继续适用。

引擎按现有方式开启 service routing 并注册到同一 etcd namespace；可以由脚本、
Docker 或 K8s 启动。自动订阅与 service 是否负责拉起引擎相互独立。

参考 `xllm_service/examples/kv_events_discovery_config.json`，设置端口规则和事件参数。
在原有 service 启动命令中增加：

```bash
--kv_events_config_file=/path/to/kv_events_discovery_config.json
```

service 的 `kv_events_config_file` 是 JSON **文件路径**；引擎的 `kv_events_config`
是 JSON **字符串**。空文件路径关闭索引；此时 CAR 在启动日志中提示，并将 P 的命中比置零。
该参数用于 `xllm_master_serving` 入口，独立的示例 HTTP/RPC serving 程序不读取此配置。

service 的 SUB 连接引擎的 PUB 地址，replay 客户端连接引擎的恢复地址。

## 自动发现配置

默认示例采用以下发现规则，无须逐实例列地址：

```json
{
  "discovery": {
    "enabled": true,
    "block_size": 128,
    "topic": "xllm-kv",
    "event_port_offset": 10000,
    "replay_port_offset": 30000,
    "refresh_interval_ms": 1000
  }
}
```

后台读取 `InstanceState` 已由 etcd watch 维护的本地快照，不额外扫描 etcd，
也不在请求线程里执行发现或连接。使用注册的 HTTP 地址 `host:port` 推导：

```text
事件地址 = tcp://host:(port + event_port_offset + dp_rank)
恢复地址 = tcp://host:(port + replay_port_offset + dp_rank)
```

例如 `10.0.0.1:8000`、DP=2 对应事件端口 18000/18001，恢复端口 38000/38001。
这里使用业务 HTTP 端口，与引擎间 brpc 端口、KV 传输端口无关。
引擎仍须开启事件发布，并按同一规则配置端口；service 不会修改运行中引擎的参数。
按一个 service 集群中的实例使用同一模型和兼容 KV 配置维护目录，
无需额外配置模型分组。`block_size` 须与引擎实际发布的事件一致。

订阅符合条件的 xLLM DEFAULT/PREFILL 实例，以及当前承担 PREFILL 的 MIX 实例。
DECODE 实例和其他后端不纳入目录。沿用现有实例状态机：租约丢失的宽限期
保留订阅，进入 SUSPECT 或移出实例目录后删除订阅及查询项。
发现间隔默认 1000 ms，可设 50–60000 ms；实例状态通常在下一个发现周期更新，
下线判定所需的探测和宽限时间沿用现有实例状态机。

同一 `incarnation_id` 的注册刷新不会重建索引；身份、地址或 DP 拓扑变化会更新订阅。
新进程复用同一地址时，旧流退休，新流在取得快照前不提供命中。
旧版无进程身份时使用 `register_ts_ms` 辅助判断；两者都缺失时仍依赖连接重建和快照恢复。
端口溢出、跨实例/DP 的端口冲突会排除相关实例并记录错误；不会混用其他实例的事件。
`max_streams` 默认 128，范围 1–128；超过上限的实例按名称顺序整实例跳过，不截断其 DP。

## 可选资源和恢复参数

示例只保留 `discovery`。以下顶层参数都有默认值，通常可以省略；按需与
`discovery` 放在同一个 JSON 对象中配置，不属于引擎的 KV 容量参数。

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `max_streams` | 128 | 最多维护的订阅流数；一个实例的每个 DP 对应一条流 |
| `max_blocks_per_stream` | 262144 | 每条流的块元数据上限，包含为父链和 tokenless 事件保留的映射，用于控制目录内存 |
| `max_message_bytes` | 8388608（8 MiB） | 单条实时消息或快照回复的大小上限 |
| `max_buffered_bytes` | 16777216（16 MiB） | 获取快照期间暂存实时增量的预算，计算 wire bytes 和解码记录开销 |
| `replay_timeout_ms` | 2000 | 等待一次完整快照回复的超时时间；超时后流保持未同步并重试 |
| `resync_interval_ms` | 30000 | 两次周期性快照校准之间的间隔，用于发现静默丢失的尾部事件 |

## 静态配置

显式端点的 `streams` 配置仍供协议回归使用，由测试代码构造，examples 不再额外提供一份。
静态清单中的事件端口和恢复端口须与引擎实际启动参数一致。
`discovery` 与非空 `streams` 互斥，避免两种来源重复管理同一订阅。

每个 `streams` 元素描述一个实例的一个 DP，包含：

| 字段 | 说明 |
| --- | --- |
| `instance_name` | 人工指定的实例身份，建议与已有实例目录一致 |
| `dp_rank` | 此事件流的 DP，默认 0，必须与 batch 一致 |
| `block_size` | 每块 token 数，默认 128，必须与 Stored 事件一致 |
| `endpoint` | 必填、可连接的 TCP 事件地址，不能填写 `*`/`0.0.0.0` |
| `replay_endpoint` | 必填、可连接的 TCP 快照恢复地址 |
| `topic` | 与引擎发布的 topic 完全一致，默认空字符串 |
| `hash_seed` | 仅用于目录内部内容键，默认 1024，无须对齐引擎内部哈希 |

当前 xLLM 按基础端口加 `dp_rank` 发布各 DP；多 DP 需要分别配置每个实际端点，
并保证事件与 replay 端口范围互不重叠。配置会拒绝重复的实例/DP 和重复端点。

顶层资源上限和恢复参数见上表。配置启动时校验，修改后重启 service 生效。
每个 stream 使用独立的后台订阅线程及内存索引；连接失败会保持未同步并后台重试。

## 查询

service 正常就绪后，在其现有 HTTP 端口调用：

```bash
curl -sS http://127.0.0.1:8888/v1/internal/kv-cache/query \
  -H 'Content-Type: application/json' \
  -d '{"token_ids":[1,2,3,4]}'
```

如果 service 配置了 `internal_api_token`，同时带上 `X-Internal-Token` 请求头。
该接口沿用现有 HTTP 服务的就绪条件，不会为了查询索引提前开放业务端口。
输入为已经应用正确 chat template 的 token IDs，不在这里重复分词。
传入空 `token_ids` 可以查看配置流的同步状态和驻留块数量。

返回 `matches` 数组，按实例/DP 分开，主要字段为：

| 字段 | 说明 |
| --- | --- |
| `synchronized` | 是否具有有效的快照及连续增量；false 时命中为 0 |
| `sequence` | 最近纳入索引的事件序号 |
| `device_hit_tokens` | 连续可复用前缀中可以从 Device 获取的 token 数 |
| `host_only_hit_tokens` | 同一连续前缀中只能从 Host 获取的 token 数 |
| `reusable_prefix_tokens` | 上述两项之和，不重复计算两层共有的副本 |
| `device_blocks` / `host_blocks` | 此实例/DP 索引内各层驻留块数，允许副本重叠 |
| `error` | 未同步原因 |

查询只计算完整块，遇到同一实例/DP 在两层都没有的块就停止。
不会把多个 DP 的局部前缀拼接，也不会把同机不同实例的 Host 视为共享缓存。
返回的是异步目录的观测结果，后续是否实际复用仍由引擎当时的驻留状态决定。

## CAR 路由

在原有 service 命令中同时设置：

```bash
--load_balance_policy=CAR \
--car_prefill_lambda=0.5 \
--kv_events_config_file=/path/to/kv_events_discovery_config.json
```

每次请求从本地实例目录取得当前可调度的 P/DEFAULT、D 候选及负载快照，
查询本进程内的 KV 索引；请求路径不执行 ZMQ、HTTP 或 etcd 目录查询。
未收到首个负载报告的实例，以及负载非法或处于 SUSPECT 的实例，不参与 CAR。
主 service 优先使用刚收到的心跳负载，无须等待周期性 etcd 同步。
MIX 按当前角色进入 P/D 候选；只有 DEFAULT 可在没有 D 的情况下独立服务。

P/DEFAULT 采用成本评分，选择评分最小的实例：

```text
kv_hit_ratio = (Device 命中 tokens + host_weight × Host-only 命中 tokens) / 输入 tokens
队列代价 = waiting_requests_num / P 候选最大 waiting_requests_num
kv_hit_ratio 显式限制在 [0, 1]
P 实例评分 = lambda × (1 - kv_hit_ratio) + (1 - lambda) × 队列代价
```

`--car_prefill_lambda` 默认 0.5，允许范围 [0, 1]：越大越偏向缓存命中，
越小越偏向短队列。0 表示只比较队列，1 表示只比较缓存未命中成本。
P 的评分不再包含 Device KV 使用率；非 PD 的 DEFAULT 实例也使用该公式。

队列全为空时其代价为 0；命中比与队列比均采用浮点运算。
Device/Host 共有的块仅计入 Device；所有匹配均要求连续完整前缀。
`--car_host_hit_weight` 默认 0.5，允许范围 [0, 1]，代表 Host-only 复用相对
Device 的收益权重，属于可调启发式参数，并非测量得到的搬运耗时模型。
D 的选点顺序保持不变，同样写成取最小成本的形式：

```text
D 实例评分 = gpu_cache_usage_perc + waiting_requests_num / D 候选最大 waiting_requests_num
```

D 不使用缓存命中比，不将 D 的预留块视为可复用 P 前缀。
评分相同的候选轮转优先级，防止长期固定命中同一个实例。

当前服务请求仅能指定实例，不能指定实例内 DP。为此，多 DP 实例采用
**所有 DP 中最低的 kv_hit_ratio**，不取最佳 DP，也不累加跨 DP 命中。
任一 DP 缺失或未同步时，该实例的 kv_hit_ratio 为 0；仍可按上述成本公式被选中。
这一保守规则避免假设引擎一定选择命中最多的 DP。后续精确 DP 定向需要另行扩展
服务到引擎的路由协议。

CAR 同时比对实例 incarnation，旧版无 incarnation 时比较注册时间，
避免发现刷新前为同地址的新进程计入旧索引命中。生产 CAR 使用自动发现配置；
静态流模式用于协议测试/观测，缺少一致的进程身份时不提供路由命中收益。
订阅断线、事件断档、快照未完成等状态只影响缓存评分，不阻塞请求选点。

旧心跳 protobuf 的 `cache_event` 字段仅为兼容已有引擎保留并标记 deprecated，
service 不读取它，也不再读写 `XLLM:CACHE:`。引擎尚未关闭旧心跳上报时，
该字段会被忽略。各 service 副本各自订阅和恢复目录，
主从切换无需迁移旧 KV etcd 数据。原有 `--block_size` 不用于新索引，
实际事件块大小以发现配置中的 `block_size` 为准。

## 事件与恢复语义

消费当前 xLLM 的三帧协议：`topic | uint64 大端序号 | MessagePack`。
batch 为 `[timestamp, events, dp_rank]`，事件为带 `type` 的 map，支持
`BlockStored`、`BlockRemoved`、`AllBlocksCleared` 及 `gpu`/`cpu` 介质。
同一事件通道承载 Device/Host 元数据，索引支持纯文本前缀，不包含 LoRA 或多模态上下文。

目录用完整前缀上下文和 tokens 生成内部 XXH3-128 内容键，并保留
`engine_hash → 内容键` 映射供删除和 tokenless 副本事件使用。不能把线上折叠的
64 位 wire hash 直接当成旧 service 的 XXH3-128 键。
Device 删除只移除 Device 副本，Host 副本独立保留。

初次连接、断档、错误或重连后，通过 xLLM replay 端点请求**当前全量快照**：
DEALER 发送 `[空分隔帧, 8字节起始序号]`；接收
`[空分隔帧, 快照水位序号, MessagePack]`，再等待 UINT64_MAX/空 payload 结束标记。
这不是任意历史事件重放，也不是三帧 live topic envelope。
恢复时有界缓冲实时增量，按快照水位去重后一次性发布完整查询视图。
PUB 通道自身发送的 `AllBlocksCleared + Stored` 全量 batch 也能恢复目录。

序号断档、未知父链、消息/缓冲/索引超限、解析失败或订阅断线时，受影响流失效，
查询显式返回未同步，不继续提供旧命中。周期性快照校准也可发现静默丢失的尾部事件；
没有新增 KV 事件本身不等于实例断线。网络接收和恢复均不占用请求处理线程。
全量快照按 xLLM 的父块在先顺序处理：父块在两层都已淘汰的孤立后缀不计入命中，
也不会阻止其他独立前缀进入索引；增量中缺失父链则触发快照恢复。

## 真实联测结果

2026-10-21 在 ARM64 Ascend A3 环境使用 GLM-5.3 W8A8 六层视图完成 native RPC 2P2D 联测。
四个实例各使用一张 Ascend 卡，TP=1、DP=1；先启动独立 etcd，再启动 service 和引擎。
六层视图裁剪配置、量化描述和权重索引，12 个权重分片均为原模型的软链接，未复制权重。
服务通过 etcd 发现实例，按发现规则自动订阅两个 P 的 ZMQ 端点。

共完成 20 次真实请求，结果包括：

- A/B 前缀分别预热到 P0/P1；四轮交替重复共 8 次请求全部保持对应 P，
  每次引擎命中计数增加 384，客户端 `prompt_tokens_details.cached_tokens=384`。
- 9 次不同前缀请求制造 Device 淘汰。A 的索引变为 Device 命中 0、Host-only 命中 384；
  再次请求 A 仍选择 P0，实际命中 384，随后索引变为 Device 命中 384、Host-only 命中 0。
- 两个 D 的生成计数均有增加，覆盖两条 Decode 路径。

本轮设置 `max_cache_size=16777216`、`host_blocks_factor=4`、
`max_seqs_per_batch=1`、`max_tokens_per_batch=512`、`num_speculative_tokens=0`，
启用 Python 模型和 ACL Graph。就绪时四张卡分别占用 37051–37381 MiB HBM，
包含原有约 3 GiB 基线占用；该数字为就绪快照，不是显存峰值。
本轮验证六层模型的事件、分层复用和路由正确性，不评价减层模型的生成质量。
引擎退出仍需超时强制清理，本轮未改动该行为。
