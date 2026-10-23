# Session aware 路由

<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

`SESSION_AWARE` 根据客户端的 `X-Session-ID` 维护实例亲和性，适用于多轮对话和
agent 连续请求。客户端每轮仍发送完整请求；session ID 只是路由身份，不保存对话内容。

## 启用

在 `xllm_master_serving` 的原启动命令中增加：

```bash
--load_balance_policy=SESSION_AWARE \
--session_fallback_policy=RR \
--session_affinity_ttl_s=300 \
--session_affinity_max_entries=65536
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `session_fallback_policy` | `RR` | 首次请求、无 session ID 或绑定失效时的选点策略，支持 `RR`、`CAR` |
| `session_affinity_ttl_s` | 300 | 最后一个活跃请求结束后的空闲超时，必须大于 0 |
| `session_affinity_max_entries` | 65536 | 单个 service 进程的会话数上限，必须大于 0 |

选择 `CAR` 时，使用现有 CAR 负载和 KV 命中评分，配置见 [KV events](kv_events.md)。
同一个有效会话的后续请求直接使用绑定，不因评分或负载变化而迁移。

```bash
curl http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Session-ID: conversation-42' \
  -d '{"model":"my-model","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"stream":true}'
```

`/v1/chat/completions`、`/v1/completions` 和 `/v1/messages` 均读取该 header，
header 名称不区分大小写。空值或纯空白按无会话处理；不同 model 的同名 session 分开维护。

## 选点与生命周期

1. 首次选点建立临时占位，使同一 session 的并发首轮请求选择相同目标。
   引擎确认派发成功或已经返回正常结果后，才确认绑定；未派发成功的占位在请求结束时释放。
   CAR 等首次选点工作在全局会话锁外执行，完成后重新检查并采用已有占位，避免阻塞其他会话。
2. 非 PD 绑定 DEFAULT 实例；PD 分别检查 P、D 的目标身份。
   单侧实例退出、变为 SUSPECT、切换角色或进程代际变化时，重新选点并尽量保留另一侧。
   建立会话绑定要求选中的所有实例提供非空 `incarnation_id`，不再使用注册时间兜底；
   缺失时返回调度失败，无 session 的请求仍按原有 RR/CAR 策略处理。
   P/D 名称和进程 ID 统一保存在 `Routing` 中；绑定、准入和获取 Channel 时检查身份，
   避免使用同名新注册实例的连接。注册时间更新不会改变同一进程的会话绑定。
3. 活跃请求持有绑定，不参与空闲过期或容量淘汰。最后一个请求结束后开始空闲计时。
   清理在后续携带 session 的路由时执行；容量不足时先淘汰最久未使用的空闲会话。
   如果容量全部被活跃会话占用，新会话返回调度失败，已有会话继续服务。
4. 派发或请求处理错误使绑定失效，下一次请求可以重新选点；客户端取消只释放当前请求。
   请求持有代际编号，迟到的 ACK、失败或完成通知不能覆盖或删除后来的绑定。
   服务不会自动重放已经发给引擎的请求。

实现中，`SessionAwarePolicy` 组织已有绑定与 RR/CAR 选点，`SessionTable` 封装绑定、
TTL、LRU 和版本转换，`SessionLease` 负责请求持有与结束通知。实例身份的绑定、
准入校验和连接获取由 `InstanceMgr` 统一处理。

## 范围

- 会话表位于 service 内存中，重启后清空。多个 service 副本之间不共享绑定；
  如需跨请求保持亲和，应在上层入口将同一 session 固定到同一个 service 副本。
- 亲和性以实例为单位。引擎内部 DP 的选择仍由引擎负责；service 不能通过该策略固定 DP。
- Session ID 不保证 KV 驻留，也不保留 KV tensor。Host/Device 淘汰和实际复用仍由引擎管理。
- 会话绑定只影响服务层路由，现有引擎协议与启动参数保持一致。
