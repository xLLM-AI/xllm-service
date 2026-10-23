# xLLM Service 代码架构

本次重构保留现有 CLI、HTTP/RPC 协议、etcd 注册格式及 CAR 评分公式，把原
`scheduler/` 按职责拆开。引擎仍由现有部署方式启动；本次没有新增 JSON 进程托管、
自动扩缩容或 Kubernetes 控制器。

## 模块与依赖

| 模块 | 主要文件 | 职责 |
| --- | --- | --- |
| 入口 | `main.cpp` | 读取配置、安装信号处理、运行应用 |
| 应用装配 | `service_app.h/.cpp` | 创建依赖，按顺序启动与停止发现、控制、请求和网络服务 |
| 配置 | `config/service_config.h`、`config_loader.h/.cpp` | 配置值与现有 gflags 的转换；业务模块不直接读取全局 flags |
| 请求路由 | `router/router.h/.cpp` | 预处理请求、查询本地 KV 目录、调用策略并绑定实例 incarnation |
| 策略 | `router/policies/`、`routing_policy.h`、`routing_types.h` | RR、CAR、SLO_AWARE 根据一致的状态快照返回路由决策，不访问 etcd 或引擎 |
| 实例状态 | `cluster/instance_state.h/.cpp` | 本地实例集合、角色索引、健康状态和引擎上报负载；提供状态更新接口和一致快照 |
| 调度状态 | `router/routing_state.h/.cpp` | 服务侧请求/token 预留、时间预测、精确代际绑定及 SLO 调度状态提交 |
| 实例控制 | `cluster/instance_controller.h/.cpp` | 注册、替换、健康检查、租约丢失宽限期、下线清理以及 P/D Link/Unlink |
| 引擎客户端 | `cluster/instance_client.h/.cpp` | 按实例代际缓存 Channel；执行健康探测、P/D Link/Unlink 和推理请求提交 |
| 实例发现 | `coordinator/instance_discovery.h/.cpp` | etcd 初始状态和 watch 转换为类型化的实例/负载事件；异步串行消费 |
| 服务成员 | `coordinator/service_membership.h/.cpp` | 服务自身租约、主备选举和主服务负载发布 |
| 请求生命周期 | `router/request_manager.h/.cpp` | 准入、请求存储、完成/取消、指标释放及停机取消 |
| 预处理 | `request/request_preprocessor.h/.cpp`、`chat_message.h` | 模板渲染和分词；请求输入类型独立于 Jinja 实现 |
| 有序输出 | `request/output_dispatcher.h/.cpp` | 按请求 ID 固定映射到串行队列，保证结果和取消事件的顺序 |
| KV 索引 | `kv_conductor/` | 后台 ZMQ 订阅、replay/快照恢复、Host/Device 前缀索引；路由只读取本地目录 |
| HTTP | `http_service/` | 请求协议解析、OpenAI/Anthropic 响应编码；`response_handler` 构造输出回调 |
| RPC | `rpc_service/` | 心跳、实例信息和批量 Generations；消费引擎已解析的结构化结果 |

`InstanceState` 保存引擎注册与心跳提供的事实，`RoutingState` 保存服务本地的调度预留与预测器。
快照中的注册数据通过只读共享指针复用，路由和心跳不会重复拷贝 profiling 数组。
CAR/RR 在获取快照后计算；SLO 在路由侧锁内完成选路与预留，不占用实例状态锁执行策略。
`InstanceController` 和发现模块通过公开接口更新状态，不访问其他对象的容器和锁。

## 请求链路

```text
HTTP → 请求协议解析 → Router → RequestPreprocessor
                             → KVDirectory 本地匹配
                             → RoutingState + RoutingPolicy → 路由绑定与预留
     → ResponseHandler 绑定 HTTP 回调 → RequestManager 准入
     → InstanceClient → P → D
RPC Generations ← D
     → RequestManager → OutputDispatcher → HTTP 回调 → 客户端
```

PD 的首 token 和后续 token 仍统一由 D 返回。tool call/reasoning 的解析仍在引擎，
服务只进行结构化字段和客户端协议之间的转换。vLLM 后端继续转发原始 HTTP JSON/SSE。

## 并发与生命周期约束

- `RequestManager` 统一持有请求准入与实例删除的屏障。Controller 通过同步下线回调进入
  `retire_instance()`：按代际移除实例、排入取消事件、清理调度预留。并发准入在同一屏障内
  重新校验实例代际，避免请求绑定已经下线的实例。
- 选路绑定的是策略快照中的具体代际，地址被复用时拒绝旧路由，不重新绑定到新实例。
- `InstanceClient` 中的连接按地址与代际校验；旧实例的迟到清理不会删除新实例的 Channel。
- 同一请求的结果和取消事件在请求锁内入队，随后在同一输出队列执行。终止后的
  输出不再调用客户端回调，指标只释放一次。
- 实例状态锁内不执行策略计算、健康探测、Link/Unlink 或 etcd 请求。候选快照持有只读注册数据，避免悬空引用。
- `ServiceApp` 构造函数不启动线程或发现。`start()` 完成模型预处理资源准备后启动
  RPC 控制入口、服务成员和实例发现；有可用实例时开放业务 HTTP 入口。
- 收到停机信号后停止成员与实例发现、停止健康检查及 KV 订阅，再取消并排空请求，
  最后等待网络服务退出。信号处理函数只记录信号，不执行日志或直接 `exit()`。
- 后续配置驱动启动可接入 `cluster/` 的启动描述和进程管理；进程监督与引擎网络客户端
  保持独立。当前未创建没有实际实现的占位类。
