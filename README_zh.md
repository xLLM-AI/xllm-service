<!-- Copyright 2022 JD Co.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this project except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. -->

[English](./README.md) | [中文](./README_zh.md)


<p align="center">
    <img src="docs/assets/xllm_service_title.png" alt="xLLM" style="width:50%; height:auto;">
</p>

## 1. 简介
**xLLM-service** 是一个基于 xLLM 推理引擎开发的服务层框架，为集群化部署提供高效率、高容错、高灵活性的大模型推理服务。

xLLM-service 旨在解决企业级服务场景中的关键挑战：
- 如何于在离线混合部署环境中，保障在线服务的SLA，提升离线任务的资源利用率。
- 如何适应实际业务中动态变化的请求负载，如输入/输出长度出现剧烈波动。
- 解决多模态模型请求的性能瓶颈。
- 保障集群计算实例的高可靠性。

--- 

## 2. 核心特性

xLLM-service 通过对计算资源池的动态管理、请求的智能调度与抢占，以及计算实例的实时监控，实现了以下核心能力：
- 在线与离线任务的统一调度，在线请求的抢占式执行，离线请求best-effort执行；
- PD比例的自适应动态调配，支持实例PD角色的高效切换；
- 多模态请求的EPD三阶段分离，不同阶段的资源智能分配；
- 多节点容错架构，快速感知实例错误信息，自动决策最优的被中断请求再调度方案。

---

## 3. 代码结构

```
├── xllm-service/
|   : 主代码目录
│   ├── chat_template/               # 
│   ├── common/                      # 
│   ├── examples/                    # 
│   ├── http_service/                # 
│   ├── rpc_service/                 # 
|   ├── tokenizers/                  #
|   └── master.cpp                   # 
```
---


## 4. 快速开始
#### 安装
```bash
git clone git@coding.jd.com:xllm-ai/xllm_service.git
cd xllm_service
git submodule init
git submodule update
```
#### 编译
编译执行
```bash
# prepare.sh 会检查 libcpprest-dev；在 Debian/Ubuntu root 环境中缺失时自动安装，
# 否则请先手动安装 libcpprest-dev，然后脚本会应用 cpprestsdk patch。
sh prepare.sh
mkdir -p build && cd build
cmake .. && make -j 8
```

---
## 5. 成为贡献者
您可以通过以下方法为 xLLM-Service 作出贡献:

1. 在Issue中报告问题
2. 提供改进建议
3. 补充文档
    + Fork仓库
    + 修改文档
    + 提出pull request
4. 修改代码
    + Fork仓库
    + 创建新分支
    + 加入您的修改
    + 提出pull request

感谢您的贡献！ 🎉🎉🎉
如果您在开发中遇到问题，请参阅**[xLLM-Service中文指南](./docs/docs_zh/readme.md)**

---

## 6. 社区支持

如果你在xLLM的开发或使用过程中遇到任何问题，欢迎在项目的Issue区域提交可复现的步骤或日志片段。
如果您有企业内部Slack，请直接联系xLLM Core团队。

欢迎沟通和联系我们:

<div align="center">
  <img src="docs/assets/wechat_qrcode1.png" alt="qrcode1" width="30%" />
  <img src="docs/assets/wechat_qrcode2.png" alt="qrcode2" width="30%" />
</div>

## 7. 致谢

感谢以下为xLLM-Servic作出贡献的[开发者](https://github.com/jd-opensource/xllm-service/graphs/contributors)
<a href="https://github.com/jd-opensource/xllm-service/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/xllm-service" />
</a>

---

## 8. 许可证
[Apache License](LICENSE)

#### xLLM-Service 由 JD.com 提供 
#### 感谢您对xLLM的关心与贡献!
