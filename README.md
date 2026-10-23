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


## 1. Project Overview
**xLLM-service** is a service-layer framework developed based on the **xLLM** inference engine, providing efficient, fault-tolerant, and flexible LLM inference services for clustered deployment.

xLLM-service targets to address key challenges in enterprise-level service scenarios:

- How to ensure the SLA of online services and improve resource utilization of offline tasks in a hybrid online-offline deployment environment.

- How to react to changing request loads in actual businesses, such as fluctuations in input/output lengths.

- Resolving performance bottlenecks of multimodal model requests.

- Ensuring high reliability of computing instances.

---

## 2. Key Features
With management of computing resource pools, intelligent scheduling and preemption of hybrid requests, and real-time monitoring of computing instances, xLLM-service achieves the following key features:

- Unified scheduling of online and offline requests, with preemptive execution for online requests and best-effort execution for offline requests.

- Adaptive dynamic allocation of PD ratios, supporting efficient switching of instance PD roles.

- EPD three-stage disaggregation for multimodal requests, with intelligent resource allocation for different stages.

- Fault-tolerant architecture, fast detection of instance error and automatic rescheduling for interrupted requests. 

---

## 3. Core Architecture

```
xllm_service/
├── config/                # Startup configuration
├── router/                # Routing and request lifecycle
├── cluster/               # Instance state, health and engine clients
├── coordinator/           # etcd discovery and service membership
├── request/               # Request data, preprocessing and ordered output
├── kv_conductor/          # Asynchronous KV events and prefix index
├── http_service/          # HTTP API and response encoding
├── rpc_service/           # RPC API and structured output adaptation
├── chat_template/         # Chat templates
├── tokenizer/             # Tokenizer pool
├── common/                # Shared types and utilities
├── service_app.cpp        # Composition and lifecycle
└── main.cpp               # Entry point and signal handling
```

---


## 4. Quick Start
#### Installation
```bash
git clone git@coding.jd.com:xllm-ai/xllm_service.git
cd xllm_service
git submodule init
git submodule update
```
#### Compilation
compile xllm-service: 
```bash
sh prepare.sh # apply patch
mkdir -p build && cd build
cmake .. && make -j 8
```

--- 

## 5. Contributing

There are several ways you can contribute to xLLM:

1. Reporting Issues (Bugs & Errors)
2. Suggesting Enhancements
3. Improving Documentation
    + Fork the repository
    + Add your view in document
    + Send your pull request
4. Writing Code
    + Fork the repository
    + Create a new branch
    + Add your feature or improvement
    + Send your pull request

We appreciate all kinds of contributions! 🎉🎉🎉
If you have problems about development, please check our document: * **[Document](./docs/docs/readme.md)**

---

## 6. Community & Support

If you encounter any issues along the way, you are welcomed to submit reproducible steps and log snippets in the project's Issues area, or contact the xLLM Core team directly via your internal Slack.

Welcome to contact us:

<div align="center">
  <img src="docs/assets/wechat_qrcode1.png" alt="qrcode1" width="30%" />
  <img src="docs/assets/wechat_qrcode2.png" alt="qrcode2" width="30%" />
</div>

---
## 7. About the Contributors

Thanks to all the following [developers](https://github.com/jd-opensource/xllm-service/graphs/contributors) who have contributed to xLLM.
<a href="https://github.com/jd-opensource/xllm-service/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jd-opensource/xllm-service" />
</a>

---

## 8. License

[Apache License](LICENSE)

#### xLLM is provided by JD.com 
#### Thanks for your Contributions!
