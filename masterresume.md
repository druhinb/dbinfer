# Druhin Bhowal

(425) 362-0237 | [dbhowal@cs.washington.edu](https://www.google.com/search?q=mailto%3Adbhowal%40cs.washington.edu) | [linkedin.com/in/druhinb](https://linkedin.com/in/druhinb) | [github.com/druhinb](https://github.com/druhinb) | U.S. Citizen

---

## Education

### University of Washington — Seattle, WA

**B.S. in Computer Science** | GPA: 3.93/4.0 | _Expected Jun 2027_

- **Coursework:** Adv. Operating Systems, Compilers, Systems Programming, Concurrency & Parallelism, Computer Security, Data Structures, Algorithms, Discrete Math, Linear Algebra, Machine Learning, Deep Learning, NLP, Autonomous Robotics, Distributed Systems

---

## Experience

### Roblox — San Mateo, CA

**Software Engineer Intern** | _June 2026 – Present_

- Added server logs and dead-server history to the creator dashboard: **500K** events/sec through **Flink** into sharded **ClickHouse**.

* Owned the read path, tuning sharding and partitioning to serve server details in **under 100ms** and logs in **under 300ms** at p99.

- Built a reusable \textbf{.NET} package serving feature usage and adoption telemetry, now adopted across the org.
- Contributing to Roblox's from-scratch **Lambda**-style serverless compute platform (RPCs, worker queues, AWS Lambda parity).
- Helping design job scheduling, compute network topology, and low-latency result return for massive game workloads.

### Paul G. Allen School of Computer Science & Engineering — Seattle, WA

**Undergraduate Teaching Assistant** | _Jan 2026 – June 2026_

- Taught **Operating Systems** to **150+** students, debugging memory corruption, page tables, etc. in student **xk** kernels.

### DropzoneAI — Seattle, WA

**Research & Development Intern** | _Sept 2025 – Dec 2025_

- Built agent infrastructure at an **In-Q-Tel**-backed Series B (**$50M+** raised) automating long-horizon security alert triage.
- Designed the agent's long-term memory on **Postgres** and **pgvector** so investigations carry context across alerts, raising speed **35%**.
- Shipped **Azure**, **Jira**, and **CrowdStrike** integrations into the agent's skillset, driving autonomous alert resolution to **90%**.
- Built adapters normalizing alerts, tickets, and EDR telemetry into a unified evidence model the agent reasons over.
- Built the analyst Workbench in **React** and **TypeScript**, linking live investigation sessions to their historical evidence.

### Amazon Web Services — Seattle, WA

**Software Development Engineer Intern** | _June 2025 – Sept 2025_

- Built new shard-failover pipeline for **Kinesis Data Streams** (**Tier-0**, **99.99%** SLA), retiring a recurring class of long-tail outages.
- Modeled hardware telemetry in a **Java**/**Rust** service to proactively migrate at-risk shards, cutting heat-related disruptions **24%**.
- Designed real-time host-failure detection over fleet telemetry, triggering shard failover before customer latency degraded.
- Drove the cross-team rollout across heat, storage, and control-plane stakeholders owning the affected systems.
- Prototyped a **Spark** and **Bedrock** log-analytics layer with SQL-like semantic anomaly search, cutting incident triage **30%**.

### DevMatch — Seattle, WA

**Software Engineer Intern** | _Oct 2024 – June 2025_

- Built a browser IDE on **AKS** with Docker-in-Docker sidecars and per-tenant isolation, scaling users **3x** across **20,000+** runs.
- Built an **OpenTelemetry**, **Prometheus**, and **Grafana** observability stack with alerting, cutting incident response **63%**.
- Built a **LlamaIndex** agentic pipeline over candidate profiles, cutting the hiring cycle **52%** with higher match accuracy.

---

## Projects

### [dbinfer: LLM Inference Engine](https://github.com/druhinb/dbinfer) | _C++23_

- Built a **C++23** LLM inference engine running **GGUF** models on Apple Silicon, mmap-ing weights zero-copy from disk.
- Hand-wrote the CPU forward pass (RMSNorm, RoPE, grouped-query attention, SwiGLU) with optional **Metal** GPU offload path.
- Validated kernels against **PyTorch** reference tensors and checked greedy output token-for-token against **llama.cpp**.

### [drubuntu: ARMv8 Multikernel OS](https://github.com/druhinb/drubuntu) | _C, ARMv8_

- Built a buddy allocator down to **4KB** pages and a hand-written four-level page-table walker on a bare **Barrelfish** skeleton.
- Implemented two IPC paths: **LMP** for synchronous same-core calls and **UMP**, a lock-free shared-memory cross-core channel.
- Wrote an ELF loader, per-process VSpace/CSpace setup, and secondary-core bring-up to run processes across all cores.

### [Shape of Thought: Interpretability of Supervised Vision Transformers](https://github.com/druhinb/LatentInvestigation) | _Python, PyTorch, Vision Transformers_

- Probed how **DINOv2**, **I-JEPA**, **MoCo v3**, and supervised ViTs encode latent 3D structure on ShapeNet and 3D-R2N2.
- Trained linear, MLP, and 3D Conv probes over frozen ViT features to localize 3D geometry across patch and pooled tokens.
- Found self-distillation encodes 3D geometry with **40%** lower reconstruction error than supervised ViT baselines.

### [Snowfight: 4X Strategy Game](https://github.com/druhinb/snowfight) | _C#, SDL 2.0, Systems Programming_

- Wrote a **C#**/**SDL 2.0** game engine with batched rendering, an audio mixer, input system, and a **60 FPS** fixed-tick loop.
- Separated input, simulation, and rendering behind an event-driven architecture for deterministic frame timing.
- Generated infinite procedural worlds from layered noise, streaming chunks on demand to keep memory usage bounded.

---

## Technical Skills

- **Languages:** C++, C, Rust, C#, Java, Python, TypeScript/JavaScript, SQL, Bash, x86-64 Assembly
- **Technologies:** Docker, Kafka, Flink, ClickHouse, Kubernetes, AWS, Azure, Postgres, Spark, MongoDB, Node.js, React, PyTorch, OpenTelemetry, Grafana, Prometheus, gRPC, REST, Linux, Git
- **Concepts:** Distributed Systems, Parallel Computing, Performance Optimization, Operating Systems, Kernel Development, Compilers, Concurrency, Observability, Backend, Databases, ML, NLP
