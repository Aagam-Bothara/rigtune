# Security

rigtune is alpha software, maintained by one person. This page says how to report a vulnerability and, more usefully, what rigtune does and does not protect, so you can decide how to deploy it.

## Reporting a vulnerability

Please do not open a public issue. Report it privately through GitHub instead: the repository's **Security** tab, then **Report a vulnerability**. Include the version (`pip show rigtune`), the command you ran, and what someone could do with it.

There is no guaranteed response time; reports are read and answered as soon as possible. Fixes go into the latest release only.

## What rigtune protects, and what it does not

**Authentication is off unless you set a key.** With `RIGTUNE_API_KEY` (or the variable `--api-key-env` names) set, `rigtune serve` requires `Authorization: Bearer <key>` on every route but `/health`, `/rigtune/profile` and `/rigtune/drift` included, and `/health` then says only whether the backend is alive and ready, without its process id, upstream address or error text. `--require-api-key` refuses to start when the variable is empty. Without a key nothing checks who is calling. The key is compared in constant time and is not passed on to the engines, which rigtune launches without one; it is sent in clear unless TLS terminates in front of rigtune.

**Limits are off unless you ask for them.** By default rigtune refuses, delays and cuts short nothing: an upgrade changes nothing that worked before. `--max-in-flight N` (or `measured`) bounds the inference requests served at once, with a bounded queue (`--queue`, `--queue-timeout`) and 429 beyond it; `--idle-timeout` closes a stream that has sent nothing for that long with an SSE error event; `--max-body-mb` refuses larger bodies with 413, chunked uploads included; `--body-timeout` refuses slow uploads with 408. These are process-wide bounds, not per-client rate quotas. See [safeguards](docs/runtime-hardening.md#serving).

**It listens on this machine only, unless told otherwise.** `rigtune serve` binds `127.0.0.1`. `--host 0.0.0.0`, or `RIGTUNE_HOST`, exposes it to every network the machine can reach, and it warns when it starts that way without a key. Before doing that, set a key and put a TLS-terminating reverse proxy in front of it with the per-client rate limits your deployment needs. The Docker image sets `RIGTUNE_HOST=0.0.0.0`, because inside a container that is the only address `-p` can reach; publish with `-p 127.0.0.1:8000:8000` to keep a container local as well.

**`/rigtune/profile` describes the machine.** It returns the hardware (GPU and CPU models, memory, driver versions), the model, the engine's launch arguments, which can include local file paths, and the calibration results. It does not return secrets from the environment: the launch environment it stores holds only the variables rigtune sets itself, such as `CUDA_VISIBLE_DEVICES` and `NCCL_P2P_DISABLE`, never the process environment, so tokens such as `HF_TOKEN` stay out of it.

**The proxy changes streaming requests.** It adds `stream_options: {"include_usage": true}` to a streaming request that did not set `include_usage` itself, so the drift report can count tokens. Everything else passes through unchanged.

**Traffic monitoring never reads text.** The drift report records token counts, concurrency and time to first token. It never records prompts or answers.

**A disk KV-cache tier writes your prompts' KV cache to disk.** Only with `--kv-tier auto`, only on SGLang, and only when host RAM cannot hold the working set: SGLang then keeps cache blocks, computed from the prompts it served, in files. (vLLM 0.29's disk tier has no size limit, so it is never offered.) What rigtune guarantees:

- The files go only under rigtune's own cache dir, in a directory per engine, `~/.rigtune/kv-tier/pid<rigtune pid>-port<port>` (or under `RIGTUNE_HOME`), created empty and readable by your user only (mode 0700). Never on a network filesystem: rigtune reads the cache dir's filesystem type from `/proc/mounts` and offers no disk tier on NFS, SMB, Ceph, GlusterFS, 9p or FUSE (RunPod's `/workspace` volume among them), nor under `/workspace` where the mount table cannot be read.
- SGLang is always given a size cap for the directory (`SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE`, the size rigtune checked against the free disk), past which it evicts the least recently used files itself.
- rigtune deletes the directory before and after every calibration trial, and when rigtune exits normally or on SIGTERM.

What it does not guarantee: a SIGKILL, an out-of-memory kill or a power loss runs no cleanup, so the files stay until the next rigtune calibration or server starts, which removes every tier directory whose owning rigtune process is gone. The files are not encrypted, and SGLang does not delete them on its own. A host-RAM tier holds the same data in memory only; vLLM maps it from `/dev/shm`, which it unlinks once mapped.

A server rigtune supervises gets the same: an empty directory at start and at every restart, deleted when it stops. A profile whose tier no longer fits, or whose disk tier would now land on a network filesystem, is served without the tier, with a warning.

**The trial ledger keeps every calibration trial.** Each calibration appends its trials to `~/.rigtune/ledger/` (under `RIGTUNE_HOME`, or where `RIGTUNE_LEDGER` points; `RIGTUNE_LEDGER=off` turns it off). On POSIX its files are created mode 0600 and its folders (`ledger/` and one per machine) 0700, readable by your user only; folders above them, such as `~/.rigtune`, keep the default mode, and Windows ignores modes, so there the folder's inherited permissions apply. It holds what a cached profile holds: the hardware (GPU and CPU models, driver versions), the model id, which is a local path for a local model, the engine's launch arguments, which can include local file paths, the launch environment rigtune sets for the engine (such as `CUDA_VISIBLE_DEVICES`, never the process environment), and each trial's metrics and error text, which can quote the engine's log. It holds no prompt or answer text, no variable from your environment, such as `HF_TOKEN`, and no host name. Each record's id is the sha256 of its content, which finds a line cut short or changed by accident; it is not a signature, and anyone who can write the file can change a record and its id together, so the ledger is an audit trail for your own use, not evidence against tampering. It is never sent anywhere and nothing rigtune publishes, such as `compare --verdict` results, reads from it. Unlike a profile it is not replaced by the next calibration: delete the directory to remove it.

**Engines run locally, as you.** vLLM, SGLang and llama.cpp run as child processes listening on `127.0.0.1`, with the permissions of the user who started rigtune; their own vulnerabilities belong upstream. rigtune never passes `--trust-remote-code`, so a model repository cannot run code through it.

**Some weights can come from someone other than the model's author.** The model you name, and any draft model speculative decoding adds, come from that model's own organisation on the Hub. Two kinds of file can come from a third party: pre-quantized 4-bit and 8-bit checkpoints, which are off unless you allow them (`--quant auto,awq,gptq,w8a8`), and llama.cpp's GGUF files, which it may download from known quantizers such as bartowski or TheBloke. rigtune checks what those files are, from a checkpoint's quantization config, not who published them. They are safetensors or GGUF, which hold data rather than code, so the risk is altered answers rather than code execution. With `--backend vllm` or `--backend sglang` and the default `--quant auto`, rigtune downloads only from the named model's own organisation.

**Power control acts on the whole machine.** `--power` and `rigtune power` set GPU power caps and clock locks through the driver, which needs root, and they apply to every process using that GPU. A crash can leave them set; `rigtune power reset` undoes that.
