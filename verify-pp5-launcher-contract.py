"""Check the live container's effective command against the original 9010 profile."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inspect", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--graph-mode", choices=("PIECEWISE", "FULL_AND_PIECEWISE"), default="PIECEWISE")
    p.add_argument("--runner", choices=("reference-default", "v1"), default="reference-default")
    p.add_argument("--source-verification", type=Path)
    p.add_argument("--optimized-profile", action="store_true")
    p.add_argument("--engram-chunk-rows", type=int)
    args = p.parse_args()
    container = json.loads(args.inspect.read_text(encoding="utf-8"))[0]
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    command = container["Config"]["Cmd"]
    env = dict(s.split("=",1) for s in container["Config"]["Env"] if "=" in s)
    flags = {}
    i = 0
    while i < len(command):
        item = command[i]
        if item.startswith("--"):
            if "=" in item:
                key, value = item.split("=",1)
            elif i+1 < len(command) and not command[i+1].startswith("--"):
                key, value = item, command[i+1]
                i += 1
            else:
                key, value = item, True
            flags[key] = value
        i += 1
    defaults = reference["reference_executable_defaults"]
    matches = {}
    for flag, name in {"--port":"PORT", "--max-model-len":"MAXLEN",
                       "--gpu-memory-utilization":"GPU_UTIL", "--max-num-batched-tokens":"MAX_BATCHED",
                       "--max-num-seqs":"MAX_SEQS"}.items():
        assert flags[flag] == defaults[name], (flag,flags.get(flag),defaults[name])
        matches[flag] = flags[flag]
    for flag, expected in {"--pipeline-parallel-size":"5", "--tensor-parallel-size":"1",
                            "--kv-cache-dtype":"fp8_ds_mla", "--tokenizer-mode":"deepseek_v41",
                            "--tool-call-parser":"deepseek_v41", "--reasoning-parser":"deepseek_v41"}.items():
        assert flags[flag] == expected, (flag,flags.get(flag))
        matches[flag] = expected
    for flag in ("--enable-prefix-caching", "--trust-remote-code", "--disable-custom-all-reduce", "--enable-auto-tool-choice"):
        assert flags.get(flag) is True, flag
        matches[flag] = True
    assert json.loads(flags["--engram-config"]) == {"cpu_offload":True}
    assert "--speculative-config" not in flags and defaults["PLAIN"] == "1"
    assert "--cpu-offload-gb" not in flags and "--enable-dbo" not in flags
    for key, value in {"CUDA_VISIBLE_DEVICES":defaults["GPUS"],
                        "VLLM_PP_LAYER_PARTITION":defaults["PP_PARTITION"],
                        "VLLM_CPU_OFFLOAD_GB_PER_RANK":defaults["OFFLOAD_RANKS"],
                        "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB":"128",
                        "NCCL_ALGO":"Ring", "NCCL_PROTO":"Simple"}.items():
        assert env[key] == value, (key,env.get(key),value)
        matches[key] = value
    compilation = json.loads(flags["--compilation-config"])
    assert compilation == {"cudagraph_mode":args.graph_mode, "cudagraph_capture_sizes":[1,2,4,8], "max_cudagraph_capture_size":8}
    if args.runner == "v1":
        assert env.get("VLLM_USE_V2_MODEL_RUNNER") == "0"
    if args.graph_mode == "FULL_AND_PIECEWISE":
        assert args.runner == "v1" and args.source_verification, "FULL requires the tested V1 profile and live source proof"
        source = json.loads(args.source_verification.read_text(encoding="utf-8"))
        assert source["ok"] and not source["missing"] and not source["unapproved_differences"]
    assert env["VLLM_USE_BREAKABLE_CUDAGRAPH"] == "1"
    assert env["DS41_ENGRAM_MANIFEST"].endswith("model.safetensors.index.json")
    assert flags["--host"] == "127.0.0.1"
    if args.optimized_profile:
        assert args.graph_mode=="FULL_AND_PIECEWISE" and args.runner=="v1"
        assert env.get("VLLM_PP5_LAZY_RECEIVE")=="1"
        assert env.get("VLLM_PP5_SHARED_MAPPING")=="1"
        assert env.get("VLLM_SERVER_DEV_MODE", "0")=="0"
        assert "--worker-extension-cls" not in flags and "--profiler-config" not in flags
    if args.engram_chunk_rows is not None:
        assert env.get("VLLM_ENGRAM_MMAP_CHUNK_ROWS")==str(args.engram_chunk_rows)
    differences=list(reference["intentional_differences"])
    if args.graph_mode=="FULL_AND_PIECEWISE":
        differences=[x for x in differences if not x.startswith("PIECEWISE")]
        differences.append("FULL_AND_PIECEWISE with SSD-specific host breaks; DBO disabled")
    if args.runner=="v1":differences.append("Explicit V1 runner selected by controlled local measurement")
    if args.optimized_profile:differences.append("Deferred PP receive and one shared GPU token/request map")
    if args.engram_chunk_rows is not None:differences.append(f"SSD row gather chunk_rows={args.engram_chunk_rows}")
    result = {"ok":True, "container":container["Name"], "image_id":container["Image"],
              "reference_tag":reference["launcher_tag"], "matched":matches,
              "plain_mode":True, "expert_offload_gb":0,
              "disk_engram_manifest":env["DS41_ENGRAM_MANIFEST"], "compilation":compilation,
              "runner":args.runner,
              "optimized_profile":args.optimized_profile,
              "engram_chunk_rows":int(env.get("VLLM_ENGRAM_MMAP_CHUNK_ROWS","1024")),
              "source_verification":str(args.source_verification) if args.source_verification else None,
              "intentional_differences":differences}
    args.out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"ok":True,"matched_settings":len(matches),"plain_mode":True,
                      "expert_offload_gb":0,"compilation":compilation}))
    print("LIVE_LAUNCHER_CONTRACT_VERIFIED")


if __name__ == "__main__":
    main()
