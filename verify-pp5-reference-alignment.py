"""Verify package parity and execute the changed input/normalization AST paths."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def nodes(source):
    return ast.parse(source).body


def compile_nodes(body):
    return compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), *body], type_ignores=[])), "<actual-source>", "exec")


def input_block(source, capture):
    if capture:
        cls = next(n for n in nodes(source) if isinstance(n, ast.ClassDef) and n.name == "ModelCudaGraphManager")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "capture")
        fn = next(n for n in method.body if isinstance(n, ast.FunctionDef) and n.name == "create_forward_fn")
    else:
        fn = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef)
                  and any(isinstance(s, ast.Assign) and any(isinstance(t, ast.Name)
                  and t.id == "model_inputs" for t in s.targets) for s in n.body))
    i = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "model_inputs" for t in n.targets))
    assert isinstance(fn.body[i+1], ast.If)
    return compile_nodes(fn.body[i:i+2])


class Tensor:
    def __init__(self, n, copied=False):
        self.shape = (n, 2)
        self.copied = copied

    def __getitem__(self, key):
        return Tensor(min(self.shape[0], key.stop))

    def copy_(self, other):
        assert self.shape == other.shape
        self.copied = True
        return self


def regressions(package, reference_engram):
    results = {}
    capture = input_block((package / "v1/worker/gpu/cudagraph_utils.py").read_text(), True)
    for first in (False, True):
        for needs in (False, True):
            scope = {"self": SimpleNamespace(is_first_pp_rank=first),
                "model": SimpleNamespace(requires_raw_input_tokens=needs),
                "requires_raw_input_tokens": lambda model: model.requires_raw_input_tokens,
                "input_buffers": SimpleNamespace(input_ids=[7,11,13,17], positions=[0,1,2,3]),
                "model_state": SimpleNamespace(prepare_dummy_inputs=lambda *a: {}),
                "num_reqs": 1, "num_tokens":3, "intermediate_tensors":["a","b","c","pad"]}
            exec(capture, scope)
            assert scope["model_inputs"]["input_ids"] == ([7,11,13] if first or needs else None)
            if not first:
                assert scope["model_inputs"]["intermediate_tensors"] == ["a","b","c"]
    results["capture_capability_cases"] = 4
    runner = input_block((package / "v1/worker/gpu/model_runner.py").read_text(), False)
    for needs, dummy, recv_rows, raises in ((True,False,3,False),(False,False,3,False),
                                          (True,False,2,True),(True,True,2,False)):
        scope = {"self": SimpleNamespace(is_first_pp_rank=False,
                    model=SimpleNamespace(requires_raw_input_tokens=needs),
                    model_state=SimpleNamespace(prepare_inputs=lambda *a: {}), req_states=None,
                    intermediate_tensors=SimpleNamespace(tensors={"hidden":Tensor(4)})),
                "input_ids":[7,11,13], "inputs_embeds":None,
                "input_batch":SimpleNamespace(positions=[0,1,2], num_tokens_after_padding=3),
                "intermediate_tensors":SimpleNamespace(tensors={"hidden":Tensor(recv_rows)}),
                "requires_raw_input_tokens":lambda model:model.requires_raw_input_tokens,
                "dummy_run":dummy, "IntermediateTensors":lambda x:x}
        try:
            exec(runner, scope)
        except RuntimeError as exc:
            assert raises and "pipeline is out of sync" in str(exc), str(exc)
        else:
            assert not raises
            assert scope["model_inputs"]["input_ids"] == ([7,11,13] if needs else None)
            assert scope["model_inputs"]["intermediate_tensors"]["hidden"].copied == (not dummy)
    results["runner_capability_and_desync_cases"] = 4
    tokenizer = next(n for n in nodes((package / "tokenizers/deepseek_v41.py").read_text())
                     if isinstance(n, ast.FunctionDef) and n.name == "_normalize_messages")
    scope = {"copy":copy, "IMAGE_PLACEHOLDER":"<image>"}
    exec(compile_nodes([tokenizer]), scope)
    normalize = scope["_normalize_messages"]
    for kind in ("text","input_text","output_text"):
        original = [{"role":"user","content":[{"type":kind,"text":"hello"}]}]
        assert normalize(original)[0]["content"] == "hello"
        assert isinstance(original[0]["content"], list)
    for kind in ("image_url","input_image","image_pil"):
        assert normalize([{"role":"user","content":[{"type":kind}]}])[0]["content"] == "<image>"
    try:
        normalize([{"role":"user","content":[{"type":"unsupported"}]}])
    except ValueError:
        pass
    else:
        raise AssertionError("Unsupported tokenizer content was accepted")
    results["tokenizer_content_cases"] = 7
    ref_tree = ast.parse(reference_engram.read_text())
    live_tree = ast.parse((package / "models/deepseek_v4_1/common/engram.py").read_text())
    preserved = ("_stream_copy_engram", "_engram_head_shard_weight_loader", "_exact_pinned_tensor",
                 "_start_prefetch", "_finish_prefetch", "_ready_rows")
    for name in preserved:
        old = next(n for n in ast.walk(ref_tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        new = next(n for n in ast.walk(live_tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False), name
    # Exercise actual Engram dispatch: disk staging and reference resident-table fallback.
    for disk in (False, True):
        calls = []
        table = SimpleNamespace(prefetch=lambda *a:calls.append("prefetch")) if disk else None
        staged = Tensor(4)
        embedding = SimpleNamespace(disk_table=table, vocab_start_idx=0, vocab_end_idx=100,
            lookup=lambda *a, **kw:calls.append("lookup"), tp_size=1)
        obj = SimpleNamespace(embed_tokens=embedding, staged_rows=staged,
            _prefetch_stream=None, _prefetch_pending=False,
            _ready_rows=lambda n:Tensor(n))
        scope = {"torch":SimpleNamespace(cuda=SimpleNamespace(is_current_stream_capturing=lambda:False))}
        for name in ("prepare_embeddings","embed"):
            fn = next(n for n in ast.walk(live_tree) if isinstance(n, ast.FunctionDef) and n.name == name)
            exec(compile_nodes([fn]), scope)
        scope["prepare_embeddings"](obj, Tensor(3))
        assert scope["embed"](obj, Tensor(3)).shape == (3,2)
        assert calls == (["prefetch","lookup"] if disk else ["lookup"]), calls
    results["engram_reference_methods_unchanged"] = len(preserved)
    results["engram_disk_and_resident_dispatch_cases"] = 2
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-engram", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = args.package.parent
    missing, mismatch = [], []
    for name, expected in manifest["expected"].items():
        path = root / name
        if not path.is_file(): missing.append(name)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected: mismatch.append(name)
    extra_bad = []
    for name, expected in manifest["extra_hashes"].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            extra_bad.append(name)
    assert not missing and not mismatch and not extra_bad, (missing,mismatch,extra_bad)
    for name in [*manifest["restored_exact"], *manifest["allowed_changed"], *manifest["adapter_files"]]:
        path = root / name
        compile(path.read_bytes(), str(path), "exec")
    result = {"ok":True, "reference_commit":manifest["commit"],
        "reference_source_files":manifest["reference_files"],
        "reference_identical":manifest["reference_files"]-len(manifest["allowed_changed"]),
        "allowed_changed":manifest["allowed_changed"], "missing":missing,
        "unapproved_differences":mismatch, "runtime_extras_unchanged":len(manifest["extra_hashes"]),
        "regressions":regressions(args.package, args.reference_engram),
        "native_binary_provenance":manifest["native_binary_provenance"]}
    if args.out:
        args.out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    print("REFERENCE_ALIGNMENT_VERIFIED")


if __name__ == "__main__":
    main()
