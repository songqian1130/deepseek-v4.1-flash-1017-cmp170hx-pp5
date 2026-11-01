"""Exercise the actual capture-input AST without importing vLLM or loading weights."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace


def capture_input_block(source: str):
    tree = ast.parse(source)
    manager = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelCudaGraphManager"
    )
    capture = next(
        node for node in manager.body
        if isinstance(node, ast.FunctionDef) and node.name == "capture"
    )
    factory = next(
        node for node in capture.body
        if isinstance(node, ast.FunctionDef) and node.name == "create_forward_fn"
    )
    start = next(
        i for i, node in enumerate(factory.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "model_inputs" for t in node.targets)
    )
    assignment, nonfirst = factory.body[start:start + 2]
    if not isinstance(nonfirst, ast.If):
        raise AssertionError("Capture PP input branch no longer follows model_inputs")
    return compile(ast.Module(body=[assignment, nonfirst], type_ignores=[]),
                   "<actual-capture-input-block>", "exec")


def run_case(code, *, first: bool, model_type: str):
    ids = [7, 11, 13, 17]
    scope = {
        "self": SimpleNamespace(
            is_first_pp_rank=first,
            vllm_config=SimpleNamespace(model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type=model_type)))),
        "input_buffers": SimpleNamespace(input_ids=ids, positions=[0, 1, 2, 3]),
        "model_state": SimpleNamespace(prepare_dummy_inputs=lambda reqs, tokens: {}),
        "num_reqs": 1,
        "num_tokens": 3,
        "intermediate_tensors": ["h0", "h1", "h2", "padding"],
    }
    exec(code, scope)
    return scope["model_inputs"], ids[:3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--expect-broken", action="store_true")
    args = parser.parse_args()
    code = capture_input_block(args.source.read_text(encoding="utf-8"))
    actual, ids = run_case(code, first=False, model_type="deepseek_v41")
    if args.expect_broken:
        assert actual["input_ids"] is None, "Expected to reproduce the old missing-ID bug"
        print(json.dumps({"before_patch_bug_reproduced": True}))
        return
    assert actual["input_ids"] == ids, "DeepSeek non-first PP stage lost its token IDs"
    assert actual["inputs_embeds"] is None
    assert actual["intermediate_tensors"] == ["h0", "h1", "h2"]
    for first in (False, True):
        for model_type in ("deepseek_v41", "unrelated_model"):
            actual, ids = run_case(code, first=first, model_type=model_type)
            expected = ids if first or model_type == "deepseek_v41" else None
            assert actual["input_ids"] == expected, (first, model_type, actual)
            assert actual["positions"] == [0, 1, 2]
    print(json.dumps({"capture_input_cases_passed": 4,
                      "eager_warmup_and_capture_share_input_factory": True,
                      "gpu_runtime_validation": "pending"}))


if __name__ == "__main__":
    main()
