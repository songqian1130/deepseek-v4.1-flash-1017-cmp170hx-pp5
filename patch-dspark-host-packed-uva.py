from pathlib import Path

path = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
    "quantization/utils/marlin_utils_fp4.py"
)
text = path.read_text()

anchor = "logger = init_logger(__name__)\n"
helper = r'''

def _dspark_host_packed_output_enabled() -> bool:
    """Keep the DSpark Marlin packed aggregate out of device memory."""
    import os

    if os.environ.get("VLLM_DSPARK_CONSTRUCTION_UVA", "0") not in (
        "1",
        "true",
        "True",
    ):
        return False
    import vllm.compilation.backends as compilation_backends

    return compilation_backends.model_tag == "dspark_head"
'''
if "_dspark_host_packed_output_enabled" not in text:
    if anchor not in text:
        raise SystemExit("marlin logger anchor not found")
    text = text.replace(anchor, anchor + helper, 1)

old = '''    out: torch.Tensor | None = None
    stage = torch.empty(host.shape[1:], dtype=host.dtype, device=device)
    for i in range(e):
        stage.copy_(host[i])
        qweight = stage.view(torch.int32).T.contiguous()
        packed = ops.gptq_marlin_repack(
            b_q_weight=qweight,
            size_k=size_k,
            size_n=size_n,
            num_bits=4,
            is_a_8bit=is_a_8bit,
        )
        if out is None:
            out = torch.empty((e, *packed.shape), dtype=packed.dtype, device=device)
        out[i] = packed
    del stage, host
    assert out is not None
    return out
'''
new = '''    out: torch.Tensor | None = None
    host_out: torch.Tensor | None = None
    stage = torch.empty(host.shape[1:], dtype=host.dtype, device=device)
    use_host_output = _dspark_host_packed_output_enabled()
    for i in range(e):
        stage.copy_(host[i])
        qweight = stage.view(torch.int32).T.contiguous()
        packed = ops.gptq_marlin_repack(
            b_q_weight=qweight,
            size_k=size_k,
            size_n=size_n,
            num_bits=4,
            is_a_8bit=is_a_8bit,
        )
        if out is None and host_out is None:
            shape = (e, *packed.shape)
            if use_host_output:
                from vllm.model_executor.offloader.base import pin_exact

                host_out = pin_exact(
                    torch.empty(shape, dtype=packed.dtype, device="cpu")
                )
            else:
                out = torch.empty(shape, dtype=packed.dtype, device=device)
        if host_out is not None:
            host_out[i].copy_(packed)
        else:
            assert out is not None
            out[i] = packed
        del packed
    del stage, host
    if host_out is not None:
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

        out = get_accelerator_view_from_cpu_tensor(host_out)
    assert out is not None
    return out
'''
if old not in text:
    raise SystemExit("staged Marlin body anchor not found")
text = text.replace(old, new, 1)
path.write_text(text)
print("DSPARK_HOST_PACKED_UVA_PATCHED")
