from pathlib import Path
import re

path = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/mxfp4.py")
text = path.read_text()

anchor = "logger = init_logger(__name__)\n"
helper = r'''
def _dspark_construction_uva_enabled() -> bool:
    """Use host-backed UVA storage while constructing the DSpark draft MoE."""
    import os

    if os.environ.get("VLLM_DSPARK_CONSTRUCTION_UVA", "0") not in (
        "1",
        "true",
        "True",
    ):
        return False
    # set_model_tag("dspark_head") surrounds the independent draft model
    # construction. Do not change construction for the target backbone.
    import vllm.compilation.backends as compilation_backends

    return (
        compilation_backends.model_tag == "dspark_head"
        and current_platform.is_cuda()
    )


def _dspark_weight_zeros(*shape, dtype: torch.dtype) -> torch.Tensor:
    if not _dspark_construction_uva_enabled():
        return torch.zeros(*shape, dtype=dtype)

    # The draft's raw MXFP4 tensors are only staging buffers. Create them in
    # exact-size pinned host memory and expose CUDA UVA views immediately, so
    # construction never allocates the raw expert block in HBM.
    from vllm.model_executor.offloader.base import pin_exact, should_pin_memory
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    cpu_data = torch.zeros(*shape, dtype=dtype, device="cpu")
    if should_pin_memory():
        cpu_data = pin_exact(cpu_data)
    return get_accelerator_view_from_cpu_tensor(cpu_data)


def _dspark_parameter(
    data: torch.Tensor, *, requires_grad: bool = False
) -> torch.nn.Parameter:
    parameter = torch.nn.Parameter(data, requires_grad=requires_grad)
    if _dspark_construction_uva_enabled():
        # device_loading_context() will process the UVA-backed raw tensor and
        # re-offload the packed replacement after Marlin repacking.
        parameter._vllm_is_uva_offloaded = True
    return parameter
'''
if anchor not in text:
    raise SystemExit("mxfp4 logger anchor not found")
if "_dspark_construction_uva_enabled" not in text:
    text = text.replace(anchor, anchor + helper, 1)

pattern = re.compile(r"torch\.nn\.Parameter\((\s*)torch\.zeros\(")
text, count = pattern.subn(
    lambda m: f"_dspark_parameter({m.group(1)}_dspark_weight_zeros(",
    text,
)
if count < 10:
    raise SystemExit(f"expected both MXFP4 create_weights blocks, replaced={count}")

path.write_text(text)
print(f"DSPARK_CONSTRUCTION_UVA_PATCHED parameters={count}")

