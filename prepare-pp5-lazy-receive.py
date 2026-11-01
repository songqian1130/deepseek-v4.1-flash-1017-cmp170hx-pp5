"""Produce the narrow lazy-receive worker patch against the verified baseline."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
raw=a.source.read_bytes();assert hashlib.sha256(raw).hexdigest()=='6fd8c48008da7664095c3a3a66c0f15b1842e1fdf8afbcb4911de77e3233a242'
s=raw.decode(); original=s
old='''        comm_postprocess: list[Callable[[], None]] | None = None,
    ) -> None:
        super().__init__(tensors)
        self._comm_handles = comm_handles'''
new='''        comm_postprocess: list[Callable[[], None]] | None = None,
        comm_loader: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__(tensors)
        self._comm_loader = comm_loader
        self._comm_handles = comm_handles'''
assert s.count(old)==1;s=s.replace(old,new)
old='''        if self._comm_waited:
            return
        if self._comm_handles:'''
new='''        if self._comm_waited:
            return
        if self._comm_loader is not None:
            global PP5_DEFERRED_RECEIVE_COUNT
            tensors, handles, postprocess = self._comm_loader()
            assert tensors is not None
            self.tensors = tensors
            self._comm_handles = handles
            self._comm_postprocess = postprocess
            self._comm_loader = None
            PP5_DEFERRED_RECEIVE_COUNT += 1
        if self._comm_handles:'''
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace('class AsyncIntermediateTensors(IntermediateTensors):','PP5_DEFERRED_RECEIVE_COUNT = 0\n\n\nclass AsyncIntermediateTensors(IntermediateTensors):',1)
needle='''        self._pp_send_work: list[Handle] = []'''
replacement=needle+'''
        self._pp5_defer_receive = os.environ.get("VLLM_PP5_LAZY_RECEIVE", "0") == "1"
        if self._pp5_defer_receive:
            assert not self.use_v2_model_runner
            assert self.parallel_config.tensor_parallel_size == 1
            assert self.parallel_config.pipeline_parallel_size == 5
            assert self.model_config.hf_config.model_type == "deepseek_v41"
            assert not self.parallel_config.enable_dbo'''
assert s.count(needle)==1;s=s.replace(needle,replacement)
old='''        if forward_pass and not get_pp_group().is_first_rank:
            tensor_dict, comm_handles, comm_postprocess = (
                get_pp_group().irecv_tensor_dict(
                    all_gather_group=get_tp_group(),
                    all_gather_tensors=all_gather_tensors,
                )
            )
            assert tensor_dict is not None
            intermediate_tensors = AsyncIntermediateTensors(
                tensor_dict,
                comm_handles=comm_handles,
                comm_postprocess=comm_postprocess,
            )'''
new='''        if forward_pass and not get_pp_group().is_first_rank:
            def receive_intermediates():
                return get_pp_group().irecv_tensor_dict(
                    all_gather_group=get_tp_group(),
                    all_gather_tensors=all_gather_tensors,
                )

            if getattr(self, "_pp5_defer_receive", False) and not self.use_v2_model_runner:
                # V1 prepares attention metadata before _preprocess consumes
                # intermediates. Let all PP stages do this work concurrently.
                # The original receiver, handles and postprocessing still run
                # on this same thread/stream before tensors are accessed.
                intermediate_tensors = AsyncIntermediateTensors(
                    {}, comm_loader=receive_intermediates
                )
            else:
                tensor_dict, comm_handles, comm_postprocess = receive_intermediates()
                assert tensor_dict is not None
                intermediate_tensors = AsyncIntermediateTensors(
                    tensor_dict,
                    comm_handles=comm_handles,
                    comm_postprocess=comm_postprocess,
                )'''
assert s.count(old)==1;s=s.replace(old,new);ast.parse(s)
a.out.mkdir(parents=True,exist_ok=True)
(a.out/'gpu_worker.py').write_text(s,encoding='utf-8',newline='\n')
(a.out/'lazy-receive.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='a/vllm/v1/worker/gpu_worker.py',tofile='b/vllm/v1/worker/gpu_worker.py')),encoding='utf-8',newline='\n')
(a.out/'manifest.json').write_text(json.dumps({'original_sha256':hashlib.sha256(raw).hexdigest(),'candidate_sha256':hashlib.sha256(s.encode()).hexdigest(),'scope':'Defer V1 PP CPU metadata receive until intermediate tensor consumption; same receiver and stream. Default remains original path.'},indent=2))
print('PP5_LAZY_RECEIVE_CANDIDATE_PREPARED')
