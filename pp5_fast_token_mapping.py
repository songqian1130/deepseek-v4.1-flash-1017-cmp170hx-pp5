"""GPU query-boundary lookup preserving device-side speculative boundaries."""
import torch
from vllm.triton_utils import triton,tl


@triton.jit
def _token_request_lookup(starts,out,NUM_REQS:tl.constexpr,NUM_MAPPED:tl.constexpr,NUM_TOKENS:tl.constexpr,BLOCK:tl.constexpr):
    token=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    lo=tl.full((BLOCK,),0,tl.int32)
    hi=tl.full((BLOCK,),NUM_REQS+1,tl.int32)
    # Upper bound handles repeated starts from zero-length padded requests.
    while tl.sum((lo<hi).to(tl.int32),0)>0:
        active=lo<hi
        mid=(lo+hi)//2
        boundary=tl.load(starts+mid,mask=active,other=2147483647)
        right=active&(boundary<=token)
        lo=tl.where(right,mid+1,lo)
        hi=tl.where(active&~right,mid,hi)
    req=tl.minimum(tl.maximum(lo-1,0),NUM_REQS-1)
    req=tl.where(token<NUM_MAPPED,req,0)
    tl.store(out+token,req,mask=token<NUM_TOKENS)


def token_to_req_indices(self,buffer):
    num_tokens=self.num_actual_tokens
    if self._token_to_req_indices_cache is not None:
        assert self._token_to_req_indices_cache.device==buffer.device
        assert self._token_to_req_indices_cache.dtype==torch.int32
        assert self._token_to_req_indices_cache.shape[0]>=num_tokens
        return self._token_to_req_indices_cache[:num_tokens]
    mapped=int(self.query_start_loc_cpu[-1])
    capacity=max(mapped,num_tokens)
    assert buffer.shape[0]>=capacity
    if capacity:
        _token_request_lookup[(triton.cdiv(capacity,128),)](
            self.query_start_loc,buffer,self.query_start_loc.numel()-1,mapped,capacity,128)
    self._token_to_req_indices_cache=buffer[:capacity]
    return self._token_to_req_indices_cache[:num_tokens]
