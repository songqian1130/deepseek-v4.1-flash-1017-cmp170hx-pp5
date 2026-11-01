"""Fixed, localhost-only experiment controls for PP5; no arbitrary-code RPC."""
import json
import time


class PP5PerformanceExtension:
    def pp5_perf_configure(self, config_json="{}"):
        config=json.loads(config_json)
        assert set(config)<= {'lazy_receive','timings','reset'}, config
        parallel=self.vllm_config.parallel_config
        if config.get('lazy_receive'):
            assert not self.use_v2_model_runner
            assert parallel.tensor_parallel_size==1 and parallel.pipeline_parallel_size==5
            assert not parallel.enable_dbo
        self._pp5_defer_receive=bool(config.get('lazy_receive',getattr(self,'_pp5_defer_receive',False)))
        if config.get('reset',True):self._pp5_perf_times={}
        enabled=bool(config.get('timings',False))
        runner=self.model_runner
        originals=getattr(self,'_pp5_timing_originals',None)
        if originals is None:
            originals={name:getattr(runner,name) for name in ('_build_attention_metadata','_preprocess','_model_forward')}
            self._pp5_timing_originals=originals
        for name,original in originals.items():
            if not enabled:
                setattr(runner,name,original)
                continue
            def timed(*args,_fn=original,_name=name,**kwargs):
                began=time.perf_counter()
                try:return _fn(*args,**kwargs)
                finally:self._pp5_perf_times.setdefault(_name,[]).append((time.perf_counter()-began)*1000)
            setattr(runner,name,timed)
        self._pp5_timings_enabled=enabled
        return self.pp5_perf_state()

    def pp5_perf_state(self):
        import statistics
        import vllm.v1.worker.gpu_worker as worker_module
        stats={}
        for name,values in getattr(self,'_pp5_perf_times',{}).items():
            values=sorted(values)
            stats[name]={'calls':len(values),'mean_ms':statistics.mean(values),'median_ms':statistics.median(values),'p95_ms':values[min(len(values)-1,int(len(values)*.95))]}
        return {'rank':self.rank,'lazy_receive':getattr(self,'_pp5_defer_receive',False),
                'timings_enabled':getattr(self,'_pp5_timings_enabled',False),
                'deferred_receives':getattr(worker_module,'PP5_DEFERRED_RECEIVE_COUNT',0),'timings':stats}
