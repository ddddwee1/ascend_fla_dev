"""Isolated warm-cache measurements; no source changes or device work."""
from __future__ import annotations

import importlib
import time
from unittest.mock import patch


def measure(function, *, batches=20, iterations=10000):
    samples = []
    for _ in range(batches):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            function()
        samples.append((time.perf_counter_ns() - start) / iterations)
    return dict(iterations=iterations, batches=batches, ns_per_call=samples)


def cache_profile(api, pairs, block_dim):
    runtime = importlib.import_module('ascend_fla.runtime.compile')
    entry, compiled = pairs[0]
    key = (id(entry), 'a5', block_dim, (), 'cce')
    assert runtime._sig_memo[key][1] == compiled.signature
    assert runtime._process_cache[compiled.signature] is compiled
    original_memo, original_process = dict(runtime._sig_memo), dict(runtime._process_cache)

    def forbidden(*args, **kwargs):
        raise AssertionError('Warm cache attempted source extraction or signature hashing')

    results = dict(bindings={}, key_cardinality=5, empty_call=measure(lambda: None), public={}, scales=[])
    try:
        with patch.object(runtime, '_signature', forbidden), patch.object(runtime, '_kernel_source', forbidden):
            for name in ('_compiled', '_bf16_compiled'):
                function = getattr(api, name, None)
                if function is None:
                    continue
                before = function.cache_info()
                result = measure(lambda function=function: function(block_dim))
                after = function.cache_info()
                assert after.misses == before.misses
                assert after.hits - before.hits == result['batches'] * result['iterations']
                results['public'][name] = dict(**result, hit_delta=after.hits-before.hits, miss_delta=0)
            for size in (1, 64, 4096):
                runtime._sig_memo.clear()
                runtime._process_cache.clear()
                runtime._sig_memo[key] = original_memo[key]
                runtime._process_cache[compiled.signature] = compiled
                for index in range(size-1):
                    runtime._sig_memo[(-index-1, 'unrelated', block_dim, (), 'cce')] = (None, 'unrelated')
                    runtime._process_cache[f'gda05-unrelated-{index}'] = None

                def signature():
                    result = runtime._signature_memoized(entry, 'a5', block_dim, {}, 'cce')
                    assert result == compiled.signature

                def process():
                    result = runtime.compile_kernel(entry, device='a5', block_dim=block_dim, backend='cce')
                    assert result is compiled

                results['scales'].append(dict(entries=size, signature=measure(signature), process=measure(process)))
    finally:
        runtime._sig_memo.clear()
        runtime._sig_memo.update(original_memo)
        runtime._process_cache.clear()
        runtime._process_cache.update(original_process)
    results['cold_paths_rejected'] = True
    results['scope'] = ('Fixed empty bindings and a fixed five-field key. Dictionary lookup is average O(1); '
                        'the finite scale experiment does not prove asymptotic complexity. '
                        'Samples include loop, call and hit-identity assertion costs; no subtraction.')
    return results
