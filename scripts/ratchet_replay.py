# Bare-torch replay of the sparse indexer's per-chunk allocation pattern during a
# long chunked prefill. No model. Measures torch reserved memory and MemAvailable
# as the prefix grows to L. Run twice: default allocator vs expandable_segments.
#
# Usage (in the recipe's venv, ~2 seconds per run, no weights needed):
#   python scripts/ratchet_replay.py
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/ratchet_replay.py
# Compare the RESULT line's peak_reserved / segments / num_alloc_retries between
# the two runs. See README.md "Long context: measured ceiling" for the numbers
# this reproduced on GB10.
import os, sys, torch, time
from vllm import envs
from vllm.v1.attention.backends.mla.indexer import split_indexer_prefill_chunks
L=int(os.environ.get("L","262144")); MNBT=int(os.environ.get("MNBT","2048"))
RATIO=int(os.environ.get("RATIO","4")); TOPK=2048
cap=envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB*1024*1024
ws=int(os.environ.get("WS", str(1<<22)))
def avail():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"): return int(l.split()[1])/1048576
print(f"MODE={os.environ.get('PYTORCH_CUDA_ALLOC_CONF','default')} L={L} MNBT={MNBT} RATIO={RATIO} cap_MB={cap>>20} ws={ws}", flush=True)
a0=avail(); dev="cuda"; peak=0; t0=time.time()
for step, pos in enumerate(range(0, L, MNBT)):
    q=min(MNBT, L-pos); seq=pos+q; N=seq//RATIO if RATIO>1 else seq
    chunks=split_indexer_prefill_chunks(torch.tensor([N]), torch.tensor([q]), ws, cap)
    live=[]
    token_to_seq=torch.empty(N, dtype=torch.int32, device=dev)
    for (rs, qs) in chunks:
        sub_m=qs.stop-qs.start
        ks=torch.empty(sub_m,dtype=torch.int32,device=dev); ke=torch.empty(sub_m,dtype=torch.int32,device=dev)
        logits=torch.empty((sub_m, N), dtype=torch.float32, device=dev)
        pool_topk=torch.empty((sub_m, TOPK//RATIO if RATIO>1 else TOPK), dtype=torch.int32, device=dev)
        live.append((ks,ke,logits,pool_topk))   # held until the step ends, like the loop body
    torch.cuda.synchronize()
    r=torch.cuda.memory_reserved()/2**30; peak=max(peak,r)
    del live, token_to_seq
    if step%16==0 or seq>=L:
        print(f"step={step:4d} seq={seq:7d} N={N:6d} subchunks={len(chunks):3d} reserved={r:6.2f}G peak={peak:6.2f}G avail_drop={a0-avail():6.2f}G t={time.time()-t0:5.1f}s", flush=True)
st=torch.cuda.memory_stats()
print(f"RESULT mode={os.environ.get('PYTORCH_CUDA_ALLOC_CONF','default')} peak_reserved={peak:.2f}G final_reserved={torch.cuda.memory_reserved()/2**30:.2f}G avail_drop={a0-avail():.2f}G num_alloc_retries={st.get('num_alloc_retries')} segments={st.get('segment.all.current')}", flush=True)
