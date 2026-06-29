#!/usr/bin/env python3
# Numpy reference forward pass for SmolLM2-135M-Instruct (Llama arch) to localize
# the engine's forward-pass bug. Reads the GGUF directly, dequantizes, runs a
# standard Llama forward (RMSNorm + GQA + RoPE + SwiGLU), prints layer-0
# intermediate stats and the prefill-first token.
import struct, numpy as np, sys

PATH = "/Users/liang/inference-engine/SmolLM2-135M-Instruct-Q4_0.gguf"
# Exact prompt tokens emitted by the engine for "Hello, who are you?" (ChatML):
TOKENS = [1,9690,198,2683,359,253,5356,5646,11173,3365,3511,308,34519,28,7018,411,407,19712,8182,2,198,1,4093,198,19556,28,617,359,346,47,2,198,1,520,9531,198]

raw = open(PATH,"rb").read()
p = 0
def rd(n):
    global p
    v = raw[p:p+n]; p += n; return v
def ru32():
    global p; v = struct.unpack_from('<I', raw, p)[0]; p += 4; return v
def ru64():
    global p; v = struct.unpack_from('<Q', raw, p)[0]; p += 8; return v
def rstr():
    global p
    n = ru64(); s = raw[p:p+n]; p += n; return s.decode('utf-8','replace')

assert rd(4) == b'GGUF'
ver = ru32(); n_tensor = ru64(); n_kv = ru64()
align = 32
T = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,8:'s',9:'a',10:8,11:8,12:8}
def rval(t):
    if t in (0,1): v=ru32() if False else rd(1); return None
    if t==4: return struct.unpack_from('<I',rd(4))[0] if False else None
def skip_val(t):
    global p
    if t==8: n=ru64(); p+=n
    elif t==9:
        et=ru32(); n=ru64()
        for _ in range(n):
            if et==8: m=ru64(); p+=m
            elif et==9: skip_val(9)
            else: p+={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[et]
    else: p+={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[t]

cfg={}
for _ in range(n_kv):
    k=rstr(); t=ru32()
    if t==8: v=rstr()
    elif t==4: v=struct.unpack_from('<I',rd(4))[0]
    elif t==6: v=struct.unpack_from('<f',rd(4))[0]
    elif t in (0,1,7): v=rd(1)[0]
    elif t==9: v=None; skip_val(9)
    else: v=None; skip_val(t)
    cfg[k]=v
arch=cfg.get('general.architecture','llama')
def C(s): return cfg.get(f'{arch}.{s}', cfg.get(s))
n_layers=int(C('block_count')); dim=int(C('embedding_length')); ffn=int(C('feed_forward_length'))
n_heads=int(C('attention.head_count')); n_kv=int(C('attention.head_count_kv') or n_heads)
head_dim=dim//n_heads; base=float(C('rope.freq_base') or 10000.0); eps=float(C('attention.layer_norm_rms_epsilon') or 1e-5)
vocab=len(TOKENS)  # placeholder, real vocab from token_embd shape
print(f"arch={arch} L={n_layers} dim={dim} ffn={ffn} H={n_heads} KV={n_kv} hd={head_dim} base={base} eps={eps}")

# tensor infos
infos={}
for _ in range(n_tensor):
    name=rstr(); nd=ru32(); dims=[ru64() for _ in range(nd)]; tt=ru32(); off=ru64()
    infos[name]=(dims,tt,off)
data_off=p
data_off=(data_off+align-1)//align*align

def tdata(name):
    dims,tt,off=infos[name]
    base_ptr=data_off+off
    return dims,tt,raw[base_ptr:]

def fp16_arr(b):  # b: bytes -> float32 via numpy float16
    return np.frombuffer(b, dtype='<f2').astype(np.float32)

def dequant(name):
    dims,tt,b=tdata(name)
    nelem=int(np.prod(dims))
    if tt==0:   # F32
        a=np.frombuffer(b[:nelem*4],dtype='<f4').astype(np.float32)
    elif tt==1: # F16
        a=np.frombuffer(b[:nelem*2],dtype='<f2').astype(np.float32)
    elif tt==8: # Q8_0: blocks of 34 (fp16 d + int8[32])
        nb=nelem//32; out=np.empty(nelem,dtype=np.float32)
        rec=np.frombuffer(b[:nb*34],dtype=np.dtype([('d','<f2'),('q',np.int8,(32,))]))
        d=rec['d'].astype(np.float32)
        out=(rec['q'].astype(np.float32)*d[:,None]).reshape(-1)
        a=out
    elif tt==3: # Q4_1: blocks of 24 (fp16 d + fp16 m + uint8[16]); value = d*q + m, q unsigned 0..15
        nb=nelem//32
        rec=np.frombuffer(b[:nb*20],dtype=np.dtype([('d','<f2'),('m','<f2'),('q',np.uint8,(16,))]))
        d=rec['d'].astype(np.float32); m=rec['m'].astype(np.float32)
        q=rec['q']
        lo=(q & 0x0F).astype(np.float32); hi=(q >> 4).astype(np.float32)
        inter=np.empty((nb,32),dtype=np.float32)
        inter[:,:16]=lo*d[:,None]+m[:,None]   # blocked: low nibbles first
        inter[:,16:]=hi*d[:,None]+m[:,None]   # high nibbles second
        a=inter.reshape(-1)
    elif tt==2: # Q4_0: ggml convention — value=(nibble-8)*d, blocked layout (low nibbles [0..15], high [16..31])
        nb=nelem//32; out=np.empty(nelem,dtype=np.float32)
        rec=np.frombuffer(b[:nb*18],dtype=np.dtype([('d','<u2'),('q',np.uint8,(16,))]))
        d=rec['d'].view(np.float16).astype(np.float32)
        qi=rec['q'].astype(np.int32)   # int to avoid uint8 underflow on subtract
        lo=(qi & 0x0F) - 8             # -8..7
        hi=(qi >> 4) - 8
        inter=np.empty((nb,32),dtype=np.float32)
        inter[:,:16]=lo.astype(np.float32)*d[:,None]
        inter[:,16:]=hi.astype(np.float32)*d[:,None]
        a=inter.reshape(-1)
    else:
        raise ValueError(f"type {tt} for {name}")
    # GGUF dims are [ne0 innermost, ne1, ...] -> numpy shape reversed
    return a.reshape(list(reversed(dims)))

emb=dequant('token_embd.weight')   # [vocab, dim]
vocab=emb.shape[0]
Wq=[dequant(f'blk.{l}.attn_q.weight') for l in range(n_layers)]
Wk=[dequant(f'blk.{l}.attn_k.weight') for l in range(n_layers)]
Wv=[dequant(f'blk.{l}.attn_v.weight') for l in range(n_layers)]
Wo=[dequant(f'blk.{l}.attn_output.weight') for l in range(n_layers)]
Wg=[dequant(f'blk.{l}.ffn_gate.weight') for l in range(n_layers)]
Wu=[dequant(f'blk.{l}.ffn_up.weight') for l in range(n_layers)]
Wd=[dequant(f'blk.{l}.ffn_down.weight') for l in range(n_layers)]
ga=[dequant(f'blk.{l}.attn_norm.weight') for l in range(n_layers)]
gm=[dequant(f'blk.{l}.ffn_norm.weight') for l in range(n_layers)]
gf=dequant('output_norm.weight')

def rmsnorm(x,g): return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+eps)*g

half=head_dim//2
inv=np.power(base, -np.arange(0,half,dtype=np.float32)*2/head_dim)  # [half]

def rope(x, n_h):  # x: [L, n_h*hd]
    L=x.shape[0]
    x=x.reshape(L,n_h,head_dim)
    pos=np.arange(L,dtype=np.float32)
    th=np.outer(pos,inv)  # [L,half]
    c=np.cos(th)[:,None,:]; s=np.sin(th)[:,None,:]
    x1=x[...,:half]; x2=x[...,half:]
    o=np.empty_like(x)
    o[...,:half]=x1*c-x2*s
    o[...,half:]=x2*c+x1*s
    return o.reshape(L,n_h*head_dim)

def forward(tokens):
    x = emb[np.array(tokens, dtype=np.int64)].astype(np.float32)  # [L, dim]
    dbg = (len(tokens) == 1)
    if dbg: print(f"  embed[0][:6]={x[0,:6]}  max|x|={np.abs(x).max():.4f}")
    for l in range(n_layers):
        h = rmsnorm(x, ga[l])
        q = (h @ Wq[l].T).reshape(-1, n_heads, head_dim)
        k = (h @ Wk[l].T).reshape(-1, n_kv, head_dim)
        v = (h @ Wv[l].T).reshape(-1, n_kv, head_dim)
        L = x.shape[0]
        pos = np.arange(L, dtype=np.float32); th = np.outer(pos, inv)
        c = np.cos(th)[:, None, :]; s = np.sin(th)[:, None, :]
        def rot(t, nh):
            t = t.reshape(L, nh, head_dim); t1 = t[..., :half]; t2 = t[..., half:]
            o = np.empty_like(t); o[..., :half] = t1*c - t2*s; o[..., half:] = t2*c + t1*s
            return o
        q = rot(q, n_heads); k = rot(k, n_kv)
        gs = n_heads // n_kv
        out = np.zeros((L, n_heads, head_dim), dtype=np.float32)
        scale = 1.0/np.sqrt(head_dim)
        for i in range(L):
            for hh in range(n_heads):
                kh = hh // gs
                sc = (q[i, hh] @ k[:i+1, kh].T) * scale
                sc = sc - sc.max(); e = np.exp(sc); e = e/e.sum()
                out[i, hh] = e @ v[:i+1, kh]
        attn = out.reshape(L, n_heads*head_dim)
        x = x + attn @ Wo[l].T
        h2 = rmsnorm(x, gm[l])
        gg = h2 @ Wg[l].T; uu = h2 @ Wu[l].T
        ff = (gg/(1.0+np.exp(-gg)))*uu
        x = x + ff @ Wd[l].T
        projo = attn @ Wo[l].T; dc = ff @ Wd[l].T
        if dbg and l < 4:
            r=lambda a: float(np.sqrt(np.mean(a*a)))
            print(f"  L{l}: rms(x)={r(x):.4f} rms(q)={r(q.reshape(L,-1)):.4f} rms(v)={r(v.reshape(L,-1)):.4f} rms(attn)={r(attn):.4f} rms(attn@Wo)={r(projo):.4f} | rms(gate)={r(gg):.4f} rms(ff)={r(ff):.4f} rms(ff@Wd)={r(dc):.4f}")
    xf = rmsnorm(x, gf)
    return xf @ emb.T, x, xf

lg, _, _ = forward(TOKENS); last = lg[-1]
top = np.argsort(last)[::-1][:5]
print("PREFILL-FIRST argmax token:", int(top[0]), "(numpy reference)")
print("top5:", [(int(t), round(float(last[t]),3)) for t in top])

# Single-token comparison vs llama.cpp
for tok in [19556, 4093, 198]:
    slg, sx, sxf = forward([tok])
    slrow = slg[-1] if slg.ndim>1 else slg
    st = np.argsort(slrow)[::-1][:5]
    print(f"numpy single-token({tok}) top5:", [(int(t), round(float(slrow[t]),3)) for t in st])
    if tok==19556:
        print(f"numpy xf[:8]={np.round(sxf[0,:8],4)} rms(xf)={float(np.sqrt(np.mean(sxf*sxf))):.4f}")
        print(f"(llama.cpp get_embeddings[:8]=[-0.3069 -0.0174 0.0063 0.5143 0.6374 -0.119 0.018 0.1771])")

