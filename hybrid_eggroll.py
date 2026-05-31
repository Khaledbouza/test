"""
Hybrid EGGROLL: Transformer -> Latent Semantic Graph -> Graph Diffusion Repair -> Transformer Decoder
===================================================================================================
"""

from __future__ import annotations
import argparse
import time
import math
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, lax, vmap
from functools import partial

# --- Domain Constants ---
PERSON_NAMES   = ("john", "bob", "tom", "mary", "alice", "sue")
PERSON_GENDER  = jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.int32)
OBJECT_NAMES   = ("book", "key", "ball", "cup", "map", "coin")
ACTION_NAMES   = ("gave", "lent", "handed")
GENERIC_VERBS  = ("took", "held", "used", "kept", "dropped")
SLOT_NAMES    = ("giver", "recipient", "distractor", "object", "pronoun_bind")
BINDING_NAMES = ("giver", "recipient", "distractor")

class Config:
    def __init__(self, h):
        self.h = h
        self.L_enc = 1
        self.L_rep = 1
        self.L_dec = 1
        self.rep_steps = 4
        self.n_p = len(PERSON_NAMES)
        self.n_o = len(OBJECT_NAMES)
        self.n_a = len(ACTION_NAMES)
        self.n_g = len(GENERIC_VERBS)
        self.n_s = len(SLOT_NAMES)
        self.n_b = len(BINDING_NAMES)
        self.in_len = 9
        self.out_len = 13
        self.p_start = 0
        self.o_start = self.p_start + self.n_p
        self.a_start = self.o_start + self.n_o
        self.t_thinks = self.a_start + self.n_a
        self.t_the = self.t_thinks + 1
        self.t_he = self.t_the + 1
        self.t_she = self.t_he + 1
        self.g_start = self.t_she + 1
        self.t_bos = self.g_start + self.n_g
        self.t_eos = self.t_bos + 1
        self.t_is = self.t_eos + 1
        self.t_dot = self.t_is + 1
        self.vocab_size = self.t_dot + 1

    def __hash__(self):
        return hash((self.h, self.L_enc, self.L_rep, self.L_dec, self.rep_steps))
    def __eq__(self, other):
        return (self.h, self.L_enc, self.L_rep, self.L_dec, self.rep_steps) == (other.h, other.L_enc, other.L_rep, other.L_dec, other.rep_steps)

def rms_norm(x):
    return x / jnp.sqrt(jnp.mean(x*x, axis=-1, keepdims=True) + 1e-6)

# --- Param Management ---

def init_params(key, cfg):
    h = cfg.h
    keys = jax.random.split(key, 100)
    ki = 0
    def W(r, c):
        nonlocal ki; ki += 1
        return jax.random.normal(keys[ki], (r, c)) * (1.0 / math.sqrt(c))

    p = {
        "emb": jax.random.normal(keys[0], (cfg.vocab_size, h)) * 0.1,
        "pos_in": jax.random.normal(keys[1], (cfg.in_len, h)) * 0.1,
        "pos_out": jax.random.normal(keys[2], (cfg.out_len, h)) * 0.1,
        "slot_q": jax.random.normal(keys[3], (cfg.n_s, h)) * 0.1,
        "slot_t": jax.random.normal(keys[4], (cfg.n_s, h)) * 0.1,
        "p_emb": jax.random.normal(keys[5], (cfg.n_p, h)) * 0.1,
        "o_emb": jax.random.normal(keys[6], (cfg.n_o, h)) * 0.1,
        "b_emb": jax.random.normal(keys[7], (cfg.n_b, h)) * 0.1,
        "W_p": W(cfg.n_p, h), "b_p": jnp.zeros(cfg.n_p),
        "W_o": W(cfg.n_o, h), "b_o": jnp.zeros(cfg.n_o),
        "W_b": W(cfg.n_b, 4*h), "b_b": jnp.zeros(cfg.n_b),
        "W_out": W(cfg.vocab_size, h), "b_out": jnp.zeros(cfg.vocab_size),
    }
    for i in range(cfg.L_enc):
        p[f"e.{i}.qkv"] = W(3*h, h); p[f"e.{i}.o"] = W(h, h)
        p[f"e.{i}.f1"] = W(2*h, h); p[f"e.{i}.f2"] = W(h, 2*h)
    for i in range(cfg.L_rep):
        p[f"r.{i}.w"] = W(h, 5*h); p[f"r.{i}.b"] = jnp.zeros(h)
    for i in range(cfg.L_dec):
        p[f"d.{i}.sqkv"] = W(3*h, h); p[f"d.{i}.so"] = W(h, h)
        p[f"d.{i}.cq"] = W(h, h); p[f"d.{i}.ckv"] = W(2*h, h); p[f"d.{i}.co"] = W(h, h)
        p[f"d.{i}.f1"] = W(2*h, h); p[f"d.{i}.f2"] = W(h, 2*h)
    return p

# --- Forward Pass ---

@partial(jit, static_argnums=(3,))
def rollout(params, tokens, target_seq, cfg):
    scale = 1.0 / math.sqrt(cfg.h)

    # 1. Encoder
    h = params["emb"][tokens] + params["pos_in"][None, :, :]
    for i in range(cfg.L_enc):
        qkv = h @ params[f"e.{i}.qkv"].T
        q, k, v = jnp.split(qkv, 3, axis=-1)
        attn = jax.nn.softmax((q @ k.swapaxes(1,2)) * scale, -1)
        h = rms_norm(h + (attn @ v) @ params[f"e.{i}.o"].T)
        h = rms_norm(h + jnp.tanh(h @ params[f"e.{i}.f1"].T) @ params[f"e.{i}.f2"].T)
    enc = h

    # 2. Semantic Bottleneck
    sq = params["slot_q"] + params["slot_t"]
    attn = jax.nn.softmax((sq[None,:,:] @ enc.swapaxes(1,2)) * scale, -1)
    slots = rms_norm(attn @ enc + sq[None,:,:])

    # 3. Parallel Graph Repair
    def rep_step(s, _):
        pl = s[:, :3, :] @ params["W_p"].T + params["b_p"]
        ol = s[:, 3, :] @ params["W_o"].T + params["b_o"]
        bc = jnp.concatenate([s[:,4,:], s[:,0,:], s[:,1,:], s[:,2,:]], -1)
        bl = bc @ params["W_b"].T + params["b_b"]
        preds = lax.stop_gradient(jnp.stack([jnp.argmax(pl[:,0],-1), jnp.argmax(pl[:,1],-1), jnp.argmax(pl[:,2],-1), jnp.argmax(ol,-1), jnp.argmax(bl,-1)], 1))

        pe = params["p_emb"][preds[:,:3]]; oe = params["o_emb"][preds[:,3]]
        bh = jax.nn.one_hot(preds[:,4], cfg.n_b)
        se = jnp.sum(pe * bh[:,:,None], 1)
        be = params["b_emb"][preds[:,4]] + se
        disc = jnp.stack([pe[:,0], pe[:,1], pe[:,2], oe, be], 1)
        ss = jnp.sum(s[:,:3] * bh[:,:,None], 1)
        pm = jnp.stack([s[:,1], s[:,0], s[:,4], s[:,4], ss + s[:,3]], 1)

        gl = jnp.tile(jnp.mean(s, 1, keepdims=True), (1, cfg.n_s, 1))
        inp = jnp.concatenate([s, gl, pm, disc, jnp.tile(params["slot_t"][None,:,:], (s.shape[0],1,1))], -1)
        for i in range(cfg.L_rep):
            s = rms_norm(jnp.tanh(inp @ params[f"r.{i}.w"].T + params[f"r.{i}.b"]))
        return s, preds

    slots, _ = lax.scan(rep_step, slots, None, length=cfg.rep_steps)

    # 4. Transformer Decoder
    tin = target_seq[:, :-1]
    h = params["emb"][tin] + params["pos_out"][None, :tin.shape[1], :]
    mask = jnp.tril(jnp.ones((tin.shape[1], tin.shape[1])))
    for i in range(cfg.L_dec):
        qkv = h @ params[f"d.{i}.sqkv"].T; q,k,v = jnp.split(qkv, 3, axis=-1)
        attn = jax.nn.softmax(jnp.where(mask==0, -1e9, (q @ k.swapaxes(1,2))*scale), -1)
        h = rms_norm(h + (attn @ v) @ params[f"d.{i}.so"].T)
        cq = h @ params[f"d.{i}.cq"].T; ckv = slots @ params[f"d.{i}.ckv"].T; ck,cv = jnp.split(ckv, 2, axis=-1)
        cattn = jax.nn.softmax((cq @ ck.swapaxes(1,2))*scale, -1)
        h = rms_norm(h + (cattn @ cv) @ params[f"d.{i}.co"].T)
        h = rms_norm(h + jnp.tanh(h @ params[f"d.{i}.f1"].T) @ params[f"d.{i}.f2"].T)

    logits = h @ params["W_out"].T + params["b_out"]
    return logits

# --- Data Generation ---

def make_batch(key, n, cfg):
    ks = jax.random.split(key, 10)
    G = jax.random.randint(ks[0], (n,), 0, cfg.n_p)
    R = (G + jax.random.randint(ks[1], (n,), 1, cfg.n_p)) % cfg.n_p
    def gd(g, r, k):
        d = jax.random.randint(k, (), 0, cfg.n_p)
        return lax.while_loop(lambda x: (x==g)|(x==r), lambda _: jax.random.randint(jax.random.fold_in(k,1), (), 0, cfg.n_p), d)
    D = vmap(gd)(G, R, jax.random.split(ks[2], n))
    O = jax.random.randint(ks[3], (n,), 0, cfg.n_o)
    A = jax.random.randint(ks[4], (n,), 0, cfg.n_a)
    B = jax.random.randint(ks[5], (n,), 0, 2)
    Ref = jnp.where(B==0, G, R)
    P = jnp.where(PERSON_GENDER[Ref]==0, cfg.t_he, cfg.t_she)
    GV = jax.random.randint(ks[6], (n,), 0, cfg.n_g)

    tokens = jnp.stack([D, jnp.full(n, cfg.t_thinks), G, cfg.a_start+A, R, jnp.full(n, cfg.t_the), cfg.o_start+O, P, cfg.g_start+GV], 1)
    target = jnp.stack([jnp.full(n, cfg.t_bos), D, jnp.full(n, cfg.t_thinks), G, cfg.a_start+A, R, jnp.full(n, cfg.t_the), cfg.o_start+O, jnp.full(n, cfg.t_dot), P, jnp.full(n, cfg.t_is), Ref, jnp.full(n, cfg.t_eos)], 1)
    return tokens, target

# --- Training & ES ---

@partial(jit, static_argnums=(2,3))
def loss_fn(params, batch, cfg, backprop=True):
    logits = rollout(params, batch[0], batch[1], cfg)
    y = batch[1][:, 1:]; lp = jax.nn.log_softmax(logits, -1)
    t_lp = jnp.take_along_axis(lp, y[:,:,None], -1)[:,:,0]
    mask = jnp.ones_like(y, jnp.float32)
    if backprop: mask = mask.at[:, 10].set(0.0)
    loss = -jnp.sum(t_lp * mask) / jnp.sum(mask) / math.log(2)
    tok_acc = jnp.mean((jnp.argmax(logits, -1) == y) * mask) / jnp.mean(mask)
    ref_acc = jnp.mean(jnp.argmax(logits[:, 10], -1) == y[:, 10])
    return loss, (tok_acc, ref_acc)

@partial(jit, static_argnums=(3,))
def train_step(params, opt, key, cfg, step):
    batch = make_batch(key, 16, cfg)
    (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(params, batch, cfg, True)
    b1, b2, eps, lr = 0.9, 0.999, 1e-8, 1e-3
    m = jax.tree_util.tree_map(lambda m, g: b1*m + (1-b1)*g, opt["m"], g)
    v = jax.tree_util.tree_map(lambda v, g: b2*v + (1-b2)*(g*g), opt["v"], g)
    mh = jax.tree_util.tree_map(lambda x: x / (1 - b1**step), m)
    vh = jax.tree_util.tree_map(lambda x: x / (1 - b2**step), v)
    params = jax.tree_util.tree_map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, mh, vh)
    return params, {"m":m, "v":v}, l, aux

@partial(jit, static_argnums=(2,3,4,5))
def egg_step(params, key, cfg, sigma, lr, pop_size):
    batch = make_batch(key, 16, cfg)
    p_keys = jax.random.split(key, pop_size // 2)
    def get_diff(pk):
        def score(s):
            noise = jax.tree_util.tree_map(lambda x: jax.random.normal(pk, x.shape), params)
            pn = jax.tree_util.tree_map(lambda p, n: p + s * sigma * n, params, noise)
            _, aux = loss_fn(pn, batch, cfg, False)
            return aux[0] + aux[1]*2, noise
        s1, n = score(1.0); s2, _ = score(-1.0)
        return (s1-s2), n
    diffs, noises = vmap(get_diff)(p_keys)
    def update(p, n):
        grad = jnp.mean(n * diffs[(...,) + (None,)*(n.ndim-1)], 0)
        return p + (lr / sigma) * grad
    params = jax.tree_util.tree_map(update, params, noises)
    _, aux = loss_fn(params, batch, cfg, False)
    return params, aux

# --- Main ---

def token_name(t: int, cfg: Config) -> str:
    if cfg.p_start <= t < cfg.o_start:  return PERSON_NAMES[t]
    if cfg.o_start <= t < cfg.a_start:  return OBJECT_NAMES[t - cfg.o_start]
    if cfg.a_start <= t < cfg.t_thinks:    return ACTION_NAMES[t - cfg.a_start]
    if t == cfg.t_thinks:  return "thinks"
    if t == cfg.t_the:     return "the"
    if t == cfg.t_he:      return "he"
    if t == cfg.t_she:     return "she"
    if cfg.g_start <= t < cfg.t_bos:
        return GENERIC_VERBS[t - cfg.g_start]
    if t == cfg.t_bos:     return "[BOS]"
    if t == cfg.t_eos:     return "[EOS]"
    if t == cfg.t_is:      return "is"
    if t == cfg.t_dot:     return "."
    return f"?{t}"

def fmt_seq(tokens, cfg):
    return " ".join(token_name(int(t), cfg) for t in tokens)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h", type=int, default=8)
    p.add_argument("--bp", type=int, default=50)
    p.add_argument("--egg", type=int, default=50)
    args = p.parse_args()

    cfg = Config(args.h); key = jax.random.PRNGKey(0)
    params = init_params(key, cfg)
    opt = {"m": jax.tree_util.tree_map(jnp.zeros_like, params), "v": jax.tree_util.tree_map(jnp.zeros_like, params)}

    print(f"Hybrid EGGROLL: {args.h}h, {args.bp}bp, {args.egg}egg steps")

    if args.bp > 0:
        print("Phase 1: Backprop Training (Ref token masked)...")
        start = time.time()
        for i in range(1, args.bp + 1):
            params, opt, loss, aux = train_step(params, opt, jax.random.fold_in(key, i), cfg, i)
            if i % 10 == 0:
                print(f"  step {i:4d} | loss {loss:.3f} | tok {aux[0]:.3f} | ref {aux[1]:.3f}")
        print(f"  Done in {time.time()-start:.1f}s")

    if args.egg > 0:
        print("\nPhase 2: EGGROLL Optimization (Full sequence)...")
        start = time.time()
        for i in range(1, args.egg + 1):
            params, aux = egg_step(params, jax.random.fold_in(key, i+10000), cfg, 0.4, 0.05, 32)
            if i % 10 == 0:
                print(f"  step {i:4d} | tok {aux[0]:.3f} | ref {aux[1]:.3f}")
        print(f"  Done in {time.time()-start:.1f}s")

    tk, tg = make_batch(jax.random.fold_in(key, 999), 1, cfg)
    logits = rollout(params, tk, tg, cfg)
    print("\n--- Final Test ---")
    print(f"Input:  {fmt_seq(tk[0], cfg)}")
    print(f"Target: {fmt_seq(tg[0, 1:], cfg)}")
    print(f"Pred:   {fmt_seq(jnp.argmax(logits[0], -1), cfg)}")

if __name__ == "__main__":
    main()
