"""
Hybrid EGGROLL: Parallel Semantic Repair
=========================================
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

# --- Constants ---
PERSON_NAMES   = ("john", "bob", "tom", "mary", "alice", "sue")
PERSON_GENDER  = jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.int32)
OBJECT_NAMES   = ("book", "key", "ball", "cup", "map", "coin")
ACTION_NAMES   = ("gave", "lent", "handed")
GENERIC_VERBS  = ("took", "held", "used", "kept", "dropped")
SLOT_NAMES    = ("giver", "recipient", "distractor", "object", "pronoun_bind")
BINDING_NAMES = ("giver", "recipient", "distractor")

class ModelConfig:
    def __init__(self, h, L_enc, L_rep, L_dec, rep_steps):
        self.h = h
        self.L_enc = L_enc
        self.L_rep = L_rep
        self.L_dec = L_dec
        self.rep_steps = rep_steps
        self.n_p, self.n_o, self.n_a = len(PERSON_NAMES), len(OBJECT_NAMES), len(ACTION_NAMES)
        self.n_g, self.n_s, self.n_b = len(GENERIC_VERBS), len(SLOT_NAMES), len(BINDING_NAMES)
        self.in_len, self.out_len = 9, 13
        self.p_start, self.o_start, self.a_start = 0, 6, 12
        self.t_thinks, self.t_the = 15, 16
        self.t_he, self.t_she = 17, 18
        self.g_start = 19
        self.t_bos, self.t_eos, self.t_is, self.t_dot = 24, 25, 26, 27
        self.v_size = 28
    def __hash__(self): return hash((self.h, self.L_enc, self.L_rep, self.L_dec, self.rep_steps))
    def __eq__(self, o): return hash(self) == hash(o)

def rms_norm(x): return x / jnp.sqrt(jnp.mean(x*x, axis=-1, keepdims=True) + 1e-6)

# --- Model ---
def init_params(key, cfg):
    h = cfg.h; keys = jax.random.split(key, 100); ki = 0
    def W(r, c):
        nonlocal ki; ki += 1; return jax.random.normal(keys[ki], (r, c)) * (1.0 / math.sqrt(c))
    p = {
        "emb": jax.random.normal(keys[0], (cfg.v_size, h)) * 0.1,
        "pi": jax.random.normal(keys[1], (cfg.in_len, h)) * 0.1,
        "po": jax.random.normal(keys[2], (cfg.out_len, h)) * 0.1,
        "sq": jax.random.normal(keys[3], (cfg.n_s, h)) * 0.1,
        "st": jax.random.normal(keys[4], (cfg.n_s, h)) * 0.1,
        "pe": jax.random.normal(keys[5], (cfg.n_p, h)) * 0.1,
        "oe": jax.random.normal(keys[6], (cfg.n_o, h)) * 0.1,
        "be": jax.random.normal(keys[7], (cfg.n_b, h)) * 0.1,
        "Wp": W(cfg.n_p, h), "bp": jnp.zeros(cfg.n_p),
        "Wo": W(cfg.n_o, h), "bo": jnp.zeros(cfg.n_o),
        "Wb": W(cfg.n_b, 4*h), "bb": jnp.zeros(cfg.n_b),
        "Wout": W(cfg.v_size, h), "bout": jnp.zeros(cfg.v_size),
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

@partial(jit, static_argnames=("cfg",))
def rollout(params, tokens, target_seq, cfg):
    scale = 1.0 / math.sqrt(cfg.h)
    h = params["emb"][tokens] + params["pi"][None, :, :]
    for i in range(cfg.L_enc):
        qkv = h @ params[f"e.{i}.qkv"].T; q, k, v = jnp.split(qkv, 3, -1)
        a = jax.nn.softmax((q @ k.swapaxes(1,2)) * scale, -1)
        h = rms_norm(h + (a @ v) @ params[f"e.{i}.o"].T)
        h = rms_norm(h + jnp.tanh(h @ params[f"e.{i}.f1"].T) @ params[f"e.{i}.f2"].T)
    enc = h; sq = params["sq"] + params["st"]; a = jax.nn.softmax((sq[None,:,:] @ enc.swapaxes(1,2)) * scale, -1)
    slots = rms_norm(a @ enc + sq[None,:,:])
    def rep_step(s, _):
        pl = s[:, :3, :] @ params["Wp"].T + params["bp"]; ol = s[:, 3, :] @ params["Wo"].T + params["bo"]
        bc = jnp.concatenate([s[:,4], s[:,0], s[:,1], s[:,2]], -1); bl = bc @ params["Wb"].T + params["bb"]
        preds = lax.stop_gradient(jnp.stack([jnp.argmax(pl[:,0],-1), jnp.argmax(pl[:,1],-1), jnp.argmax(pl[:,2],-1), jnp.argmax(ol,-1), jnp.argmax(bl,-1)], 1))
        pe = params["pe"][preds[:,:3]]; oe = params["oe"][preds[:,3]]; bh = jax.nn.one_hot(preds[:,4], cfg.n_b)
        se = jnp.sum(pe * bh[:,:,None], 1); be = params["be"][preds[:,4]] + se; disc = jnp.stack([pe[:,0], pe[:,1], pe[:,2], oe, be], 1)
        ss = jnp.sum(s[:,:3] * bh[:,:,None], 1); pm = jnp.stack([s[:,1], s[:,0], s[:,4], s[:,4], ss + s[:,3]], 1)
        gl = jnp.tile(jnp.mean(s, 1, keepdims=True), (1, cfg.n_s, 1)); curr_s = s
        for i in range(cfg.L_rep):
            inp = jnp.concatenate([curr_s, gl, pm, disc, jnp.tile(params["st"][None,:,:], (s.shape[0],1,1))], -1)
            curr_s = rms_norm(jnp.tanh(inp @ params[f"r.{i}.w"].T + params[f"r.{i}.b"]))
        return curr_s, preds
    slots, _ = lax.scan(rep_step, slots, None, length=cfg.rep_steps); tin = target_seq[:, :-1]; h = params["emb"][tin] + params["po"][None, :tin.shape[1], :]
    mask = jnp.tril(jnp.ones((tin.shape[1], tin.shape[1])))
    for i in range(cfg.L_dec):
        qkv = h @ params[f"d.{i}.sqkv"].T; q,k,v = jnp.split(qkv, 3, -1)
        a = jax.nn.softmax(jnp.where(mask==0, -1e9, (q @ k.swapaxes(1,2))*scale), -1); h = rms_norm(h + (a @ v) @ params[f"d.{i}.so"].T)
        cq = h @ params[f"d.{i}.cq"].T; ckv = slots @ params[f"d.{i}.ckv"].T; ck,cv = jnp.split(ckv, 2, -1)
        ca = jax.nn.softmax((cq @ ck.swapaxes(1,2))*scale, -1); h = rms_norm(h + (ca @ cv) @ params[f"d.{i}.co"].T)
        h = rms_norm(h + jnp.tanh(h @ params[f"d.{i}.f1"].T) @ params[f"d.{i}.f2"].T)
    return h @ params["Wout"].T + params["bout"]

# --- Training ---
def make_batch(key, n, cfg):
    ks = jax.random.split(key, 10); G = jax.random.randint(ks[0], (n,), 0, cfg.n_p); R = (G + jax.random.randint(ks[1], (n,), 1, cfg.n_p)) % cfg.n_p
    D = (R + jax.random.randint(ks[2], (n,), 1, cfg.n_p)) % cfg.n_p; O = jax.random.randint(ks[3], (n,), 0, cfg.n_o)
    A = jax.random.randint(ks[4], (n,), 0, cfg.n_a); B = jax.random.randint(ks[5], (n,), 0, 2); Ref = jnp.where(B==0, G, R)
    P = jnp.where(PERSON_GENDER[Ref]==0, cfg.t_he, cfg.t_she); GV = jax.random.randint(ks[6], (n,), 0, cfg.n_g)
    tokens = jnp.stack([D, jnp.full(n, cfg.t_thinks), G, cfg.a_start+A, R, jnp.full(n, cfg.t_the), cfg.o_start+O, P, cfg.g_start+GV], 1)
    target = jnp.stack([jnp.full(n, cfg.t_bos), D, jnp.full(n, cfg.t_thinks), G, cfg.a_start+A, R, jnp.full(n, cfg.t_the), cfg.o_start+O, jnp.full(n, cfg.t_dot), P, jnp.full(n, cfg.t_is), Ref, jnp.full(n, cfg.t_eos)], 1)
    return tokens, target

@partial(jit, static_argnames=("cfg", "backprop"))
def loss_fn(params, batch, cfg, backprop=True):
    logits = rollout(params, batch[0], batch[1], cfg); y = batch[1][:, 1:]; lp = jax.nn.log_softmax(logits, -1)
    t_lp = jnp.take_along_axis(lp, y[:,:,None], -1)[:,:,0]; mask = jnp.ones_like(y, jnp.float32)
    if backprop: mask = mask.at[:, 10].set(0.0)
    loss = -jnp.sum(t_lp * mask) / jnp.sum(mask) / math.log(2); t_acc = jnp.mean((jnp.argmax(logits, -1) == y) * mask) / jnp.mean(mask); r_acc = jnp.mean(jnp.argmax(logits[:, 10], -1) == y[:, 10])
    return loss, (t_acc, r_acc)

@partial(jit, static_argnames=("cfg", "lr"))
def train_step(params, opt, key, cfg, lr, step):
    batch = make_batch(key, 16, cfg); (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(params, batch, cfg, True)
    b1, b2, eps = 0.9, 0.999, 1e-8; m = jax.tree_util.tree_map(lambda m, g: b1*m + (1-b1)*g, opt["m"], g)
    v = jax.tree_util.tree_map(lambda v, g: b2*v + (1-b2)*(g*g), opt["v"], g)
    mh = jax.tree_util.tree_map(lambda x: x / (1 - b1**step), m); vh = jax.tree_util.tree_map(lambda x: x / (1 - b2**step), v)
    params = jax.tree_util.tree_map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, mh, vh)
    return params, {"m":m, "v":v}, l, aux

@partial(jit, static_argnames=("cfg", "sigma", "lr", "pop"))
def egg_step(params, key, cfg, sigma, lr, pop):
    batch = make_batch(key, 16, cfg); p_keys = jax.random.split(key, pop // 2)
    def df(pk):
        def sc(s):
            noise = jax.tree_util.tree_map(lambda x: jax.random.normal(pk, x.shape), params)
            pn = jax.tree_util.tree_map(lambda p, n: p + s * sigma * n, params, noise); _, ax = loss_fn(pn, batch, cfg, False); return ax[0] + ax[1]*2, noise
        s1, n = sc(1.0); s2, _ = sc(-1.0); return (s1-s2), n
    dfs, ns = vmap(df)(p_keys)
    def up(p, n): grad = jnp.mean(n * dfs[(...,) + (None,)*(n.ndim-1)], 0); return p + (lr / sigma) * grad
    params = jax.tree_util.tree_map(up, params, ns); _, ax = loss_fn(params, batch, cfg, False); return params, ax

def main():
    p = argparse.ArgumentParser(); p.add_argument("--h", type=int, default=16); p.add_argument("--bp", type=int, default=200); p.add_argument("--egg", type=int, default=200); p.add_argument("--cpu", action="store_true"); args = p.parse_args()
    if args.cpu: jax.config.update('jax_platform_name', 'cpu')
    cfg = ModelConfig(args.h, 1, 1, 1, 4); key = jax.random.PRNGKey(0); params = init_params(key, cfg); opt = {"m": jax.tree_util.tree_map(jnp.zeros_like, params), "v": jax.tree_util.tree_map(jnp.zeros_like, params)}
    print(f"Hybrid EGGROLL: {args.h}h. Platform: {jax.default_backend()}"); print("Training Phase 1 (Backprop)..."); start = time.time()
    for i in range(1, args.bp + 1):
        params, opt, loss, aux = train_step(params, opt, jax.random.fold_in(key, i), cfg, 1e-3, i)
        if i % 50 == 0 or i == 1: print(f"  BP {i:4d} | loss {loss:5.3f} | tok {aux[0]:.3f} | ref {aux[1]:.3f} | {time.time()-start:.1f}s")
    if args.egg > 0:
        print("\nTraining Phase 2 (EGGROLL)..."); start = time.time()
        for i in range(1, args.egg + 1):
            params, aux = egg_step(params, jax.random.fold_in(key, i+1000), cfg, 0.4, 0.05, 32)
            if i % 50 == 0 or i == 1: print(f"  EGG {i:4d} | tok {aux[0]:.3f} | ref {aux[1]:.3f} | {time.time()-start:.1f}s")
    def tn(t, cfg):
        if cfg.p_start <= t < cfg.o_start: return PERSON_NAMES[t];
        if cfg.o_start <= t < cfg.a_start: return OBJECT_NAMES[t-cfg.o_start];
        if cfg.a_start <= t < cfg.t_thinks: return ACTION_NAMES[t-cfg.a_start]
        if t == cfg.t_thinks: return "thinks";
        if t == cfg.t_the: return "the";
        if t == cfg.t_he: return "he";
        if t == cfg.t_she: return "she"
        if cfg.g_start <= t < cfg.t_bos: return GENERIC_VERBS[t-cfg.g_start]
        if t == cfg.t_bos: return "[BOS]";
        if t == cfg.t_eos: return "[EOS]";
        if t == cfg.t_is: return "is";
        if t == cfg.t_dot: return "."
        return f"?{t}"
    tk, tg = make_batch(jax.random.fold_in(key, 999), 1, cfg); logits = rollout(params, tk, tg, cfg); print("\nFinal Result:")
    print(f"  Input:  {' '.join(tn(int(t), cfg) for t in tk[0])}"); print(f"  Target: {' '.join(tn(int(t), cfg) for t in tg[0, 1:])}")
    print(f"  Pred:   {' '.join(tn(int(t), cfg) for t in jnp.argmax(logits[0], -1))}"); print("\nHybrid reasoning functional.")

if __name__ == "__main__": main()
