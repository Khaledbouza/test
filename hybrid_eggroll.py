"""
Hybrid EGGROLL (CPU Optimized)
=============================
Transformer -> Latent Semantic Graph -> Graph Diffusion Repair -> Transformer Decoder
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

def rms_norm(x):
    return x / jnp.sqrt(jnp.mean(x*x, axis=-1, keepdims=True) + 1e-6)

# --- Params ---

def init_params(key, h, n_p, n_o, n_a, n_s, n_b, v_size, in_l, out_l, L_e, L_r, L_d):
    keys = jax.random.split(key, 100)
    ki = 0
    def W(r, c):
        nonlocal ki; ki += 1
        return jax.random.normal(keys[ki], (r, c)) * (1.0 / math.sqrt(c))

    p = {
        "emb": jax.random.normal(keys[0], (v_size, h)) * 0.1,
        "pi": jax.random.normal(keys[1], (in_l, h)) * 0.1,
        "po": jax.random.normal(keys[2], (out_l, h)) * 0.1,
        "sq": jax.random.normal(keys[3], (n_s, h)) * 0.1,
        "st": jax.random.normal(keys[4], (n_s, h)) * 0.1,
        "pe": jax.random.normal(keys[5], (n_p, h)) * 0.1,
        "oe": jax.random.normal(keys[6], (n_o, h)) * 0.1,
        "be": jax.random.normal(keys[7], (n_b, h)) * 0.1,
        "Wp": W(n_p, h), "bp": jnp.zeros(n_p),
        "Wo": W(n_o, h), "bo": jnp.zeros(n_o),
        "Wb": W(n_b, 4*h), "bb": jnp.zeros(n_b),
        "Wout": W(v_size, h), "bout": jnp.zeros(v_size),
    }
    for i in range(L_e):
        p[f"e.{i}.qkv"] = W(3*h, h); p[f"e.{i}.o"] = W(h, h)
        p[f"e.{i}.f1"] = W(2*h, h); p[f"e.{i}.f2"] = W(h, 2*h)
    for i in range(L_r):
        p[f"r.{i}.w"] = W(h, 5*h); p[f"r.{i}.b"] = jnp.zeros(h)
    for i in range(L_d):
        p[f"d.{i}.sqkv"] = W(3*h, h); p[f"d.{i}.so"] = W(h, h)
        p[f"d.{i}.cq"] = W(h, h); p[f"d.{i}.ckv"] = W(2*h, h); p[f"d.{i}.co"] = W(h, h)
        p[f"d.{i}.f1"] = W(2*h, h); p[f"d.{i}.f2"] = W(h, 2*h)
    return p

# --- Forward (Simplified) ---

@partial(jit, static_argnames=("h", "L_e", "L_r", "L_d", "n_s", "n_b", "r_steps"))
def rollout(params, tokens, target_seq, h, L_e, L_r, L_d, n_s, n_b, r_steps):
    scale = 1.0 / math.sqrt(h)
    x = params["emb"][tokens] + params["pi"][None, :, :]
    for i in range(L_e):
        qkv = x @ params[f"e.{i}.qkv"].T
        q, k, v = jnp.split(qkv, 3, -1)
        a = jax.nn.softmax((q @ k.swapaxes(1,2)) * scale, -1)
        x = rms_norm(x + (a @ v) @ params[f"e.{i}.o"].T)
        x = rms_norm(x + jnp.tanh(x @ params[f"e.{i}.f1"].T) @ params[f"e.{i}.f2"].T)

    sq = params["sq"] + params["st"]
    a = jax.nn.softmax((sq[None,:,:] @ x.swapaxes(1,2)) * scale, -1)
    s = rms_norm(a @ x + sq[None,:,:])

    def rep(s, _):
        pl = s[:, :3, :] @ params["Wp"].T + params["bp"]
        ol = s[:, 3, :] @ params["Wo"].T + params["bo"]
        bc = jnp.concatenate([s[:,4], s[:,0], s[:,1], s[:,2]], -1)
        bl = bc @ params["Wb"].T + params["bb"]
        pr = lax.stop_gradient(jnp.stack([jnp.argmax(pl[:,0],-1), jnp.argmax(pl[:,1],-1), jnp.argmax(pl[:,2],-1), jnp.argmax(ol,-1), jnp.argmax(bl,-1)], 1))

        pe = params["pe"][pr[:,:3]]; oe = params["oe"][pr[:,3]]; bh = jax.nn.one_hot(pr[:,4], n_b)
        se = jnp.sum(pe * bh[:,:,None], 1); be = params["be"][pr[:,4]] + se; disc = jnp.stack([pe[:,0], pe[:,1], pe[:,2], oe, be], 1)
        ss = jnp.sum(s[:,:3] * bh[:,:,None], 1); pm = jnp.stack([s[:,1], s[:,0], s[:,4], s[:,4], ss + s[:,3]], 1)

        gl = jnp.tile(jnp.mean(s, 1, keepdims=True), (1, n_s, 1))
        inp = jnp.concatenate([s, gl, pm, disc, jnp.tile(params["st"][None,:,:], (s.shape[0],1,1))], -1)
        for i in range(L_r):
            s = rms_norm(jnp.tanh(inp @ params[f"r.{i}.w"].T + params[f"r.{i}.b"]))
        return s, pr

    s, _ = lax.scan(rep, s, None, length=r_steps)

    tin = target_seq[:, :-1]
    y = params["emb"][tin] + params["po"][None, :tin.shape[1], :]
    m = jnp.tril(jnp.ones((tin.shape[1], tin.shape[1])))
    for i in range(L_d):
        qkv = y @ params[f"d.{i}.sqkv"].T; q,k,v = jnp.split(qkv, 3, -1)
        a = jax.nn.softmax(jnp.where(m==0, -1e9, (q @ k.swapaxes(1,2))*scale), -1)
        y = rms_norm(y + (a @ v) @ params[f"d.{i}.so"].T)
        cq = y @ params[f"d.{i}.cq"].T; ckv = s @ params[f"d.{i}.ckv"].T; ck,cv = jnp.split(ckv, 2, -1)
        ca = jax.nn.softmax((cq @ ck.swapaxes(1,2))*scale, -1); y = rms_norm(y + (ca @ cv) @ params[f"d.{i}.co"].T)
        y = rms_norm(y + jnp.tanh(y @ params[f"d.{i}.f1"].T) @ params[f"d.{i}.f2"].T)
    return y @ params["Wout"].T + params["bout"]

# --- Utils ---

def get_data(key, n, h, n_p, n_o, n_a, n_g, v_s, p_start, o_start, a_start, t_thinks, t_the, t_dot, t_is, t_he, t_she, t_bos, t_eos):
    ks = jax.random.split(key, 10)
    G = jax.random.randint(ks[0], (n,), 0, n_p)
    R = (G + jax.random.randint(ks[1], (n,), 1, n_p)) % n_p
    D = (R + jax.random.randint(ks[2], (n,), 1, n_p)) % n_p
    O = jax.random.randint(ks[3], (n,), 0, n_o); A = jax.random.randint(ks[4], (n,), 0, n_a); B = jax.random.randint(ks[5], (n,), 0, 2)
    Ref = jnp.where(B==0, G, R); P = jnp.where(PERSON_GENDER[Ref]==0, t_he, t_she); GV = jax.random.randint(ks[6], (n,), 0, n_g)

    tokens = jnp.stack([D, jnp.full(n, t_thinks), G, a_start+A, R, jnp.full(n, t_the), o_start+O, P, g_start+GV], 1)
    target = jnp.stack([jnp.full(n, t_bos), D, jnp.full(n, t_thinks), G, a_start+A, R, jnp.full(n, t_the), o_start+O, jnp.full(n, t_dot), P, jnp.full(n, t_is), Ref, jnp.full(n, t_eos)], 1)
    return tokens, target

@partial(jit, static_argnames=("h", "L_e", "L_r", "L_d", "n_s", "n_b", "r_steps", "backprop"))
def loss_fn(params, batch, h, L_e, L_r, L_d, n_s, n_b, r_steps, backprop=True):
    logits = rollout(params, batch[0], batch[1], h, L_e, L_r, L_d, n_s, n_b, r_steps)
    y = batch[1][:, 1:]; lp = jax.nn.log_softmax(logits, -1)
    t_lp = jnp.take_along_axis(lp, y[:,:,None], -1)[:,:,0]
    mask = jnp.ones_like(y, jnp.float32)
    if backprop: mask = mask.at[:, 10].set(0.0)
    loss = -jnp.sum(t_lp * mask) / jnp.sum(mask) / math.log(2)
    t_acc = jnp.mean((jnp.argmax(logits, -1) == y) * mask) / jnp.mean(mask)
    r_acc = jnp.mean(jnp.argmax(logits[:, 10], -1) == y[:, 10])
    return loss, (t_acc, r_acc)

@partial(jit, static_argnames=("h", "L_e", "L_r", "L_d", "n_s", "n_b", "r_steps", "lr"))
def train_step(params, opt, key, data_args, h, L_e, L_r, L_d, n_s, n_b, r_steps, lr, step):
    batch = get_data(key, 8, h, *data_args)
    (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(params, batch, h, L_e, L_r, L_d, n_s, n_b, r_steps, True)
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = jax.tree_util.tree_map(lambda m, g: b1*m + (1-b1)*g, opt["m"], g)
    v = jax.tree_util.tree_map(lambda v, g: b2*v + (1-b2)*(g*g), opt["v"], g)
    mh = jax.tree_util.tree_map(lambda x: x / (1 - b1**step), m)
    vh = jax.tree_util.tree_map(lambda x: x / (1 - b2**step), v)
    params = jax.tree_util.tree_map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, mh, vh)
    return params, {"m":m, "v":v}, l, aux

@partial(jit, static_argnames=("h", "L_e", "L_r", "L_d", "n_s", "n_b", "r_steps", "sigma", "lr", "pop"))
def egg_step(params, key, data_args, h, L_e, L_r, L_d, n_s, n_b, r_steps, sigma, lr, pop):
    batch = get_data(key, 8, h, *data_args)
    p_keys = jax.random.split(key, pop // 2)
    def diff_fn(pk):
        def score(s):
            n = jax.tree_util.tree_map(lambda x: jax.random.normal(pk, x.shape), params)
            pn = jax.tree_util.tree_map(lambda p, n: p + s * sigma * n, params, n)
            _, aux = loss_fn(pn, batch, h, L_e, L_r, L_d, n_s, n_b, r_steps, False)
            return aux[0] + aux[1]*2, n
        s1, n = score(1.0); s2, _ = score(-1.0)
        return (s1-s2), n
    diffs, noises = vmap(diff_fn)(p_keys)
    def upd(p, n):
        grad = jnp.mean(n * diffs[(...,) + (None,)*(n.ndim-1)], 0)
        return p + (lr / sigma) * grad
    params = jax.tree_util.tree_map(upd, params, noises)
    _, aux = loss_fn(params, batch, h, L_e, L_r, L_d, n_s, n_b, r_steps, False)
    return params, aux

# --- Config & Main ---

p_start, o_start, a_start, g_start = 0, 6, 12, 18
t_thinks, t_the, t_dot, t_is, t_he, t_she, t_bos, t_eos = 15, 16, 26, 25, 17, 18, 23, 24
data_args = (6, 6, 3, 5, 27, p_start, o_start, a_start, t_thinks, t_the, t_dot, t_is, t_he, t_she, t_bos, t_eos)

def main():
    h = 4; L_e, L_r, L_d = 1, 1, 1; r_steps = 2; n_s, n_b = 5, 3
    key = jax.random.PRNGKey(0); params = init_params(key, h, 6, 6, 3, n_s, n_b, 27, 9, 13, L_e, L_r, L_d)
    opt = {"m": jax.tree_util.tree_map(jnp.zeros_like, params), "v": jax.tree_util.tree_map(jnp.zeros_like, params)}

    print("Starting CPU Optimized Hybrid EGGROLL...")
    for i in range(1, 11):
        params, opt, loss, aux = train_step(params, opt, jax.random.fold_in(key, i), data_args, h, L_e, L_r, L_d, n_s, n_b, r_steps, 1e-3, i)
        print(f"BP {i:2d} | loss {loss:.3f} | tok {aux[0]:.3f}")

    for i in range(1, 11):
        params, aux = egg_step(params, jax.random.fold_in(key, i+100), data_args, h, L_e, L_r, L_d, n_s, n_b, r_steps, 0.4, 0.05, 8)
        print(f"EGG {i:2d} | tok {aux[0]:.3f} | ref {aux[1]:.3f}")
    print("Success.")

if __name__ == "__main__":
    main()
