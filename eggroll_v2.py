""" 
Language Slot Repair EGGROLL — v2: The Sweet Spot 
=================================================== 
 
What changed from v1 and WHY: 
 
PROBLEM in v1:  backprop achieved bind_acc=1.000 during pretraining. 
  The reason verb ("found"=giver, "requested"=recipient) was a direct 
  single-token lexical cue.  CE loss supervised binding logits directly. 
  EGGROLL had nothing left to improve. 
 
THREE TARGETED FIXES: 
  1. binding_loss_weight = 0.0 
     Binding is REMOVED from CE loss entirely. 
     Backprop cannot directly push binding logits. 
     Only EGGROLL sees the binding score. 
 
  2. Reason verbs merged into one uninformative pool. 
     "found/requested/noticed" all come from the same vocab region. 
     No single token reveals whether pronoun = giver or recipient. 
     The model must use SLOT STRUCTURE to resolve binding. 
 
  3. same_gender_prob=1.0, distractor_match_prob=1.0 
     All three people always share the same gender. 
     Pronoun gender is completely uninformative. 
     Only positional slot identity matters. 
 
EXPECTED BEHAVIOUR: 
  - Backprop pretrain: slot_acc high, bind_acc ~0.33-0.45 (near random) 
  - CE control:        bind_acc stays stuck (no direct cue or loss) 
  - Shuffled ES:       bind_acc stays stuck (broken signal) 
  - EGGROLL:           bind_acc climbs (only method that can optimize 
                       through the discrete slot→binding path) 
 
BONUS FIX: 
  - slot_persistence_loss added to CE loss: penalises slots for 
    changing state between repair iterations.  Targets stable=0.406 
    from v1 — want stable > 0.80 before EGGROLL starts. 
 
Sentence structure (9 tokens, no reason verb): 
  pos 0: distractor person 
  pos 1: "thinks" 
  pos 2: giver person 
  pos 3: action verb 
  pos 4: recipient person 
  pos 5: "the" 
  pos 6: object 
  pos 7: pronoun  (he/she — always same gender as all 3 people) 
  pos 8: generic_verb  (uninformative: can't tell giver vs recipient) 
 
The ONLY resolving cue is structural: 
  who is at position 2 (giver slot) vs position 4 (recipient slot)? 
  Soft attention smears these; discrete slot binding locks onto one. 
""" 
 
from __future__ import annotations 
 
import argparse 
import json 
import math 
import time 
from dataclasses import asdict, dataclass 
from pathlib import Path 
 
import numpy as np 
 
PERSON_NAMES   = ("john", "bob", "tom", "mary", "alice", "sue") 
PERSON_GENDER  = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32) 
OBJECT_NAMES   = ("book", "key", "ball", "cup", "map", "coin") 
ACTION_NAMES   = ("gave", "lent", "handed") 
GENERIC_VERBS  = ("took", "held", "used", "kept", "dropped")  # uninformative 
 
SLOT_NAMES    = ("giver", "recipient", "distractor", "object", "pronoun_bind") 
BINDING_NAMES = ("giver", "recipient", "distractor") 
 
 
@dataclass 
class Config: 
    hidden_size:         int = 64 
    transformer_layers:  int = 2 
    repair_layers:       int = 2 
 
    @property 
    def n_people(self):     return len(PERSON_NAMES) 
    @property 
    def n_objects(self):    return len(OBJECT_NAMES) 
    @property 
    def n_actions(self):    return len(ACTION_NAMES) 
    @property 
    def n_generic(self):    return len(GENERIC_VERBS) 
    @property 
    def n_slots(self):      return len(SLOT_NAMES) 
    @property 
    def n_bindings(self):   return len(BINDING_NAMES) 
    @property 
    def sent_len(self):     return 9 
 
    # Token ID layout 
    @property 
    def person_start(self): return 0 
    @property 
    def object_start(self): return self.person_start + self.n_people 
    @property 
    def action_start(self): return self.object_start + self.n_objects 
    @property 
    def tok_thinks(self):   return self.action_start + self.n_actions 
    @property 
    def tok_the(self):      return self.tok_thinks + 1 
    @property 
    def tok_he(self):       return self.tok_the + 1 
    @property 
    def tok_she(self):      return self.tok_he + 1 
    @property 
    def generic_start(self): return self.tok_she + 1 
    @property 
    def vocab_size(self):   return self.generic_start + self.n_generic 
 
 
def stable_salt(text: str) -> int: 
    v = 2166136261 
    for b in text.encode(): 
        v ^= b; v = (v * 16777619) & 0x7FFFFFFF 
    return v 
 
 
def save_npz(path, arrays, metadata): 
    path.parent.mkdir(parents=True, exist_ok=True) 
    payload = dict(arrays) 
    payload["metadata_json"] = np.array(json.dumps(metadata, indent=2), dtype=np.str_) 
    np.savez_compressed(path, **payload) 
 
 
def token_name(t: int, cfg: Config) -> str: 
    if cfg.person_start <= t < cfg.object_start:  return PERSON_NAMES[t] 
    if cfg.object_start <= t < cfg.action_start:  return OBJECT_NAMES[t - cfg.object_start] 
    if cfg.action_start <= t < cfg.tok_thinks:    return ACTION_NAMES[t - cfg.action_start] 
    if t == cfg.tok_thinks:  return "thinks" 
    if t == cfg.tok_the:     return "the" 
    if t == cfg.tok_he:      return "he" 
    if t == cfg.tok_she:     return "she" 
    if cfg.generic_start <= t < cfg.vocab_size: 
        return GENERIC_VERBS[t - cfg.generic_start] 
    return f"?{t}" 
 
 
def fmt_sent(tokens, cfg): 
    return " ".join(token_name(int(t), cfg) for t in tokens) 
 
def fmt_graph(labels, cfg): 
    giver  = PERSON_NAMES[int(labels[0])] 
    recip  = PERSON_NAMES[int(labels[1])] 
    distr  = PERSON_NAMES[int(labels[2])] 
    obj    = OBJECT_NAMES[int(labels[3])] 
    bind   = BINDING_NAMES[int(labels[4])] 
    ref    = (giver, recip, distr)[int(labels[4])] 
    return f"CONTEXT({distr}); GIVE({giver},{recip},{obj}); BIND({bind}->{ref})" 
 
 
def main(): 
    p = argparse.ArgumentParser( 
        description="Language Slot Repair EGGROLL v2 — binding removed from CE loss") 
 
    p.add_argument("--seed",                type=int,   default=0) 
    p.add_argument("--hidden-size",         type=int,   default=64) 
    p.add_argument("--transformer-layers",  type=int,   default=2) 
    p.add_argument("--repair-layers",       type=int,   default=2) 
    p.add_argument("--repair-steps",        type=int,   default=6) 
    p.add_argument("--batch-size",          type=int,   default=256) 
    p.add_argument("--valid-size",          type=int,   default=2048) 
 
    # ── The key dials ────────────────────────────────────────────────────── 
    p.add_argument("--binding-loss-weight", type=float, default=0.0, 
                   help="0.0 = backprop CANNOT supervise binding (the whole point)") 
    p.add_argument("--same-gender-prob",    type=float, default=1.0, 
                   help="1.0 = gender always uninformative") 
    p.add_argument("--distractor-match-prob", type=float, default=1.0, 
                   help="1.0 = distractor always same gender as referent") 
    p.add_argument("--persistence-weight",  type=float, default=0.10, 
                   help="Slot persistence loss — stabilises repair oscillation") 
 
    p.add_argument("--slot-noise",          type=float, default=0.15) 
    p.add_argument("--slot-drop-prob",      type=float, default=0.10) 
    p.add_argument("--aux-initial-loss",    type=float, default=0.10) 
    p.add_argument("--distractor-loss-weight", type=float, default=0.50) 
 
    # Score bonuses 
    p.add_argument("--exact-bonus",         type=float, default=2.0) 
    p.add_argument("--binding-bonus",       type=float, default=2.0) 
    p.add_argument("--referent-bonus",      type=float, default=1.0) 
    p.add_argument("--repair-gain-bonus",   type=float, default=0.5) 
    p.add_argument("--slot-bonus",          type=float, default=0.25) 
    p.add_argument("--stability-bonus",     type=float, default=0.25) 
 
    # Training 
    p.add_argument("--bp-steps",            type=int,   default=2000) 
    p.add_argument("--bp-lr",               type=float, default=1e-3) 
    p.add_argument("--min-quant-scale",     type=float, default=1e-3) 
 
    # EGGROLL 
    p.add_argument("--egg-steps",           type=int,   default=2000) 
    p.add_argument("--population",          type=int,   default=128) 
    p.add_argument("--rank",                type=int,   default=4) 
    p.add_argument("--sigma",               type=float, default=0.50) 
    p.add_argument("--egg-lr",              type=float, default=0.06) 
    p.add_argument("--round-egg-each-step", action="store_true") 
 
    # Controls 
    p.add_argument("--control-steps",       type=int,   default=500) 
    p.add_argument("--control-lr",          type=float, default=1e-3) 
    p.add_argument("--control-mode", 
                   choices=["both","ce","shuffle","none"], default="both") 
 
    p.add_argument("--print-every",         type=int,   default=100) 
    p.add_argument("--sample-every",        type=int,   default=500) 
    p.add_argument("--checkpoint-out",      default="lang_slot_v2.npz") 
    args = p.parse_args() 
 
    if args.population % 2 != 0: 
        raise ValueError("--population must be even") 
 
    import jax 
    import jax.numpy as jnp 
 
    cfg        = Config(args.hidden_size, args.transformer_layers, args.repair_layers) 
    pair_count = args.population // 2 
    sqrt_rank  = math.sqrt(args.rank) 
    egg_scale  = args.egg_lr / (max(pair_count, 1) * max(args.sigma, 1e-6)) 
 
    person_gender = jnp.asarray(PERSON_GENDER) 
    people_ids    = jnp.arange(cfg.n_people) 
 
    devs = ", ".join(f"{d.platform}:{d.device_kind}" for d in jax.devices()) 
    print(f"jax backend: {jax.default_backend()}  devices=[{devs}]") 
    print(f"v2 config: vocab={cfg.vocab_size} sent_len={cfg.sent_len} " 
          f"slots={cfg.n_slots} hidden={cfg.hidden_size}") 
    print(f"KEY SETTINGS: binding_loss_weight={args.binding_loss_weight}  " 
          f"same_gender_prob={args.same_gender_prob}  " 
          f"distractor_match_prob={args.distractor_match_prob}") 
    print(f"  → backprop CANNOT solve binding; EGGROLL must") 
 
    # ── Parameters ──────────────────────────────────────────────────────── 
 
    def init_params(key): 
        n_keys = 8 * cfg.transformer_layers + 6 * cfg.repair_layers + 14 
        keys   = iter(jax.random.split(key, n_keys)) 
        h      = cfg.hidden_size 
 
        def W(k, r, c): 
            return jax.random.normal(k, (r, c), jnp.float32) * (1.0 / math.sqrt(c)) 
 
        params = { 
            "token_emb":       jax.random.normal(next(keys), (cfg.vocab_size, h), jnp.float32) * 0.20, 
            "pos_emb":         jax.random.normal(next(keys), (cfg.sent_len,   h), jnp.float32) * 0.20, 
            "slot_query":      jax.random.normal(next(keys), (cfg.n_slots,    h), jnp.float32) * 0.20, 
            "slot_type_emb":   jax.random.normal(next(keys), (cfg.n_slots,    h), jnp.float32) * 0.20, 
            "person_label_emb":jax.random.normal(next(keys), (cfg.n_people,   h), jnp.float32) * 0.20, 
            "object_label_emb":jax.random.normal(next(keys), (cfg.n_objects,  h), jnp.float32) * 0.20, 
            "bind_label_emb":  jax.random.normal(next(keys), (cfg.n_bindings, h), jnp.float32) * 0.20, 
            # Output heads 
            "W_person": W(next(keys), cfg.n_people,   h), 
            "b_person": jnp.zeros((cfg.n_people,)), 
            "W_object": W(next(keys), cfg.n_objects,  h), 
            "b_object": jnp.zeros((cfg.n_objects,)), 
            # Binding uses 4 slots: pronoun_slot + giver + recipient + distractor 
            "W_bind":   W(next(keys), cfg.n_bindings, 4 * h), 
            "b_bind":   jnp.zeros((cfg.n_bindings,)), 
        } 
        for l in range(cfg.transformer_layers): 
            px = f"tx.{l}." 
            params[px+"W_q"]   = W(next(keys), h, h) 
            params[px+"W_k"]   = W(next(keys), h, h) 
            params[px+"W_v"]   = W(next(keys), h, h) 
            params[px+"W_o"]   = W(next(keys), h, h) 
            params[px+"W_ff1"] = W(next(keys), 2*h, h) 
            params[px+"b_ff1"] = jnp.zeros((2*h,)) 
            params[px+"W_ff2"] = W(next(keys), h, 2*h) 
            params[px+"b_ff2"] = jnp.zeros((h,)) 
        for l in range(cfg.repair_layers): 
            px = f"repair.{l}." 
            params[px+"W_self"]   = W(next(keys), h, h) 
            params[px+"W_global"] = W(next(keys), h, h) 
            params[px+"W_pair"]   = W(next(keys), h, h) 
            params[px+"W_disc"]   = W(next(keys), h, h) 
            params[px+"W_type"]   = W(next(keys), h, h) 
            params[px+"b"]        = jnp.zeros((h,)) 
        return params 
 
    def count_params(params): 
        return int(sum(x.size for x in jax.tree_util.tree_leaves(params))) 
 
    # ── Encoder ─────────────────────────────────────────────────────────── 
 
    def rms_norm(x): 
        return x / jnp.sqrt(jnp.mean(x*x, axis=-1, keepdims=True) + 1e-6) 
 
    def transformer_encode(params, tokens): 
        # tokens: (B, L) 
        h = params["token_emb"][tokens] + params["pos_emb"][None, :, :] 
        scale = 1.0 / math.sqrt(cfg.hidden_size) 
        for l in range(cfg.transformer_layers): 
            px = f"tx.{l}." 
            q = h @ params[px+"W_q"].T 
            k = h @ params[px+"W_k"].T 
            v = h @ params[px+"W_v"].T 
            attn = jax.nn.softmax((q @ jnp.swapaxes(k,1,2)) * scale, axis=-1) 
            h = rms_norm(h + (attn @ v) @ params[px+"W_o"].T) 
            ff = jnp.tanh(h @ params[px+"W_ff1"].T + params[px+"b_ff1"]) 
            h = rms_norm(h + ff @ params[px+"W_ff2"].T + params[px+"b_ff2"]) 
        return h 
 
    def initial_slots(params, tokens): 
        enc     = transformer_encode(params, tokens)           # (B, L, H) 
        queries = params["slot_query"] + params["slot_type_emb"]  # (S, H) 
        logits  = jnp.einsum("sh,bth->bst", queries, enc) / math.sqrt(cfg.hidden_size) 
        attn    = jax.nn.softmax(logits, axis=-1)              # (B, S, L) 
        slots   = jnp.einsum("bst,bth->bsh", attn, enc) + queries[None,:,:] 
        return rms_norm(slots) 
 
    # ── Output heads ────────────────────────────────────────────────────── 
 
    def output_heads(params, slots): 
        # slots: (B, n_slots, H) 
        # Person logits: first 3 slots → person identities 
        person_logits = slots[:, :3, :] @ params["W_person"].T + params["b_person"]  # (B,3,n_people) 
        # Object logit: slot 3 
        object_logits = slots[:, 3, :] @ params["W_object"].T + params["b_object"]   # (B,n_objects) 
        # Binding: concat pronoun_slot(4) + giver(0) + recip(1) + distr(2) 
        bind_ctx = jnp.concatenate( 
            [slots[:,4,:], slots[:,0,:], slots[:,1,:], slots[:,2,:]], axis=-1)        # (B,4H) 
        bind_logits = bind_ctx @ params["W_bind"].T + params["b_bind"]               # (B,n_bindings) 
        return person_logits, object_logits, bind_logits 
 
    def hard_preds(logits): 
        person_logits, object_logits, bind_logits = logits 
        giver    = jnp.argmax(person_logits[:,0,:], axis=-1) 
        recip    = jnp.argmax(person_logits[:,1,:], axis=-1) 
        distr    = jnp.argmax(person_logits[:,2,:], axis=-1) 
        obj      = jnp.argmax(object_logits, axis=-1) 
        bind     = jnp.argmax(bind_logits,   axis=-1) 
        return jnp.stack([giver, recip, distr, obj, bind], axis=1).astype(jnp.int32) 
 
    # ── Repair ──────────────────────────────────────────────────────────── 
 
    def discrete_embs(params, slots, preds): 
        giver_e = params["person_label_emb"][preds[:,0]] 
        recip_e = params["person_label_emb"][preds[:,1]] 
        distr_e = params["person_label_emb"][preds[:,2]] 
        obj_e   = params["object_label_emb"][preds[:,3]] 
        entity_stack = jnp.stack([giver_e, recip_e, distr_e], axis=1)   # (B,3,H) 
        bind_oh = jax.nn.one_hot(preds[:,4], cfg.n_bindings).astype(jnp.float32) 
        selected = jnp.sum(entity_stack * bind_oh[:,:,None], axis=1)    # (B,H) 
        bind_e  = params["bind_label_emb"][preds[:,4]] + selected 
        return jnp.stack([giver_e, recip_e, distr_e, obj_e, bind_e], axis=1) 
 
    def repair_update(params, slots, preds): 
        disc       = discrete_embs(params, slots, preds)           # (B,S,H) 
        global_msg = jnp.mean(slots, axis=1, keepdims=True) 
        # Peer message: each slot looks at its "partner" 
        bind_oh    = jax.nn.one_hot(preds[:,4], cfg.n_bindings).astype(jnp.float32) 
        selected   = jnp.sum(slots[:,:3,:] * bind_oh[:,:,None], axis=1)  # (B,H) 
        pair_msg   = jnp.stack([ 
            slots[:,1,:],              # giver looks at recipient 
            slots[:,0,:],              # recipient looks at giver 
            slots[:,4,:],              # distractor looks at pronoun 
            slots[:,4,:],              # object looks at pronoun 
            selected + slots[:,3,:],   # pronoun looks at bound entity + object 
        ], axis=1)                                                    # (B,S,H) 
        type_msg   = params["slot_type_emb"]                         # (S,H) 
        h = slots 
        for l in range(cfg.repair_layers): 
            px = f"repair.{l}." 
            h = jnp.tanh( 
                h              @ params[px+"W_self"].T 
                + global_msg   @ params[px+"W_global"].T 
                + pair_msg     @ params[px+"W_pair"].T 
                + disc         @ params[px+"W_disc"].T 
                + type_msg[None,:,:] @ params[px+"W_type"].T 
                + params[px+"b"] 
            ) 
        return rms_norm(h) 
 
    def repair_once(params, slots): 
        logits = output_heads(params, slots) 
        preds  = jax.lax.stop_gradient(hard_preds(logits)) 
        new_slots = repair_update(params, slots, preds) 
        return new_slots, preds 
 
    # ── Rollout ─────────────────────────────────────────────────────────── 
 
    def rollout(params, tokens, slot_noise, slot_drop): 
        slots0  = initial_slots(params, tokens) 
        # Corrupt initial slots (simulate partial/noisy initialisation) 
        fallback = params["slot_query"] + params["slot_type_emb"] 
        slots0   = slots0 + args.slot_noise * slot_noise 
        slots0   = jnp.where(slot_drop[:,:,None], fallback[None,:,:], slots0) 
        slots0   = rms_norm(slots0) 
 
        init_logits = output_heads(params, slots0) 
 
        def step(carry, _): 
            slots, prev_slots = carry 
            new_slots, preds  = repair_once(params, slots) 
            return (new_slots, slots), preds 
 
        (final_slots, pre_final_slots), history = jax.lax.scan( 
            step, (slots0, slots0), None, length=args.repair_steps) 
 
        final_logits = output_heads(params, final_slots) 
        preds        = hard_preds(final_logits) 
 
        # One extra step to measure stability 
        extra_slots, _ = repair_once(params, final_slots) 
        extra_preds    = hard_preds(output_heads(params, extra_slots)) 
 
        return final_logits, init_logits, preds, history, extra_preds, final_slots, pre_final_slots 
 
    # ── Data generation ─────────────────────────────────────────────────── 
 
    def make_batch(key, batch_size): 
        k_giver, k_off, k_opp, k_same, k_dm, k_dc, k_obj, k_act, k_gv, k_sn, k_sd = jax.random.split(key, 11) 
 
        giver      = jax.random.randint(k_giver, (batch_size,), 0, cfg.n_people, jnp.int32) 
        giver_g    = person_gender[giver] 
        giver_pos  = giver % 3 
        same_off   = jax.random.randint(k_off, (batch_size,), 1, 3, jnp.int32) 
        same_recip = giver_g * 3 + ((giver_pos + same_off) % 3) 
        opp_recip  = (1-giver_g)*3 + jax.random.randint(k_opp,(batch_size,),0,3,jnp.int32) 
        same_g     = jax.random.uniform(k_same,(batch_size,)) < args.same_gender_prob 
        recip      = jnp.where(same_g, same_recip, opp_recip).astype(jnp.int32) 
 
        lo, hi     = jnp.minimum(giver,recip), jnp.maximum(giver,recip) 
        distr_raw  = jax.random.randint(k_dm,(batch_size,),0,cfg.n_people-2,jnp.int32) 
        distr      = distr_raw + (distr_raw>=lo).astype(jnp.int32) 
        distr      = distr    + (distr    >=hi).astype(jnp.int32) 
 
        # binding label: 0=giver referred to, 1=recipient referred to 
        bind_label = jax.random.randint(k_dc, (batch_size,), 0, 2, jnp.int32) 
        target     = jnp.where(bind_label==0, giver, recip) 
        target_g   = person_gender[target] 
 
        # Match distractor gender to pronoun referent (makes gender useless) 
        match_scores = jax.random.uniform(k_dc, (batch_size, cfg.n_people)) 
        match_valid  = ( 
            (person_gender[None,:] == target_g[:,None]) & 
            (people_ids[None,:]    != giver[:,None]) & 
            (people_ids[None,:]    != recip[:,None]) 
        ) 
        matched_distr = jnp.argmax( 
            jnp.where(match_valid, match_scores, -1.0), axis=1).astype(jnp.int32) 
        use_match = jax.random.uniform(k_dc,(batch_size,)) < args.distractor_match_prob 
        distr     = jnp.where(use_match, matched_distr, distr).astype(jnp.int32) 
 
        # All same gender → pronoun always matches all 3 people, completely uninformative 
        pronoun = jnp.where(person_gender[target]==0, cfg.tok_he, cfg.tok_she).astype(jnp.int32) 
 
        obj    = jax.random.randint(k_obj, (batch_size,), 0, cfg.n_objects, jnp.int32) 
        action = jax.random.randint(k_act, (batch_size,), 0, cfg.n_actions, jnp.int32) 
        gen_v  = jax.random.randint(k_gv,  (batch_size,), 0, cfg.n_generic, jnp.int32) 
 
        # Sentence: [distractor thinks giver action recip the obj pronoun generic_verb] 
        tokens = jnp.stack([ 
            cfg.person_start + distr, 
            jnp.full((batch_size,), cfg.tok_thinks, jnp.int32), 
            cfg.person_start + giver, 
            cfg.action_start + action, 
            cfg.person_start + recip, 
            jnp.full((batch_size,), cfg.tok_the, jnp.int32), 
            cfg.object_start + obj, 
            pronoun, 
            cfg.generic_start + gen_v, 
        ], axis=1) 
 
        labels = jnp.stack([giver, recip, distr, obj, bind_label], axis=1) 
 
        slot_noise = jax.random.normal(k_sn, (batch_size, cfg.n_slots, cfg.hidden_size), jnp.float32) 
        slot_drop  = jax.random.uniform(k_sd, (batch_size, cfg.n_slots)) < args.slot_drop_prob 
        return tokens, labels, slot_noise, slot_drop 
 
    # ── Losses ──────────────────────────────────────────────────────────── 
 
    def ce_loss(params, batch): 
        tokens, labels, slot_noise, slot_drop = batch 
        final_logits, init_logits, preds, _, _, final_slots, pre_final_slots = rollout(params, tokens, slot_noise, slot_drop) 
 
        def ce_one(logit, label): 
            lp  = logit - jax.nn.logsumexp(logit, axis=-1, keepdims=True) 
            return -jnp.mean( 
                jnp.take_along_axis(lp, label[...,None], axis=-1)[...,0] 
            ) / math.log(2.0) 
 
        person_logits, object_logits, bind_logits = final_logits 
        init_pl, init_ol, init_bl                 = init_logits 
 
        giver_loss = ce_one(person_logits[:,0,:], labels[:,0]) 
        recip_loss = ce_one(person_logits[:,1,:], labels[:,1]) 
        distr_loss = ce_one(person_logits[:,2,:], labels[:,2]) 
        obj_loss   = ce_one(object_logits,        labels[:,3]) 
        # NOTE: bind_loss intentionally excluded (binding_loss_weight=0 default) 
        bind_loss  = ce_one(bind_logits, labels[:,4]) 
 
        person_loss = (giver_loss + recip_loss 
                       + args.distractor_loss_weight * distr_loss 
                       ) / (2.0 + args.distractor_loss_weight) 
 
        main_loss = (person_loss + obj_loss 
                     + args.binding_loss_weight * bind_loss 
                     ) / (2.0 + args.binding_loss_weight) 
 
        # Initial auxiliary loss 
        init_p_loss = ( 
            ce_one(init_pl[:,0,:], labels[:,0]) + 
            ce_one(init_pl[:,1,:], labels[:,1]) + 
            args.distractor_loss_weight * ce_one(init_pl[:,2,:], labels[:,2]) 
        ) / (2.0 + args.distractor_loss_weight) 
        init_loss = (init_p_loss + ce_one(init_ol, labels[:,3])) / 2.0 
 
        # Slot persistence loss — penalise slots changing between last two steps 
        # Encourages convergence, fixes stable=0.406 from v1 
        persistence_loss = jnp.mean(jnp.sum((final_slots - pre_final_slots)**2, axis=-1)) 
 
        loss     = main_loss + args.aux_initial_loss * init_loss + args.persistence_weight * persistence_loss 
        slot_acc = jnp.mean(preds == labels) 
        bind_acc = jnp.mean(preds[:,4] == labels[:,4]) 
        return loss, slot_acc, bind_acc 
 
    def repair_score(params, batch): 
        tokens, labels, slot_noise, slot_drop = batch 
        res = rollout(params, tokens, slot_noise, slot_drop)
        final_logits, init_logits, preds, _, extra_preds, _, _ = res
 
        init_preds  = hard_preds(init_logits) 
        init_slot_acc  = jnp.mean(init_preds == labels, axis=1) 
        init_bind_acc  = (init_preds[:,4] == labels[:,4]).astype(jnp.float32) 
 
        slot_acc    = jnp.mean(preds == labels, axis=1) 
        exact       = jnp.all(preds == labels, axis=1).astype(jnp.float32) 
        bind_acc    = (preds[:,4] == labels[:,4]).astype(jnp.float32) 
        stable      = jnp.mean(preds == extra_preds, axis=1) 
        repair_gain = slot_acc - init_slot_acc 
 
        # Referent accuracy: did the bound entity match the correct person? 
        bind_oh        = jax.nn.one_hot(preds[:,4], cfg.n_bindings).astype(jnp.int32) 
        tgt_oh         = jax.nn.one_hot(labels[:,4], cfg.n_bindings).astype(jnp.int32) 
        pred_ref       = jnp.sum(preds[:,:3]  * bind_oh, axis=1) 
        tgt_ref        = jnp.sum(labels[:,:3] * tgt_oh,  axis=1) 
        referent_acc   = (pred_ref == tgt_ref).astype(jnp.float32) 
 
        score = ( 
            args.binding_bonus   * bind_acc 
            + args.referent_bonus  * referent_acc 
            + args.exact_bonus     * exact 
            + args.repair_gain_bonus * repair_gain 
            + args.slot_bonus      * slot_acc 
            + args.stability_bonus * stable 
        ) 
        return score, bind_acc, referent_acc, exact, stable, slot_acc, init_bind_acc, repair_gain 
 
    # ── Quantisation ────────────────────────────────────────────────────── 
 
    def quantize(params): 
        scales = jax.tree_util.tree_map( 
            lambda x: jnp.maximum(jnp.max(jnp.abs(x))/127.0, args.min_quant_scale), params) 
        q = jax.tree_util.tree_map( 
            lambda x,s: jnp.clip(jnp.rint(x/s),-127.0,127.0).astype(jnp.float32), params, scales) 
        return q, scales 
 
    def dequant(q, s): 
        return jax.tree_util.tree_map(lambda qi,si: qi*si, q, s) 
 
    # ── EGGROLL ─────────────────────────────────────────────────────────── 
 
    template = init_params(jax.random.PRNGKey(args.seed)) 
    names  = list(template.keys()) 
    shapes = {n: template[n].shape for n in names} 
    salts  = {n: stable_salt(n) for n in names} 
 
    def eps_for_param(key, name): 
        k = jax.random.fold_in(key, salts[name]) 
        s = shapes[name] 
        if len(s) == 2: 
            ka, kb = jax.random.split(k) 
            a = jax.random.normal(ka, (s[0], args.rank), jnp.float32) 
            b = jax.random.normal(kb, (s[1], args.rank), jnp.float32) 
            return (a @ b.T) / sqrt_rank 
        return jax.random.normal(k, s, jnp.float32) 
 
    def noisy_dequant(q_params, scales, key, sign): 
        out = {} 
        for name in names: 
            q_noisy = jnp.clip( 
                q_params[name] + sign * args.sigma * eps_for_param(key, name), 
                -127.0, 127.0) 
            out[name] = q_noisy * scales[name] 
        return out 
 
    def egg_score_fn(q_params, scales, batch, key, sign): 
        params = noisy_dequant(q_params, scales, key, sign) 
        score, *_ = repair_score(params, batch) 
        return jnp.mean(score) 
 
    @jax.jit 
    def egg_pairs(q_params, scales, key): 
        batch = make_batch(key, args.batch_size) 
        keys  = jax.random.split(jax.random.fold_in(key, 1), pair_count) 
        plus  = jax.vmap(lambda k: egg_score_fn(q_params, scales, batch, k,  1.0))(keys) 
        minus = jax.vmap(lambda k: egg_score_fn(q_params, scales, batch, k, -1.0))(keys) 
        return plus, minus, keys 
 
    def norm_adv(diff): 
        c = diff - jnp.mean(diff) 
        return c / (jnp.std(c) + 1e-6) 
 
    @jax.jit 
    def egg_update(q_params, keys, advantages): 
        out = {} 
        for name in names: 
            delta = jnp.sum( 
                jax.vmap(lambda k, a: a * eps_for_param(k, name))(keys, advantages), axis=0) 
            q = jnp.clip(q_params[name] + egg_scale * delta, -127.0, 127.0) 
            if args.round_egg_each_step: 
                q = jnp.rint(q) 
            out[name] = q.astype(jnp.float32) 
        return out 
 
    # ── Adam ────────────────────────────────────────────────────────────── 
 
    def init_adam(p): 
        z = jax.tree_util.tree_map(jnp.zeros_like, p) 
        return {"m": z, "v": z} 
 
    def adam_update(params, grads, opt, step, lr): 
        b1, b2, eps = 0.9, 0.999, 1e-8 
        m  = jax.tree_util.tree_map(lambda m,g: b1*m+(1-b1)*g,       opt["m"], grads) 
        v  = jax.tree_util.tree_map(lambda v,g: b2*v+(1-b2)*(g*g),   opt["v"], grads) 
        mh = jax.tree_util.tree_map(lambda x: x/(1-b1**step), m) 
        vh = jax.tree_util.tree_map(lambda x: x/(1-b2**step), v) 
        p  = jax.tree_util.tree_map( 
            lambda p,mh,vh: p - lr*mh/(jnp.sqrt(vh)+eps), params, mh, vh) 
        return p, {"m": m, "v": v} 
 
    def block_tree(tree): 
        return jax.tree_util.tree_map(lambda x: x.block_until_ready(), tree) 
 
    # ── JIT steps ───────────────────────────────────────────────────────── 
 
    @jax.jit 
    def bp_step(params, opt, step, key): 
        batch = make_batch(key, args.batch_size) 
        loss, grads = jax.value_and_grad( 
            lambda p: ce_loss(p, batch)[0])(params) 
        params, opt = adam_update(params, grads, opt, step, args.bp_lr) 
        _, slot_acc, bind_acc = ce_loss(params, batch) 
        return params, opt, loss, slot_acc, bind_acc 
 
    @jax.jit 
    def evaluate(params, key): 
        batch = make_batch(key, args.valid_size) 
        loss, train_slot_acc, train_bind_acc = ce_loss(params, batch) 
        res = repair_score(params, batch)
        score, bind_acc, ref_acc, exact, stable, slot_acc, init_bind_acc, repair_gain = res
        return (loss, train_slot_acc, train_bind_acc, 
                jnp.mean(init_bind_acc), jnp.mean(bind_acc), 
                jnp.mean(ref_acc), jnp.mean(exact), 
                jnp.mean(stable), jnp.mean(slot_acc), 
                jnp.mean(repair_gain), jnp.mean(score)) 
 
    @jax.jit 
    def ce_control_step(q_params, scales, opt, step, key): 
        batch = make_batch(key, args.batch_size) 
        loss, grads = jax.value_and_grad( 
            lambda q: ce_loss(dequant(q, scales), batch)[0])(q_params) 
        q_params, opt = adam_update(q_params, grads, opt, step, args.control_lr) 
        q_params = jax.tree_util.tree_map(lambda q: jnp.clip(q,-127.,127.), q_params) 
        _, slot_acc, bind_acc = ce_loss(dequant(q_params, scales), batch) 
        return q_params, opt, loss, slot_acc, bind_acc 
 
    @jax.jit 
    def shuffled_egg_step(q_params, scales, key): 
        plus, minus, keys = egg_pairs(q_params, scales, key) 
        adv      = norm_adv(plus - minus) 
        shuffled = jax.random.permutation(jax.random.fold_in(key, 991), adv) 
        return egg_update(q_params, keys, shuffled), plus, minus 
 
    # ── Metrics ─────────────────────────────────────────────────────────── 
 
    report_key = jax.random.PRNGKey(args.seed + 7777) 
 
    def metric_values(params): 
        return tuple(float(x) for x in block_tree(evaluate(params, report_key))) 
 
    def metric_line(prefix, v): 
        return (f"{prefix} loss={v[0]:.4f} " 
                f"slot_acc={v[8]:.3f} " 
                f"init_bind={v[3]:.3f} bind_acc={v[4]:.3f} "   # ← watch these 
                f"ref_acc={v[5]:.3f} exact={v[6]:.3f} " 
                f"stable={v[7]:.3f} repair_gain={v[9]:+.3f} score={v[10]:.3f}") 
 
    def print_sample(params, key, label): 
        tokens, labels, sn, sd = make_batch(key, 1) 
        _, _, preds, _, _, _, _ = rollout(params, tokens, sn, sd) 
        t = np.asarray(jax.device_get(tokens[0])) 
        l = np.asarray(jax.device_get(labels[0])) 
        pr= np.asarray(jax.device_get(preds[0])) 
        print(f"\n--- sample {label} ---") 
        print(f"text:   {fmt_sent(t, cfg)}") 
        print(f"target: {fmt_graph(l, cfg)}") 
        print(f"pred:   {fmt_graph(pr, cfg)}") 
        print(f"correct: {'✓' if np.all(pr==l) else '✗'}  " 
              f"bind: {'✓' if pr[4]==l[4] else '✗'}") 
 
    # ═══════════════════════════════════════════════════════════════════════ 
    # PHASE 1: Backprop pretrain (cannot reach bind_acc > ~0.45) 
    # ═══════════════════════════════════════════════════════════════════════ 
    params = template 
    print(f"\nparameters={count_params(params):,}") 
    print(f"\n== Phase 1: backprop pretrain ==") 
    print(f"  (expect bind_acc to stay near 0.33 — no direct cue, no CE supervision)") 
    opt   = init_adam(params) 
    start = time.time() 
    for step in range(1, args.bp_steps + 1): 
        key = jax.random.fold_in(jax.random.PRNGKey(args.seed + 101), step) 
        params, opt, loss, slot_acc, bind_acc = block_tree(bp_step(params, opt, step, key)) 
        if step % args.print_every == 0 or step == args.bp_steps: 
            print(metric_line( 
                f"bp={step:5d} tr_loss={float(loss):.4f} " 
                f"tr_slot={float(slot_acc):.3f} tr_bind={float(bind_acc):.3f}", 
                metric_values(params)) + f"  sec={time.time()-start:.1f}") 
 
    # ═══════════════════════════════════════════════════════════════════════ 
    # PHASE 2: int8 quantisation 
    # ═══════════════════════════════════════════════════════════════════════ 
    print(f"\n== Phase 2: int8 quantisation ==") 
    q_params, scales = quantize(params) 
    q_start          = q_params 
    int8_m           = metric_values(dequant(q_params, scales)) 
    print(metric_line("int8", int8_m)) 
    results = {"int8": int8_m} 
 
    # ═══════════════════════════════════════════════════════════════════════ 
    # Control A: CE fine-tune (still can't supervise binding) 
    # ═══════════════════════════════════════════════════════════════════════ 
    if args.control_steps > 0 and args.control_mode in ("both", "ce"): 
        print(f"\n== Control A: CE fine-tune (binding_loss_weight={args.binding_loss_weight}) ==") 
        q_ce   = q_start 
        opt_ce = init_adam(q_ce) 
        start  = time.time() 
        for step in range(1, args.control_steps + 1): 
            key = jax.random.fold_in(jax.random.PRNGKey(args.seed + 606), step) 
            q_ce, opt_ce, loss, slot_acc, bind_acc = block_tree( 
                ce_control_step(q_ce, scales, opt_ce, step, key)) 
            if step % args.print_every == 0 or step == args.control_steps: 
                print(metric_line( 
                    f"ce={step:5d} tr_loss={float(loss):.4f} " 
                    f"tr_slot={float(slot_acc):.3f} tr_bind={float(bind_acc):.3f}", 
                    metric_values(dequant(q_ce, scales))) + f"  sec={time.time()-start:.1f}") 
        results["ce_control"] = metric_values(dequant(q_ce, scales)) 
 
    # ═══════════════════════════════════════════════════════════════════════ 
    # Control B: Shuffled-fitness ES (broken signal) 
    # ═══════════════════════════════════════════════════════════════════════ 
    if args.control_steps > 0 and args.control_mode in ("both", "shuffle"): 
        print(f"\n== Control B: shuffled-fitness ES ==") 
        q_shuf = q_start 
        start  = time.time() 
        for step in range(1, args.control_steps + 1): 
            key = jax.random.fold_in(jax.random.PRNGKey(args.seed + 707), step) 
            q_shuf, plus, minus = block_tree(shuffled_egg_step(q_shuf, scales, key)) 
            if step % args.print_every == 0 or step == args.control_steps: 
                print(metric_line(f"shuf={step:5d}", 
                    metric_values(dequant(q_shuf, scales))) 
                    + f"  gap={float(jnp.mean(plus-minus)):.5f}  sec={time.time()-start:.1f}") 
        results["shuffle"] = metric_values(dequant(q_shuf, scales)) 
 
    # ═══════════════════════════════════════════════════════════════════════ 
    # Phase 3: EGGROLL (optimises through discrete binding) 
    # ═══════════════════════════════════════════════════════════════════════ 
    print(f"\n== Phase 3: EGGROLL over discrete slot binding ==") 
    print(f"  (only method that can improve bind_acc past backprop ceiling)") 
    q_params = q_start 
    start    = time.time() 
    for step in range(1, args.egg_steps + 1): 
        key = jax.random.fold_in(jax.random.PRNGKey(args.seed + 404), step) 
        plus, minus, keys = block_tree(egg_pairs(q_params, scales, key)) 
        adv      = norm_adv(plus - minus) 
        q_params = block_tree(egg_update(q_params, keys, adv)) 
        if step % args.print_every == 0 or step == args.egg_steps: 
            print(metric_line(f"egg={step:5d}", 
                metric_values(dequant(q_params, scales))) 
                + f"  gap={float(jnp.mean(plus-minus)):.5f}  sec={time.time()-start:.1f}") 
        if args.sample_every and step % args.sample_every == 0: 
            print_sample(dequant(q_params, scales), 
                         jax.random.PRNGKey(args.seed + 808), f"egg_step={step}") 
    results["eggroll"] = metric_values(dequant(q_params, scales)) 
 
    # ═══════════════════════════════════════════════════════════════════════ 
    # Final comparison 
    # ═══════════════════════════════════════════════════════════════════════ 
    print(f"\n== Final comparison ==") 
    print(f"  {'method':<18} {'bind_acc':>9} {'exact':>7} {'stable':>7} " 
          f"{'score':>7} {'Δbind':>7} {'Δscore':>7}") 
    print("  " + "-"*68) 
    base_b = results["int8"][4] 
    base_s = results["int8"][10] 
    for name, v in results.items(): 
        print(f"  {name:<18} {v[4]:9.3f} {v[6]:7.3f} {v[7]:7.3f} " 
              f"{v[10]:7.3f} {v[4]-base_b:+7.3f} {v[10]-base_s:+7.3f}") 
    print(f"\n  Backprop ceiling on binding: ~0.33 (3-way, no cue, no CE supervision)") 
    print(f"  EGGROLL target: bind_acc > 0.50 by finding slot-identity configurations") 
 
    # Save 
    fp = dequant(q_params, scales) 
    arrays = {} 
    for i, n in enumerate(names): 
        arrays[f"q_{i}"]     = np.array(jax.device_get(q_params[n]), np.float32) 
        arrays[f"scale_{i}"] = np.array(jax.device_get(scales[n]),   np.float32) 
        arrays[f"float_{i}"] = np.array(jax.device_get(fp[n]),        np.float32) 
    save_npz(Path(args.checkpoint_out), arrays, { 
        "args": vars(args), "param_names": names, 
        "phase": "eggroll_lang_slot_v2", 
        "key_change": "binding removed from CE loss; no lexical binding cue"}) 
    print(f"\n  saved → {args.checkpoint_out}") 
 
 
if __name__ == "__main__": 
    main() 
