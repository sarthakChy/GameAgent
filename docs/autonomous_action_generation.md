# Autonomous Action Generation for GameAgent
## Bridging LeWM World Model + B-JEPA Action Generation

---

## The Core Problem

We have two systems that each solve half the problem:

| System | What it does well | What it can't do |
|---|---|---|
| **LeWM** | Predicts future game states accurately (`pred_loss → 0.017`) | Generate actions — only scores candidates you give it |
| **B-JEPA** | Generates action token sequences directly from visual state | Predict consequences — actions were random/inconsistent |

**The gap:** LeWM knows the world but not what to do. B-JEPA could decide what to do but didn't understand the world.

Bridging them = a fully autonomous agent.

---

## Why the Current System Needs Candidates

LeWM's predictor was trained with:

```
loss = || predict(z_t, a_t) − z_{t+1} ||²
```

It learned: **given a state and action → predict the next state.**
It never learned the reverse: **given a state → generate an action.**

The 291 candidates are a workaround. You bring your own menu, the model scores each one, and the best-scoring action wins. The agent can only ever do something from your predefined list.

---

## Why B-JEPA's Actions Were Inconsistent

B-JEPA's `LatentDynamicsModel` was a **forward model** (same direction as LeWM):
```
(z_t, action_tokens) → (μ, σ) → z_{t+1}
```

An action generator needs the **inverse**:
```
(z_t, z_{t+1}_desired) → action_tokens
```

That model never existed in b-jepa. What the cloud server actually did was sample random action embeddings and pick the one whose predicted next-state scored highest — MPC in disguise, but with purely random candidates. Hence inconsistency.

---

## The Bridge: Three Phases

```
Phase 1 (NOW):    LeWM world model + MPC with 291 candidates
Phase 2 (NEXT):   Inverse Dynamics Model → unconstrained action generation  
Phase 3 (FUTURE): Latent Actor-Critic (Dreamer-style) → goal-directed RL
```

---

## Phase 2 — Inverse Dynamics Model (IDM)

### The Idea

Train a small network that answers:
> *"Given I'm in state z_t and I want to reach z_{t+1}, what action should I take?"*

```
IDM: (z_t, z_{t+1}) → action_tokens   [B, L=128]
```

This is the exact inverse of LeWM's predictor.

### Why Training Data Is Free

The HDF5 dataset already contains everything needed:
- `pixels[i]` → LeWM encoder (frozen) → `z_t`
- `pixels[i+1]` → LeWM encoder (frozen) → `z_{t+1}`
- `action[i]` → 128 ground-truth tokens that caused the transition

No human labelling. No extra collection. Same dataset you already trained on.

### Architecture

```
z_t  ─────────┐
              ├──→ concat → project to D=192 ──→ cross-attn transformer decoder ──→ logits [B, L, vocab_size=120]
z_{t+1} ──────┘
z_diff = z_{t+1} - z_t  (direction of change, appended as additional context)

Output: autoregressive token generation, same format as action_vocab.json
Training loss: cross-entropy vs ground-truth action_tokens from HDF5
```

The decoder is a smaller version of the `MiniTransformerActionDecoder` from the b-jepa branch — only now it's trained on HIGH-QUALITY LeWM embeddings instead of frozen V-JEPA2 embeddings.

### Inference Loop With IDM (No Candidates)

```
At each 200ms tick:

1. Capture frame → LeWM encoder → z_t
2. Choose a target z_{t+1}:
     Novelty:  z_{t+1} = z_t + direction of most underexplored region
     Goal:     z_{t+1} = z_goal (pre-encoded goal screenshot)
     WM-aided: z_{t+1} = argmax over few WM rollouts (reduces candidates to ~10)
3. IDM(z_t, z_{t+1}) → action_tokens  (autoregressive, NO candidates list)
4. Decode tokens → action string → replay_frame
```

Zero candidates needed at inference. The action space is unlimited.

---

## Phase 3 — Latent Actor-Critic (Dreamer-Style)

### Architecture

```
World Model (LeWM, frozen)
z_{t+1} = WM.predict(z_t, a_t)
       |
       | imagined rollouts (no real game needed for training)
       |
Actor π(z_t) → a_t        Critic V(z_t) → scalar value
(transformer, outputs        (MLP)
 action tokens)
```

### How It Trains

```python
for epoch in range(N):
    # Start from real embeddings (seed from HDF5, not from game)
    z = sample_seed_states(gameagent_latents.h5)

    for t in range(horizon=15):
        a_tokens = actor(z)              # Actor generates action freely
        z = WM.predict(z, a_tokens)      # WM imagines next state
        r = reward_fn(z)                 # Reward computed in latent space

    # Update actor+critic on imagined returns
    returns = discounted_sum(rewards)
    update(actor, -returns.mean())       # Actor maximizes return
    update(critic, mse(critic(z), returns))
```

**The real game is only needed for final evaluation**, not training.

### Reward Options

| Reward | Formula | When to use |
|---|---|---|
| **Novelty** | `||z_{t+1} - mean(z_history)||` | Exploration, no goal |
| **Goal distance** | `-||z_t - z_goal||` | Navigate to a specific game state |
| **Coverage** | count of unique embedding clusters visited | Map exploration |
| **Human feedback** | sparse +1/-1 signal | When you know what "good" looks like |

---

## Why This Architecture Is Right

| Concern | Answer |
|---|---|
| "IDM won't generalize to states not in training data" | LeWM embeds them into the same 192-dim space, so IDM generalizes by interpolation in latent space |
| "The 200ms timing structure will be lost" | IDM outputs all 128 tokens (6 groups preserved) exactly like the training data |
| "Phase 3 needs too much RL engineering" | Actor trains entirely in imagination using WM rollouts — no game needed, no reward hacking from real env |
| "B-JEPA's action decoder was bad" | It trained on raw V-JEPA2 embeddings (1408-dim, frozen, not game-specific). IDM trains on LeWM embeddings (192-dim, game-specific, much more structured) |

---

## Implementation Roadmap

### Step 1 — Pre-compute latent dataset (Lightning AI, ~10 min)
New script: `GameAgent/scripts/precompute_latents.py`
```
Input:  gameagent.h5 (pixels, actions)
Output: gameagent_latents.h5 (z_t, z_{t+1}, action_tokens)
```

### Step 2 — Train IDM (Lightning AI, ~30 min)
New script: `GameAgent/scripts/train_idm.py`
```
Input:  gameagent_latents.h5
Output: stable-wm/checkpoints/idm/best.pt
```

### Step 3 — Update cloud server
In `lewm_cloud_server.py`, replace the MPC scoring loop:
```python
# Old (Phase 1):
sel_idx = _mpc_pick(ctx_emb, ctx_act_emb)
action_str = CANDIDATE_TEXTS[sel_idx]

# New (Phase 2):
z_target = _novelty_target(ctx_emb)        # or goal
action_tokens = IDM(z_t, z_target)         # free generation
action_str = decode_tokens(action_tokens)
```

### Step 4 (Phase 3, optional) — Train latent actor
New script: `GameAgent/scripts/train_latent_actor.py`
```
Input:  gameagent_latents.h5 (seed states), WM checkpoint
Output: stable-wm/checkpoints/actor/best.pt
```

---

## What Each Phase Gives You

| Phase | How agent decides | Candidates | Requires RL | Ceiling |
|---|---|---|---|---|
| 1 — MPC (now) | Score 291 fixed actions | Yes, 291 | No | Candidate quality |
| 2 — IDM | Generate from state delta | None | No | Training data quality |
| 3 — Actor | Learned policy via imagination | None | Yes (imagined) | Beyond training data |

---

## Recommended Next Step

**Build Phase 2 (IDM) first.**

- No RL complexity
- No reward design
- Same dataset, same hardware
- Directly eliminates the candidate bottleneck
- Estimated: 2 scripts + 1 server update + 30 min training on L40S

> [!IMPORTANT]
> Phase 2 is the most impactful change for the least effort. It converts the agent from
> "can only do predefined actions" to "can generate any valid action from the 120-token
> vocabulary" — which is what B-JEPA was trying to do, but now grounded in a proper
> world model representation.
