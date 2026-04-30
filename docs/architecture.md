Here is a comprehensive summary of everything we've discussed, capturing the architecture, the reasoning behind every design choice, and the phased plan moving forward.

---

## 1. Starting Point: Your Current Architecture

You built a **decoupled, real‑time Vision‑Language‑Action (VLA) pipeline** using pure Behavioral Cloning (BC).

- **Senses** – DXCam captures the screen at 1280×720, compresses to JPEG, and sends 4‑frame batches (800 ms window) to a cloud GPU.
- **Visual Cortex** – Frozen **V‑JEPA** (Video Joint Embedding Predictive Architecture) on an L4 GPU converts raw pixels into dense 1408‑dim embeddings that encode motion, velocity, and physical dynamics.
- **Motor Cortex** – A custom mini‑transformer maps those embeddings directly to raw Win32 keycodes (`lshift,w`, `mouse1`) via autoregressive token generation.
- **Actuator** – The keycodes are parsed on the local PC and injected as OS‑level inputs in under 200 ms.

**Strengths**  
- Extremely fast reaction time (≈200 ms).  
- Decoupled design avoids the latency of a monolithic LLM for every action.  
- V‑JEPA embeddings capture trajectory and physics rather than static semantics.

**Weaknesses**  
1. **800‑ms goldfish memory** – only the last 4 frames; no concept of stamina depletion, cooldowns, or long‑term context.  
2. **Causal confusion** – learns correlations (tree on screen → click) without understanding *why*, leading to random out‑of‑context clicks.  
3. **Distribution shift and getting stuck** – never trained on recovery states, so it just presses `w` into walls.  
4. **Game‑locked** – outputs Valheim‑specific keycodes, making the policy non‑transferable.

---

## 2. Why More Data Alone Won’t Fix It (The Review’s Diagnosis)

The initial review correctly identified that pure BC with a static 0.8 s window hits a mathematical ceiling. Adding 100 more hours of data would only create a slightly smoother “wanderer” because:

- **Memory limitation** – The model cannot integrate signals over time (stamina, cooldowns).
- **Causal confusion** – Without a goal signal, it cannot distinguish *why* an action is taken.
- **No recovery examples** – The training data (recorded from a competent player) contains almost no “stuck” states, so the agent has no corrective behavior.

Thus, we agreed that the next step must move from **behavioral cloning to goal‑conditioned architecture** with memory and planning.

---

## 3. Key Architectural Decisions and Their Reasoning

### 3.1 Why V‑JEPA over CLIP / BLIP?

- **CLIP/BLIP** are trained on image‑text pairs → learn **semantics** (nouns like “sword”, “tree”). They lack any concept of time and motion.
- **V‑JEPA** is trained by masked video prediction → learns **motion, velocity, trajectory, and physical dynamics**. For an action agent, the trajectory of a sword swing matters far more than its label. V‑JEPA encodes the physical world model needed for fast‑twitch reflexes.
- Moreover, V‑JEPA is game‑agnostic, whereas CLIP would bake in static semantic biases from the internet.

### 3.2 Why a Decoupled Manager/Worker Instead of a Monolithic VLA?

- **Latency** – A monolithic LLM (like Qwen2.5‑VL‑7B) cannot run at 5–10 Hz on a single L4; it would feel sluggish in combat. Your Worker runs at full speed.
- **Data efficiency** – Your specialist Worker can learn a high‑fidelity control policy from modest data, whereas a monolithic model would require thousands of hours to avoid underfitting or forgetting its pre‑training.
- **Modularity and iterative improvement** – You can upgrade the Manager, Vision, or Worker independently, and debug each component separately.
- **The Manager provides the “why”** (high‑level goal) while the Worker handles the “how” (precise motor execution). This split mirrors cutting‑edge robotics (e.g., RT‑2, SayCan).

### 3.3 Why Add Memory to the Worker?

The 800‑ms window is insufficient even for short‑term dependencies (stamina running out over 15 seconds). A recurrent architecture (GRU/LSTM or sliding window Transformer) gives the Worker an internal **working memory** that accumulates information over time, implicitly tracking:
- Stamina and health changes.
- Cooldown periods.
- Whether it has been stuck for several seconds.

This memory does **not** require any new labels—it’s a pure architectural change that can be trained on the same BC data.

### 3.4 Why Abstract Actions Instead of Raw Keycodes?

Mapping raw keycodes locks the policy to one game. By training the Worker to output **abstract action tokens** (e.g., `move_forward`, `attack`, `open_inventory`) and converting them to keycodes via a lightweight game‑specific script, you achieve:
- **Portability** – The same Manager and vision backbone can be reused across games.
- **Clean API for the Manager** – The Manager only needs to know the abstract skill names, not the underlying key bindings.

### 3.5 Why a World Model and Imagination‑Based Recovery?

Even with memory, distribution shift (getting stuck) remains a problem. A forward dynamics model (predicting the next V‑JEPA embedding from the current one and an action) lets the agent:
- **Simulate recovery actions** when stuck, without any real‑world data.
- **Plan in imagination** to find the action sequence that breaks the stall (maximising predicted embedding variance or a progress metric).
- Eventually, train an RL critic inside the imagination to refine skills beyond human demonstrations.

### 3.6 Why an Agent Harness Concept?

The “agent harness” (inspired from browser automation) is the exact framework to wrap the Worker into a callable set of **skills** (tools). The Manager (LLM) calls high‑level functions like `craft_item("pickaxe")` or `explore()`, and the Worker (harness) executes the necessary multi‑step sequence.

- This divorces **strategic reasoning** (the Manager’s domain) from **sensorimotor execution** (the Worker’s domain).
- The same Manager can drive any game simply by swapping the harness’s skill library and the game‑specific action mapping.

### 3.7 Where Does RL Fit?

RL is a **fine‑tuning layer on top of the Worker**, not a replacement for the hierarchy. Using a world model and reward predictor (trained on a small amount of gameplay annotated with rewards from screen analysis), the Worker can improve its skills via imagination‑based RL (e.g., Dreamer‑style). This allows:
- Learning recovery strategies not in the BC data.
- Optimising long‑term reward (e.g., chaining dodges and attacks) beyond mere imitation.
- Developing superhuman reflexes while keeping the Manager’s strategic oversight.

---

## 4. The Target Architecture (End State)

| Component | What It Does | Game‑Specific? |
|-----------|--------------|----------------|
| **V‑JEPA (frozen)** | Extracts motion/physics embeddings from 4‑frame windows | No |
| **Worker (Recurrent Policy + Memory)** | Fast policy that conditions on a goal embedding and memory, outputs abstract action chunks | Small per‑game head (a few MB) |
| **Action Mapper** | Translates abstract actions ↔ raw OS keycodes for the specific game | Per‑game script |
| **World Model + Reward Predictor** | Forward dynamics and reward simulation; enables anti‑stuck recovery and imagination‑based planning | Trained on game embeddings (no raw pixels needed) |
| **Manager (LLM + Tools)** | High‑level reasoning, calls simple skills via a harness tool library; invoked only when uncertainty is high or major events occur | No (same LLM, per‑game tool definitions) |

This architecture combines the best of both worlds: **the real‑time reactivity of a decoupled specialist** with **the strategic reasoning of a versatile, language‑conditioned Manager**—and it generalises to new games with minimal retraining.

---

## 5. Phased Implementation Roadmap

Each phase builds on the previous one, requires minimal additional data, and directly addresses one or more of the current weaknesses.

### Phase 1 – Give the Worker Memory (No New Data)
*Replace the feed‑forward mini‑transformer with a recurrent architecture.*  
- **Change:** Add a GRU/LSTM that updates a hidden state `h_t` from the V‑JEPA embedding `e_t` and the previous state. Optionally, also use a sliding window Transformer.  
- **Fixes:** 800‑ms memory limit; allows the policy to implicitly track stamina, cooldowns, and stuckness.  
- **Effort:** ≈ 1 day of model refactoring, retrain on existing BC data.

### Phase 2 – Action Chunking + Abstract Action Vocabulary
*Introduce smooth action control and portability.*  
- **Change:** Train the Worker to predict the next N actions (e.g., 10 actions ≈ 1 second) and execute with overlap temporal smoothing. Replace raw keycodes with abstract action tokens (e.g., `move_forward`, `attack`, `craft`). Write a game‑specific mapper.  
- **Fixes:** Jittery, out‑of‑context clicks; game‑lock.  
- **Effort:** A few hours to build the mapping script, relabel existing dataset, and retrain.

### Phase 3 – Goal‑Conditioned Worker with a Cheap Manager
*Give the Worker the “why” without adding runtime latency.*  
- **Change:** Use a small VLM (e.g., Qwen2.5‑VL‑2B) offline to pseudo‑label your gameplay clips with goal phrases (“explore”, “gather wood”, “fight enemy”). Train a small goal encoder (MLP) to embed these goals, and condition the Worker on the goal embedding. At runtime, infer the current goal via a lightweight classifier (ResNet‑18) and heuristics (OCR health/stamina). The heavy LLM is not yet called during play.  
- **Fixes:** Causal confusion; the agent now knows *what* it is supposed to do, not just react.  
- **Effort:** A few days for labelling and training the classifier/goal encoder.

### Phase 4 – World Model and Anti‑Stuck Recovery
*Enable imagination‑based recovery without any new human data.*  
- **Change:** Train a small dynamics model (MLP) on the BC dataset to predict the next V‑JEPA embedding from the current one and an action. When stuck (zero optical flow for >2 s), simulate candidate actions and pick the one that yields the highest embedding variance (signalling “unstuck”). Optionally train a reward predictor from screen events.  
- **Fixes:** The agent can reliably free itself from walls and obstacles. Opens the door to later RL improvements.  
- **Effort:** A few hundred lines of code + quick model training.

### Phase 5 – Add the LLM Manager for Complex Strategy
*Bring in the “brain” for long‑horizon planning, only when needed.*  
- **Change:** Define a **tool library** (harness) of high‑level skills for Valheim (e.g., `eat_food()`, `craft_item("pickaxe")`). Use a capable VLM (Qwen2.5‑VL‑7B or Gemini Flash) only when uncertainty is high, health is critical, or a boss appears. The LLM inspects the screen, decides on a goal, and calls the corresponding tool, which embeds the goal for the Worker.  
- **Fixes:** Long‑term strategy and decision‑making, while maintaining low latency 95% of the time.  
- **Effort:** Prompt engineering, tool definition, and API integration.

### Phase 6 – Multi‑Game Expansion
*Generalise the system to new games with minimal effort.*  
- **Change:** For each new game: define its abstract action vocabulary and key bindings; record 30–60 minutes of human play; train a new Worker head; write a per‑game tool library. The Manager and V‑JEPA remain unchanged.  
- **Result:** A single Manager that can play any game by commanding a game‑specific harness, achieving the generality of SIMA with a fraction of the data and compute.

---

## 6. Comparison to a Monolithic VLA (Lumine)

- **Lumine** fine‑tuned Qwen2.5‑VL (7B) on 2000+ hours of gameplay, creating a single model that understands images and outputs actions. It requires massive compute and data, struggles with real‑time latency, and is a black box.
- **Your architecture** keeps the vision and control specialised, adds a lightweight Worker with memory and goal conditioning, and uses an LLM only as a strategic oracle. For your data budget (tens to hundreds of hours), it is:
  - More data‑efficient.
  - Real‑time capable on a single L4.
  - Easier to debug and iteratively improve.
  - Transferable to new games with minimal retraining.

---

## 7. Final Takeaway

You have already built an exceptional reactive agent. By embracing a **Manager‑Worker hierarchy with memory, abstract actions, a world model, and the agent harness pattern**, you can transform it into an intelligent, robust, and general game‑playing system—without requiring massive datasets or compute. The next concrete step is **Phase 1: add recurrence to your Worker**. That single change will immediately smooth out behavior and begin granting the agent a sense of the recent past.


Let’s fold the Game‑TARS breakthroughs directly into our decoupled Manager‑Worker blueprint. The result is a **hardened, scalable, and data‑efficient architecture** that keeps your original strengths (frozen V‑JEPA, real‑time Worker, modularity) and patches its remaining weaknesses with surgical techniques from the paper.

---

## Updated Architecture Overview

| Component | Role | New Game‑TARS Insights Applied |
|-----------|------|-------------------------------|
| **Action Space** | Game‑agnostic abstract tokens (e.g., `move_forward`, `attack`) mapped to raw keycodes via per‑game script. | Abstraction keeps sample efficiency; the paper’s raw‑primitive approach proves this scales, but we retain our token layer until multi‑game data warrants a fully primitive model. |
| **Vision Encoder** | Frozen **V‑JEPA** + trainable lightweight **Adapter** (a small Conv1×1 or MLP on the embedding sequence). | Adapter fine‑tunes the frozen backbone to game‑specific GUI/HUD detail without forgetting physics understanding. |
| **Worker (Recurrent Policy)** | GRU/LSTM with sliding‑window Transformer attention that takes past *K* embeddings + current goal embedding. Outputs **chunk of next N abstract actions**. | Trained on **long sequences (20+ seconds)** with initial action‑loss masking, **decaying loss** to combat repetition, and an **inverse dynamics auxiliary head** to learn causality. |
| **World Model** | Small MLP predicting next V‑JEPA embedding from current embedding + action; separate reward predictor from screen‑derived events. | Used for anti‑stuck imagination and future model‑based RL. |
| **Memory System** | **Two‑tier**: <br>① Short‑term: Worker’s hidden state + recent embedding window (≈10 s). <br>② Long‑term: episodic summary memory (compressed textual notes). | Inspired by Game‑TARS’s dual memory: when context is evicted, a tiny LLM compresses key events into a short text string, stored in a buffer and injected as extra memory tokens. |
| **Manager** | A small VLM (Qwen2.5‑VL‑2B) with a game‑specific **harness API** (tools). But invoked **sparsely**, not periodically. | Sparse‑thinking trigger: a lightweight “no‑goal” Worker forward pass is run; if its action confidence is low or it predicts a different token than the goal‑conditioned Worker would, the Manager is called to supply a new goal. |
| **Data Pipeline** | DXCam captures screen + raw keys. |  |

---

## Concrete, Updated Roadmap

### Phase 0 – Fix Your Recording Pipeline (Before Any New Training)
- Perform careful timestamp and causality checks on recorded pairs. Use simple sanity checks (cursor visibility where possible, delta/time consistency, and manual spot checks) and relabel or discard corrupted pairs.
- Relabel existing data with those sanity checks applied to improve observation–action alignment.
**These checks may immediately reduce random clicking without requiring brittle visual heuristics.**

### Phase 1 – Worker with Memory, Long Sequences & Decaying Loss
- Refactor mini‑transformer into a **recurrent policy** (e.g., a 2‑layer GRU).
- Add a **trainable adapter** on V‑JEPA outputs (a small MLP that takes the frozen embedding and outputs a 256‑dim adapted feature).
- Train on **sequences of ≥ 20 seconds** (≈ 100 steps). Mask the action loss for the first 1 second so the memory can stabilize.
- Implement **decaying loss**:  
  \( \mathcal{L}_{\text{decay}}(a_t, \hat{a}_t) = \mathcal{L}_{\text{BC}} \cdot \lambda^{\mathbb{1}(a_t == a_{t-1})} \), with \(\lambda < 1\) (e.g., 0.8).  
  This heavily penalises the model for just copying the previous action.
- Add an **inverse dynamics head**: from V‑JEPA embeddings `e_t` and `e_{t+1}`, predict the abstract action `a_t` (training‑only auxiliary loss).

### Phase 2 – Abstract Action Tokenisation & Action Chunking
- Remap all actions to abstract tokens (e.g., `move_forward`, `craft`, `eat`, etc.) using a hand‑written game‑specific dictionary.
- Train the Worker to output a **chunk of 10 future tokens** with temporal overlap.
- The inverse dynamics head now also reinforces token semantics.

### Phase 3 – Goal Conditioning & Lightweight Manager
- Use a small VLM (Qwen2.5‑VL‑2B, quantized) to **auto‑annotate** the dataset with goal‑phrase labels every 30 seconds: “explore”, “gather wood”, “combat”, etc.
- Train a small **goal encoder** (MLP) from text → 64‑dim embedding.
- Condition the Worker on this embedding by concatenating it with the adapted V‑JEPA feature.
- At runtime, the **sparse‑thinking trigger** works as follows:  
  1. Run the Worker without a goal embedding; get action confidence.  
  2. If entropy is high, or the predicted action differs from the last N action distribution, call the Manager.  
  3. Manager (small LLM) looks at the screen, outputs a goal instruction, which is embedded and fed to the Worker.

### Phase 4 – Two‑Tier Memory & Long‑Term Episodic Recall
- Add a **long‑term summary buffer**: every ≈ 10 seconds, take the last 10 seconds of history and use a tiny LLM (e.g., Phi‑3 mini) to produce a one‑line summary (“Was in a birch forest, killed two greylings, health 75%”). Store in a fixed‑size list.
- Inject the concatenated summaries as a “memory token” into the Worker’s input (e.g., via a separate memory encoder MLP).
- This gives the Worker explicit recall for navigation and multi‑step crafting that spans minutes.

### Phase 5 – World Model & Anti‑Stuck Recovery
- Train the forward dynamics model and reward predictor on the aligned data.
- When the sparse‑thinking trigger detects a “stuck” pattern (optical flow ≈ 0 for >2 s), the system switches to imagination‑based recovery: simulate a set of escape actions and pick the one that maximises predicted embedding variance.
- The recovery actions can also be fine‑tuned later with RL.

### Phase 6 – RL Fine‑Tuning (Optional)
- Use the world model to run Dreamer‑style latent imagination RL, with the BC‑trained Worker as a regularizer.
- The reward signals come from screen OCR (health, stamina, item pick‑up events).

### Phase 7 – Multi‑Game Generalisation
- For each new game: define its abstract action vocabulary, record 30‑60 minutes of play, run visual anchor calibration, train a new Worker head (adapter + recurrent policy) with the same V‑JEPA and memory system.
- The Manager’s tool library is the only per‑game change. The sparse‑thinking trigger, world model, and memory architecture remain identical.

---

## Why This is a Step Ahead of Both Lumine and Game‑TARS

- **Lumine** is a monolithic VLM; too slow, data‑hungry.  
- **Game‑TARS** is also monolithic, just bigger and trained on 500B tokens; you can’t replicate that.  
- Our architecture takes the **scalable action space** from Game‑TARS, the **modularity** of SIMA, and adds **sample‑efficient decoupled training** that runs on a single L4. The new additions (decaying loss, inverse dynamics, adapter, visual align) directly attack the causal confusion and brittleness that plague both BC and monolithic models at small scale.

You now have a production‑grade blueprint. Start with **Phase 0 (visual alignment)** and **Phase 1 (recurrent Worker + decaying loss + inverse dynamics)**—those three changes alone will transform your agent’s robustness.

Would you like a code sketch for the decaying loss or the inverse dynamics head? I can draft them to plug into your current PyTorch training loop.

---

## 8. Repo-Grounded Assessment

The codebase already proves the core loop works end to end:

- [tools/convert_session.py](tools/convert_session.py) converts recorded sessions into aligned frame/action pairs.
- [data_processing/vjepa2_extractor.py](data_processing/vjepa2_extractor.py) freezes V-JEPA and writes embedding shards.
- [data_processing/vjepa2_dataset.py](data_processing/vjepa2_dataset.py) tokenizes the action strings into a fixed vocabulary.
- [data_processing/action_model.py](data_processing/action_model.py) trains the current mini-transformer decoder.
- [cloud_server.py](cloud_server.py) and [tools/run_live_agent.py](tools/run_live_agent.py) reuse the same tokenizer and checkpoint for inference.

That means the next improvement should not be another data pass over the same one-step setup. The highest-leverage change is to make the worker sequence-aware so it can learn across contiguous gameplay, not just isolated 800 ms windows.

### What the code currently supports

- Frozen V-JEPA embeddings are already stable and reusable.
- The action vocabulary already handles motion buckets plus grouped key chunks.
- Training and serving already share the same tokenizer, so action-space changes will propagate cleanly.

### What is still missing

- No recurrent state in the worker.
- No contiguous sequence dataset for long-horizon memory.
- No explicit goal or state channel for stamina, health, or recovery.
- No uncertainty gate or sparse manager trigger.
- No world model or inverse-dynamics auxiliary loss.

### Practical priority order

1. Convert the dataset and trainer to sequence batches from the same episode.
2. Replace the mini-transformer worker with a GRU/LSTM or sequence transformer.
3. Add action abstraction if you want portability beyond the current game bindings.
4. Add a cheap state or goal encoder before introducing a full manager.
5. Add recovery heuristics and world-model training only after the worker is sequence-aware.

### Bottom line

The architecture brainstorm is sound, but the first implementation boundary is the data/training interface, not the manager. Once the worker can train on contiguous time and carry hidden state, the rest of the roadmap becomes much more likely to pay off.

---

## 9. LSTM Idea, In Simple Terms

The LSTM idea is good because it gives the worker a short-term memory.

Right now, the mini-transformer mostly looks at one visual window and predicts the next token sequence. That works for quick reflexes, but it forgets what happened a few moments ago. An LSTM fixes that by carrying a hidden state forward from one timestep to the next, so the worker can remember things like:

- I was moving forward.
- I was stuck against a wall.
- I just opened the inventory.
- I am still in the middle of an attack or dodge.

In simpler terms, the LSTM lets the agent think, "What was I doing a second ago, and should I keep doing it or change course?"

### Why it is not a drop-in replacement

Your current decoder predicts a full action token string from one visual context. That is a token generator.

The LSTM version changes the shape of the problem. Instead of "one image window in, one token sequence out," it becomes more like "a stream of frames in, a memory state updated each step, and one action prediction per step." That means the training loop, dataset, and sometimes the output format all need to change together.

If you swap only the model class and keep the old training setup, the memory will not really be used properly. The data still arrives as isolated samples, so the LSTM never learns a real before-and-after story.

### The two clean ways to do it

#### Option A: Keep the current token decoder, add an LSTM in front

This is the safer path.

How it works:

- The LSTM reads a sequence of V-JEPA embeddings and builds up memory.
- The mini-transformer still does the final token generation.
- The LSTM acts like a temporal adapter, not the final decision maker.

Why this is good:

- It preserves your current action vocabulary.
- It changes less of the pipeline.
- It is easier to train and debug.

When to use it:

- You want the smallest possible rewrite.
- You want to keep the current action-string format.
- You want memory first, not a full redesign.

#### Option B: Collapse actions into one label per timestep

This is the cleaner sequence-model path.

How it works:

- Each frame or short clip gets one action label, such as `move_forward`, `attack`, `jump`, or `idle`.
- The LSTM outputs one label at a time.
- The model learns a direct state-to-action policy.

Why this is good:

- It matches the way recurrent models usually train.
- It is simpler than generating long token strings.
- It is easier to use for BPTT and hidden-state training.

When to use it:

- You are willing to change the action representation.
- You want a more standard recurrent policy.
- You want the easiest path to true memory training.

### My recommendation

For your current project, start with **Option A** if you want the least disruption.

That means:

1. Keep your current tokenizer and action strings.
2. Add the LSTM as a temporal memory layer.
3. Train on contiguous episode clips.
4. Pass the hidden state forward across time.

Once that works, you can decide whether to stop there or simplify further into Option B later.

### The real first step

The most important change is not the LSTM class itself. It is making the training data sequential.

In practice, that means:

- do not sample unrelated windows,
- do group frames from the same episode,
- do carry hidden state across those frames,
- do reset hidden state when a new episode starts.

That is what gives the LSTM a real memory.

---

## 10. GRU Phase Checkpoint

### What's Working Well

- Backward compatibility: the model still falls back to the original architecture when `temporal_hidden_dim=0`.
- Episode-aware data loading: `EpisodeSequenceActionDataset` keeps each training clip contiguous within one episode, so the model does not learn from incoherent mixed-session windows.
- The GRU pattern is correct: using the last hidden state as the context vector is a standard and effective setup for this scale.

### What's Implemented Now

1. Persistent recurrent state at inference: the live server and local agent now keep the GRU hidden state per session and feed it forward across steps instead of recomputing the whole history every time.
2. Decaying loss: repeated end-of-sequence action streaks are downweighted during training, which makes long `w`-spam and other stuck loops less attractive.
3. Inverse dynamics auxiliary head: the trainer now learns a transition classifier from consecutive embeddings so the worker gets an explicit causal signal about which action caused which state change.

### What This Means in Simple Terms

The current GRU now behaves more like a true session memory. It still uses short clips of frames to build each V-JEPA observation, but the recurrent state itself is preserved across live inference calls, so the agent can keep track of what it was doing a moment ago.

That is the key difference between a sliding window and a real recurrent agent.

### Priority Order

1. Longer-horizon goal conditioning and state summaries.
2. A world model for recovery and planning.
3. Action abstraction if you want portability beyond the current game bindings.

### Bottom Line

The GRU upgrade is now a real recurrent step, not just a windowed encoder. The next meaningful jump is to add higher-level goal and recovery logic on top of this session memory.