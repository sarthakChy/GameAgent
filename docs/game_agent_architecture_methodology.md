# Game Agent Architecture & Methodology

## 1. Project Overview & Constraints
This document outlines the architecture and data methodology for training a Vision-Language Model (VLM) agent to play complex 3D open-world games (like Assassin's Creed 3 and Valheim). 

**The Core Challenge:** Unlike massive enterprise models (e.g., Lumine) that utilize 1,700+ hours of data and thousands of GPU hours to learn via next-token prediction, this project operates under strict solo-developer constraints:
* **Data Limit:** ~2 to 3 hours of raw gameplay data (~16,000 to 50,000 active frames).
* **Hardware Limit:** Consumer-grade or single-node cloud compute (Lightning AI).
* **The Overfitting Risk:** Attempting end-to-end fine-tuning of a 7B+ parameter VLM (like Qwen2-VL) on just 3 hours of data will result in catastrophic forgetting and severe overfitting. The model will memorize the specific 3 hours of gameplay rather than learning generalized game mechanics.

---

## 2. The Chosen Architecture: "The Pre-Computation Hack"
To bypass hardware limitations and maximize sample efficiency, the architecture decouples the "Visual Cortex" from the "Motor Cortex."

### A. The Vision Backbone: Frozen Meta V-JEPA 2
Instead of training a vision encoder from scratch, we leverage Meta's pre-trained **V-JEPA 2** (`vjepa2_vit_large`). 
* **Why JEPA:** Joint Embedding Predictive Architectures learn by predicting latent representations of missing video segments. This forces the model to understand motion, spatial geometry, and object permanence without getting distracted by pixel-level rendering noise (e.g., weather, lighting, UI flicker).
* **The "Frozen" Strategy:** The V-JEPA model weights remain 100% frozen. It acts purely as a zero-shot feature extractor, converting game frames into dense, 1-dimensional mathematical arrays (embeddings).

### B. The Action Head (Motor Cortex)
* **Architecture:** A lightweight, custom-built PyTorch Multi-Layer Perceptron (MLP) or small Transformer decoder.
* **Function:** It takes the highly compressed V-JEPA embedding as input and predicts the corresponding 30Hz action string (`<|action_start|> X Y Z ; k1... <|action_end|>`).
* **Efficiency:** Because this network is tiny, it can be trained from scratch on 3 hours of data in minutes without requiring massive VRAM.

---

## 3. Data Pipeline & Temporal Methodology
The pipeline involves a one-time extraction process that bridges the Hugging Face `DatasetDict` and the PyTorch training loop.

### Handling the Frame-to-Action Mismatch (5Hz vs 30Hz)
* **Input Resolution:** The game frames are captured at **5Hz** (1 frame every 200ms).
* **Output Resolution:** The model must predict a 6-chunk action sequence representing **30Hz** micro-actions. 
* **Mapping:** One single 5Hz visual embedding directly maps to one full 6-chunk action sequence.

### The 800ms Sliding Window
Neural networks cannot react to motion from a single frozen frame. 
* During extraction, the script groups frames into **4-frame temporal windows** (representing 800ms of game history: `[t-3, t-2, t-1, t]`).
* This 800ms video clip is passed through V-JEPA.
* The resulting embedding is mapped *only* to the action string of the **final frame (`t`)**.
* **Why?** This gives the network the historical context (e.g., watching an enemy swing a sword for 800ms) required to accurately predict the immediate necessary reaction (e.g., dodging at frame `t`).
* **Note:** We intentionally discard the action labels for the very first 3 frames of any session, as they lack the historical context needed for accurate prediction.

---

## 4. Data Quality: The Cutscene Filter
A critical flaw in raw gameplay data is the presence of long cutscenes, loading screens, or AFK moments where the label is `is_idle = True`.

### The Danger of Action Collapse
If the agent sees a highly dynamic cutscene (camera panning, explosions) paired with a "do nothing" label, it learns a false correlation: *High visual motion = Take hands off the keyboard.* This causes the model to freeze during actual gameplay combat.

### The Dynamic Streak Filter
Instead of deleting video files, we implement a "Streak Counter" during the V-JEPA extraction script:
1.  **Micro-Pauses (Keep):** If the player is idle for less than 25 frames (5 seconds), the embeddings are saved. The model must learn that pausing to let stamina recharge is a valid action.
2.  **Cutscenes (Drop):** If the idle streak exceeds 25 frames, the script assumes a cutscene or menu. It temporarily stops saving embeddings until the user inputs a keystroke again. 
This dynamically cleans the 40% idle-bloat from the dataset without losing valid gameplay mechanics.

---

## 5. Development Roadmap
Moving forward, development will proceed in three distinct phases:

1.  **Phase 1: The Zero-Shot Extraction Script**
    * Write a Python script utilizing `torch.hub.load('facebookresearch/vjepa2')`.
    * Stream the HF dataset, apply the 800ms Sliding Window and Cutscene Filter.
    * Save the resulting `[Embedding Array, Action String]` pairs to a lightweight `.pt` file, allowing us to discard the heavy raw images locally.
2.  **Phase 2: The Translation Bridge (PyTorch Dataset)**
    * Build a custom `torch.utils.data.Dataset` class.
    * Load the `.pt` file and write the tokenization logic to convert the 6-chunk action strings into numerical tensors.
3.  **Phase 3: The Assembly Line (Lightning AI)**
    * Construct the custom PyTorch `nn.Module` (Action Decoder).
    * Write the training loop in PyTorch Lightning to map the V-JEPA embeddings to the tokenized keystrokes using Cross-Entropy Loss.
