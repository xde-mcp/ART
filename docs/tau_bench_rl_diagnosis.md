## Tau-Bench RL: Investigation & Root-Cause Hypotheses

This document captures the **most plausible reasons** the RL loop in `dev/tau-bench/run_rl.py` is unable to improve the policy, whereas the analogous loop in `examples/art-e/art_e/train.py` does.

### 0. High-level symptom
* Training loop executes end-to-end without runtime errors.
* Validation reward stays flat (or even degrades) across steps, unlike the clear upward trend seen in the Art-E run.

---

### 1. Policy **not actually queried** during rollouts
`rollout_tau_bench_task()` creates an agent via `agent_factory`, which ultimately calls `ToolCallingRLAgent` with `model=config.model`.

```150:185:dev/tau-bench/tau_bench/run.py
        return ToolCallingRLAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
            api_key=config.api_key,
            base_url=config.base_url,
        )
```

The **trainable** model's inference endpoint, however, is exposed at `hosted_vllm/<model.name>` (this is what Art-E does explicitly).  If `config.model` is the *human-readable* name (`tau-bench-rl`, `qwen-14b`, etc.) the backend may route to the *base* checkpoint – **not** the LoRA weights updated during `model.train(...)`.  This would make every rollout/eval still use the *initial* policy, so learning appears to do nothing.

Why Art-E works:
```120:141:examples/art-e/art_e/rollout.py
        if model.trainable:
            litellm_model_name = f"hosted_vllm/{model.name}"
```
It rewrites the model name so rollouts always hit the freshly-trained LoRA.  Tau-Bench never does this remapping.

---

### 2. Missing `Trajectory.tools` ⇒ tokenizer **replays wrong context**
In Art-E, every rollout attaches the **exact same tool schema** it sent to the LLM back onto the trajectory object:
```55:75:examples/art-e/art_e/rollout.py
    if model.config.use_tools:
        traj.tools = tools
```
Tau-Bench rollouts never do this; `traj.tools` remains `None`.

Why that matters:
1. **Token reconstruction mis-aligns log-probs**  
   * During training ART re-materialises the full chat using
     ```python
     tokenizer.apply_chat_template(messages, tools=trajectory.tools)
     ```
   * If `trajectory.tools` is `None`, the tool definitions are omitted, so the tokenizer generates a *shorter* sequence than what the model produced at rollout time.  All assistant tokens therefore shift left; the stored `logprobs` now correspond to **different token IDs**.  Gradient estimates become nonsense.

2. **Implicit distribution shift**  
   The model is asked to imitate answers conditioned on *less* context than it actually had.  Even if log-prob alignment were perfect, this still teaches an inconsistent mapping (`P(response | tools…)` vs. `P(response | ∅)`).

3. **Symptom in metrics**  
   Look for unusually high variance or NaNs in the backend's KL/entropy prints; those often arise when token counts and log-prob vectors disagree.

How to test quickly:
```diff
 traj = art.Trajectory(...)
-# Tau-Bench: nothing extra here
+traj.tools = env.tools_info  # mirrors the list passed to the LLM
```
Re-run a training step; if you now see non-zero gradients and rewards start to diverge within groups, the hypothesis is confirmed.

---

### 3. Zero exploration ⇒ identical trajectories ⇒ no gradient
The default temperature in the Tau-Bench CLI is `0.0`:
```33:40:dev/tau-bench/run_rl.py
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
```
With greedy decoding the *six* trajectories generated per `TrajectoryGroup` are byte-for-byte the same.  ART's REINFORCE/RLHF backend uses within-group *rank* to create a supervised preference loss; if all rewards are ties the loss is zero.

Art-E uses the **same nominal default**, but its environment is stochastic (LLM tool calls, search results, etc.) which injects reward variance even at greedy temperature.  Tau-Bench's env is purely deterministic – exploration must come from decoder randomness.

---

### 4. Reward sparsity (binary 0/1)
Tau-Bench only grants reward `1.0` on a perfectly correct final database state **and** matching textual outputs; anything else is `0.0`:
```116:146:dev/tau-bench/tau_bench/envs/base.py
            reward = 0.0
```
Early in training *all* trajectories usually earn `0`, again giving no learning signal.  Art-E's reward is a dense float in [-2,2] with many partial-credit branches, so gradients appear from step 1.

---

### 5. `logprobs` may be missing
`ToolCallingRLAgent` asks OpenAI-compatible servers for `logprobs=True`, but many vLLM deployments silently drop the field.  ART's trainer needs token-level log‐likelihoods to compute advantages; if they are `None` the backend silently skips gradient updates.  Art-E's rollout converts the *Choice* object to OpenAI dict **without** needing logprobs and has proven compatible.

---

### 6. Mismatched train/val split sizes
`TauBenchTrainingConfig.training_dataset_size` defaults to `30`, but `val_set_size` is `85`.  When the env has <115 tasks, the val indices can run past `len(env.tasks)` causing silent 0-reward trajectories that bias the mean downwards.

---

## Short-term experiments to isolate the culprit
1. **Force rollouts to use the LoRA (`hosted_vllm`)**
2. **Attach `traj.tools = env.tools_info` before `traj.finish()`**
3. **Turn on exploration:** `--temperature 0.7`
4. **Add shaped reward**
5. **Check `/train_model` logs for missing logprobs**

---

## Likely ranking of root causes
1. Policy endpoint mismatch (hosted-vllm).
2. **Missing `traj.tools` (token misalignment).**
3. Degenerate trajectories due to zero temperature.
4. Reward sparsity.
5. Missing logprobs.
6. Val-set index overflow.

Implementing (1) **and** (2) together should give a visible upward trend comparable to the Art-E run.