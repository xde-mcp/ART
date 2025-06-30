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

### 2. Zero exploration ⇒ identical trajectories ⇒ no gradient
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

### 3. Reward sparsity (binary 0/1)
Tau-Bench only grants reward `1.0` on a perfectly correct final database state **and** matching textual outputs; anything else is `0.0`:
```116:146:dev/tau-bench/tau_bench/envs/base.py
            reward = 0.0
```
Early in training *all* trajectories usually earn `0`, again giving no learning signal.  Art-E's reward is a dense float in [-2,2] with many partial-credit branches, so gradients appear from step 1.

---

### 4. `logprobs` may be missing
`ToolCallingRLAgent` asks OpenAI-compatible servers for `logprobs=True`, but many vLLM deployments silently drop the field.  ART's trainer needs token-level log‐likelihoods to compute advantages; if they are `None` the backend silently skips gradient updates.  Art-E's rollout converts the *Choice* object to OpenAI dict **without** needing logprobs and has proven compatible.

---

### 5. Mismatched train/val split sizes
`TauBenchTrainingConfig.training_dataset_size` defaults to `30`, but `val_set_size` is `85`.  When the env has <115 tasks, the val indices can run past `len(env.tasks)` causing silent 0-reward trajectories that bias the mean downwards.

---

## Short-term experiments to isolate the culprit
1. **Force rollouts to use the LoRA:** pass `model="hosted_vllm/<name>` or inject the Art-E remapping into `ToolCallingRLAgent`.
2. **Turn on exploration:** re-run with `--temperature 0.7` and inspect per-group reward variance.
3. **Add shaped reward:** temporarily replace `Env.calculate_reward()` with a softer metric (e.g. +0.1 per correct action) and observe learning.
4. **Inspect backend logs:** confirm whether `/train_model` requests include `logprobs` and whether gradient steps counter increments.

---

## Likely ranking of root causes
1. Policy endpoint mismatch (high impact, easy fix).
2. Degenerate trajectories due to zero temperature.
3. Reward sparsity.
4. Missing logprobs.
5. Val‐set index overflow.

Implementing (1) and (2) should already make the validation curve resemble the Art-E run.