<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer</h1>
</p>

<p>
Train multi-step agents for real-world tasks using GRPO.
</p>

[![PRs-Welcome][contribute-image]][contribute-url]
[![Downloads][downloads-image]][pypi-url]
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)

[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

</div>

## 📈 Train AI Agents That Actually Get Better

**Want an AI agent that learns from mistakes instead of just hallucinating?** ART makes it as easy as running a notebook.

✨ **No PhD required** - If you can `pip install`, you can train agents  
🚀 **See results fast** - Watch your agent improve in real-time  
🎮 **Start simple** - Learn with games, then build real tools

```python
# pip install openpipe-art

import art

model = art.TrainableModel("my-agent", "Qwen/Qwen2.5-14B-Instruct")
# ... your agent code ...
await model.train()  # Agent gets smarter automatically 🧠
```

**⭐ Like this?** Star us! Our devs run on GitHub stars and caffeine.

[🚀 Try in 5 minutes →](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)

## ART Overview

**New to reinforcement learning, GRPO, or LLM agents?** ART is designed to be your perfect starting point. While many RL frameworks require deep expertise in complex algorithms, ART makes production-grade agent training accessible to developers at any level.

ART is an open-source RL framework that improves agent reliability by allowing LLMs to **learn from experience**. Whether you're a seasoned ML engineer or just getting started with agent development, ART provides an ergonomic harness for integrating GRPO into any Python application—no PhD in RL required.

**Start learning immediately:** Run one of our [interactive notebooks](#-notebooks) below to see real agents improve in minutes, not hours. When you're ready to dive deeper, our comprehensive [documentation](https://art.openpipe.ai) will guide you through every concept.

## 📒 Notebooks

| Agent Task         | Example Notebook                                                                                                             | Description                                     | Comparative Performance                                                                                                                                                                             |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ART•E [RULER]**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e/art-e.ipynb)                 | Qwen 2.5 7B learns to search emails using RULER | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb) |
| **2048**           | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048                 | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue       | [Link coming soon]                                                                                                                                                                                  |
| **Tic Tac Toe**    | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe          | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**      | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames            | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |
| **AutoRL [RULER]** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/auto_rl.ipynb)                     | Train Qwen 2.5 7B to master any task            | [Link coming soon]                                                                                                                                                                                  |

## Why ART is Perfect for Beginners

**🎯 Learn by doing, not theory.** Skip months of reading papers and dive straight into training real agents. Our notebooks let you see concrete results in under 30 minutes—the same techniques used by production teams at top AI companies.

**🚀 Production-ready from day one.** Unlike academic frameworks, ART is built for real applications. Learn the exact same patterns and tools that power agents in production environments, so your skills transfer immediately to professional work.

**🛡️ Beginner-friendly with expert depth.** Start with our intelligent defaults that "just work," then gradually customize as you learn. ART grows with you—from your first agent to scaling multi-agent systems.

**💡 Real-world learning path:**

- **Week 1:** Train your first game-playing agent (2048, Tic-Tac-Toe)
- **Week 2:** Build practical tools (email search, code assistance)
- **Week 3:** Master advanced techniques (RULER rewards, multi-step planning)
- **Beyond:** Apply to your own projects with production-grade infrastructure

**⚡ Instant gratification:** Watch your agents improve in real-time with visual progress tracking—no more wondering if your training is working.

**🔧 Professional tooling:** Integrations with W&B, Langfuse, and OpenPipe provide the same observability and debugging tools used by AI teams at scale.

## Installation

ART agents can be trained from any client machine that runs python. To add to an existing project, run this command:

```
pip install openpipe-art
```

## 🤖 ART•E Agent

Curious about how to use ART for a real-world task? Check out the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post, where we detail how we trained Qwen 2.5 14B to beat o3 at email retrieval!

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop Overview

ART's functionality is divided into a **client** and a **server**. The OpenAI-compatible client is responsible for interfacing between ART and your codebase. Using the client, you can pass messages and get completions from your LLM as it improves. The server runs independently on any machine with a GPU. It abstracts away the complexity of the inference and training portions of the RL loop while allowing for some custom configuration. An outline of the training loop is shown below:

1. **Inference**

   1. Your code uses the ART client to perform an agentic workflow (usually executing several rollouts in parallel to gather data faster).
   2. Completion requests are routed to the ART server, which runs the model's latest LoRA in vLLM.
   3. As the agent executes, each `system`, `user`, and `assistant` message is stored in a Trajectory.
   4. When a rollout finishes, your code assigns a `reward` to its Trajectory, indicating the performance of the LLM.

2. **Training**
   1. When each rollout has finished, Trajectories are grouped and sent to the server. Inference is blocked while training executes.
   2. The server trains your model using GRPO, initializing from the latest checkpoint (or an empty LoRA on the first iteration).
   3. The server saves the newly trained LoRA to a local directory and loads it into vLLM.
   4. Inference is unblocked and the loop resumes at step 1.

This training loop runs until a specified number of inference and training iterations have completed.

## 🧩 Supported Models

ART should work with most vLLM/HuggingFace-transformers compatible causal language models, or at least the ones supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). Gemma 3 does not appear to be supported for the time being. If any other model isn't working for you, please let us know on [Discord](https://discord.gg/zbBHRUpwf4) or open an issue on [GitHub](https://github.com/openpipe/art/issues)!

## 🤝 Contributing

ART is in active development, and contributions are most welcome! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for more information.

## 📖 Citation

```bibtex
@misc{hilton2025art,
  author = {Brad Hilton and Kyle Corbitt and David Corbitt and Saumya Gandhi and Angky William and Bohdan Kovalenskyi and Andie Jones},
  title = {ART: Agent Reinforcement Trainer},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/openpipe/art}}
}
```

## ⚖️ License

This repository's source code is available under the [Apache-2.0 License](LICENSE).

## 🙏 Credits

ART stands on the shoulders of giants. While we owe many of the ideas and early experiments that led to ART's development to the open source RL community at large, we're especially grateful to the authors of the following projects:

- [Unsloth](https://github.com/unslothai/unsloth)
- [vLLM](https://github.com/vllm-project/vllm)
- [trl](https://github.com/huggingface/trl)
- [torchtune](https://github.com/pytorch/torchtune)
- [SkyPilot](https://github.com/skypilot-org/skypilot)

Finally, thank you to our partners who've helped us test ART in the wild! We're excited to see what you all build with it.

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg
[downloads-image]: https://img.shields.io/pypi/dm/openpipe-art?color=364fc7&logoColor=364fc7
