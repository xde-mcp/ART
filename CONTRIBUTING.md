## Contributing to ART

Clone the repository:

```bash
git clone https://github.com/OpenPipe/agent-reinforcement-training.git
cd agent-reinforcement-training
```

Install the dependencies:

```bash
uv sync
```

Then follow the SkyPilot or Local Training instructions below.

> **Warning:** There is currently a bug with tool use functionality. The issue appears to be that vLLM does not return all the token log probabilities for tool use. Further investigation is needed to determine the exact cause. For now, teaching use case-specific tool use with non-tool use models is the recommended workaround.

### SkyPilot

Copy the `.env.example` file to `.env` and set the environment variables:

```bash
cp .env.example .env
```

Ensure you have a valid SkyPilot cloud available:

```bash
uv run sky check
```

Launch a cluster:

```bash
./launch-cluster.sh # you can pass any sky launch arguments here
```

SSH into the `art` cluster with VSCode or from the command line:

```bash
ssh art
```

When you're done, you can tear down the cluster with:

```bash
uv run sky down art
```

Make sure you are on a machine with at least one H100 or A100-80GB GPU. Machines equipped with lower-end GPUs may work, but training will be slower.

### "Temporal Clue" example

Now you can run the "Temporal Clue" example in `/examples/temporal-clue.ipynb`.

It has been tested with the `Qwen/Qwen2.5-14B-Instruct` model on a 1xH100 instance.

You can monitor training progress with Weights & Biases at https://wandb.ai/your-wandb-organization/agent-reinforcement-training.

You should see immediate improvement in `val/reward` after one step.

If you run into any issues, the training output is set to maximum verbosity. Copying the outputs such as the vLLM or torchtune logs, or copying/screenshotting the plotted packed tensors, may help me debug the issue.
