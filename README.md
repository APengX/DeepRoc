# DeepRoc

DeepRoc is a tiny agent experimentation loop inspired by
[karpathy/autoresearch](https://github.com/karpathy/autoresearch).

The key takeaway from `autoresearch` is not "build a giant agent SDK".
It is:

1. constrain the agent's editable surface area,
2. make evaluation deterministic,
3. use git as the keep/discard mechanism,
4. store lightweight experiment memory outside git, and
5. keep the whole loop simple enough that the model can actually follow it.

This repository turns that idea into a reusable skeleton.

## What is implemented

DeepRoc ships a minimal runtime with:

- **prompt-as-code** via `program.md`
- **lab spec** via `deeproc.json`
- **proposal providers**
  - `replay`: load a proposal from JSON
  - `external-command`: call any CLI model/tool that reads JSON from stdin and prints JSON to stdout
- **mutable-scope enforcement** so edits stay inside allowed paths
- **evaluation command + metric regex**
- **git commit / keep / discard loop**
- **out-of-git journal** under `.deeproc/`

## Why this is similar to autoresearch

`autoresearch` is really a protocol:

`program -> edit code -> run evaluator -> parse metric -> keep/discard -> log result`

DeepRoc generalizes that protocol so you can use it for more than model training:

- benchmark-driven refactors
- prompt optimization
- code golf / latency tuning
- data pipeline improvements
- synthetic task research loops

## Repository layout

```text
src/deeproc/
  __main__.py   # CLI entrypoint
  core.py       # config, provider, git loop, evaluator, journaling

examples/simple_lab/
  program.md    # natural-language operating instructions
  deeproc.json  # lab config
  proposal.json # sample replay provider payload
  message.txt   # mutable file
  score.py      # deterministic evaluator

tests/
  test_deeproc.py
```

## Quick start

Run the included example:

```bash
python3 -m unittest
cd examples/simple_lab
PYTHONPATH=../../src python3 -m deeproc run --config deeproc.json
```

Expected output looks like:

```json
{"status": "keep", "metric": "3.000000", "kept": true, "commit": "abc1234", "output_path": ".deeproc/run-....log"}
```

The example is intentionally simple:

- the mutable file is `message.txt`
- the evaluator score is the number of times `agent` appears
- the replay provider proposes a deterministic improvement

## Config format

`deeproc.json`:

```json
{
  "name": "simple-lab",
  "program_path": "program.md",
  "mutable_paths": ["message.txt"],
  "provider": {
    "type": "replay",
    "path": "proposal.json"
  },
  "evaluation": {
    "command": ["python3", "score.py"],
    "metric_pattern": "score:\\s*([0-9.]+)",
    "goal": "max",
    "timeout_seconds": 10
  }
}
```

## Provider contract

Providers must return JSON in this shape:

```json
{
  "summary": "Short commit message",
  "rationale": "Why this change should help",
  "changes": [
    {
      "action": "write",
      "path": "message.txt",
      "content": "agent agent agent\n"
    }
  ]
}
```

Supported actions:

- `write`
- `replace`
- `append`

## Design advice if you want to grow this further

If your goal is to build something closer to a production agent framework,
the next useful layers are:

1. **Provider adapters** for OpenAI / Anthropic / Gemini / local models
2. **Patch-native edits** instead of only write/replace/append
3. **Multi-armed experiment strategy** instead of one proposal at a time
4. **Reviewer agent** that rejects risky edits before evaluation
5. **SQLite journal** for better experiment querying
6. **Parallel worktrees** for multi-agent or multi-run execution
7. **Artifact capture** for logs, diffs, metrics, and prompt traces

## One important lesson from autoresearch

The magic is mostly in the loop design, not in "agent cleverness".

If you make:

- the editable scope small,
- the objective explicit,
- the evaluator cheap,
- rollback automatic,
- and memory persistent,

even a very small framework becomes surprisingly effective.