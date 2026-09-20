# Start here: how to use this gym (15 min read)

This repo is a **red → green** practice gym. The tests evaluate *your* code
by default (`exercises/`). An untouched checkout is mostly red — that is the
starting line, not a failure.

## The loop

1. **Read the guide first.** `guides/01_prereqs_math.md` covers the math you
   need; each exercise stub lists its prerequisites at the top.
2. **Implement one exercise** in `exercises/`.
3. **Run its suite:** `.venv/bin/python -m pytest tests/test_01_foundations.py -q`
4. **Debug with** `guides/02_debugging.md` (shape errors, NaNs, dead grads,
   flat loss — each maps to one of the S/G/C invariants below).
5. **Peek at `solutions/` only after your version passes** — or after 45
   honest minutes. Comparing approaches is where the learning sticks.
6. **Track progress:** `.venv/bin/python progress.py` prints a per-module
   PASS/FAIL table against your code.

## The two modes

```bash
.venv/bin/python -m pytest tests/ -q             # default: YOUR code (exercises/)
KATA_IMPL=solutions .venv/bin/python -m pytest tests/ -q   # reference health check
```

`tests/test_00_stubs.py` always checks that untouched stubs raise
`NotImplementedError` — implement a function and its stub test flips to
failure until the real suite passes. That flip is progress.

## The three invariants (S/G/C)

Every suite enforces the same discipline:

- **S — Shape.** Tensors must scale with the batch: `(B, C, H, W)` for
  images, `(B, T, D)` for sequences. When a test fails on shape, draw the
  diagram: write each tensor's shape on paper before you touch code.
- **G — Gradient flow.** After `.backward()`, every parameter needs a
  non-`None`, non-zero `.grad`. A `None` grad means your op detached from
  the graph; an all-zero grad means a dead path (e.g. a matrix that only
  ever sees a zero input).
- **C — Convergence.** A 10-step loop must strictly decrease the loss. If
  yours doesn't, the debugging guide's checklist applies in order: (1) is
  the loss even connected to the parameters? (2) is the lr sane? (3) is the
  gradient correct (finite-difference check)?

## Suggested order (zero-to-hero path)

1. `guides/01_prereqs_math.md` — numpy, chain rule, softmax/CE by hand.
2. Module 01 (foundations) → Module 02 (autograd, layers).
3. Module 03 (optimizers) → Module 09 (capstone: your first end-to-end run).
4. Modules 04, 07 (architectures, sequences) → 05, 06 (optimization history).
5. Module 08 (generative) with `RESEARCH_LOG.md` open beside you.

Finish with the capstone re-run: train the same task with Adam vs SGD and
explain the curves before looking at anyone else's explanation.
