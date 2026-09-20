"""Build the browsable static site into docs/ (GitHub Pages serves it as-is).

Usage: .venv/bin/python site/build.py
Requires: markdown (pip install markdown) — build-time only, never committed.

Content contract: every guide, reading note, exercise stub, and reference
solution is browsable; anything needing execution shows the exact 1-2 local
commands to validate (code never runs on the site).
"""

import html
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

EXERCISES = [
    # (section_dir, stub_stem, test_file, k_filter or None)
    ("01_foundations", "linear_logistic", "test_01_foundations.py",
     "ols or ridge or logistic or sigmoid"),
    ("01_foundations", "perceptron_mlp", "test_01_foundations.py",
     "perceptron or mlp or activation or numerical"),
    ("01_foundations", "losses", "test_01_foundations.py",
     "mse or bce or cross_entropy or stable"),
    ("02_deep_learning", "autograd", "test_02_deep_learning.py", "autograd"),
    ("02_deep_learning", "layers", "test_02_deep_learning.py",
     "linear or norm or dropout"),
    ("02_deep_learning", "conv2d", "test_02_deep_learning.py", "conv"),
    ("03_training_algos", "optimizers", "test_03_training.py",
     "sgd or adam or zero_grad"),
    ("03_training_algos", "schedulers", "test_03_training.py", "cosine or warmup"),
    ("03_training_algos", "training_loop", "test_03_training.py", "train_loop"),
    ("04_architectures", "resnet", "test_04_architectures.py",
     "residual or bottleneck"),
    ("04_architectures", "lstm", "test_04_architectures.py", "lstm"),
    ("04_architectures", "transformer", "test_04_architectures.py",
     "attention or mha or rmsnorm or rope or decoder"),
    ("05_optimization", "second_order", "test_05_optimization.py", "newton or lbfgs"),
    ("05_optimization", "adaptive_history", "test_05_optimization.py",
     "adagrad or rmsprop or adamw or lion or lookahead"),
    ("05_optimization", "sharpness", "test_05_optimization.py", "sam or swa"),
    ("06_regularization", "regularizers", "test_06_regularization.py",
     "dropconnect or stochastic or smoothing or mixup"),
    ("06_regularization", "normalization", "test_06_regularization.py",
     "weightnorm or spectral or penalty"),
    ("07_sequences", "rnn_history", "test_07_sequences.py", "rnn or gru or vanishing"),
    ("07_sequences", "attention_evolution", "test_07_sequences.py",
     "alibi or window or gqa"),
    ("07_sequences", "ssm", "test_07_sequences.py", "ssm or discretize"),
    ("08_generative", "vae", "test_08_generative.py", "vae or kl or elbo"),
    ("08_generative", "gan", "test_08_generative.py",
     "gan or wgan or spectral_normalize"),
    ("08_generative", "diffusion", "test_08_generative.py",
     "diffusion or ddpm or ddim or guidance"),
    ("09_capstone", "end_to_end", "test_09_capstone.py", None),
]

SECTION_TITLES = {
    "01_foundations": "01 · Foundations", "02_deep_learning": "02 · Deep Learning",
    "03_training_algos": "03 · Training", "04_architectures": "04 · Architectures",
    "05_optimization": "05 · Optimization", "06_regularization": "06 · Regularization",
    "07_sequences": "07 · Sequences", "08_generative": "08 · Generative",
    "09_capstone": "09 · Capstone",
}

CSS = """
:root { --accent: #166534; --ink: #1c1c1c; --muted: #555; --line: #e2e2e2; --code: #f1f3f4; }
* { box-sizing: border-box; }
body { margin: 0; color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
  line-height: 1.6; }
nav.top { border-bottom: 1px solid var(--line); padding: 0.6rem 1.2rem; }
nav.top a { color: var(--accent); text-decoration: none; margin-right: 1.1rem; font-weight: 600; }
nav.top a:hover { text-decoration: underline; }
main { max-width: 76ch; margin: 0 auto; padding: 1.5rem 1.2rem 3rem; }
h1 { font-size: 1.7rem; line-height: 1.25; }
h2 { font-size: 1.3rem; margin-top: 2rem; border-bottom: 1px solid var(--line); padding-bottom: 0.3rem; }
a { color: var(--accent); }
code { background: var(--code); padding: 0.1em 0.35em; border-radius: 4px;
  font-size: 0.88em; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
pre { background: var(--code); padding: 1rem; border-radius: 6px; overflow-x: auto; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.93rem; }
th, td { border: 1px solid var(--line); padding: 0.45rem 0.6rem; text-align: left; vertical-align: top; }
th { background: #f7f7f5; }
.cmd { background: #14231a; color: #d7e8dc; padding: 0.8rem 1rem; border-radius: 6px;
  overflow-x: auto; font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 0.88rem; }
.cmd .prompt { color: #7fc79a; }
details.solution { border: 1px solid var(--line); border-radius: 6px; margin: 1rem 0; }
details.solution summary { cursor: pointer; padding: 0.6rem 1rem; font-weight: 600;
  background: #f7f7f5; border-radius: 6px; }
details.solution div.body { padding: 0 1rem 1rem; }
p.meta { color: var(--muted); font-size: 0.92rem; }
footer { border-top: 1px solid var(--line); padding: 1rem 1.2rem 2rem;
  color: var(--muted); font-size: 0.88rem; }
ul.checks { list-style: none; padding-left: 0; }
ul.checks li { padding: 0.15rem 0; }
ul.checks li code { font-size: 0.82em; }
"""

NAV = ('<nav class="top"><a href="{root}index.html">Kata</a>'
       '<a href="{root}guides/00_start_here.html">Guides</a>'
       '<a href="{root}readings.html">Readings</a>'
       '<a href="{root}exercises.html">Exercises</a>'
       '<a href="{root}RESEARCH_LOG.html">Research log</a></nav>')

FOOTER = ('<footer>Validate locally after solving: '
          '<code>.venv/bin/python -m pytest tests/ -q</code> '
          '(your code) · <code>.venv/bin/python progress.py</code> (tracker). '
          'MIT licensed.</footer>')


def md_to_html(text: str) -> str:
    out = markdown.markdown(text, extensions=["tables", "fenced_code", "toc"])
    # Local .md links become same-dir .html links.
    out = re.sub(r'href="([^"]*?)\.md(#?[^"]*)"', r'href="\1.html\2"', out)
    return out


def page(title: str, body: str, depth: int) -> str:
    root = "../" * depth
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{html.escape(title)} · ML Kata</title>"
            f"<link rel=\"stylesheet\" href=\"{root}style.css\"></head><body>"
            + NAV.format(root=root) + f"<main>{body}</main>" + FOOTER + "</body></html>")


def write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def code_block(source: str) -> str:
    return f"<pre><code>{html.escape(source)}</code></pre>"


def validate_cmd(test_file: str, kfilter: str | None) -> str:
    cmd = f".venv/bin/python -m pytest tests/{test_file} -q"
    if kfilter:
        cmd += f" -k &quot;{kfilter}&quot;"
    return (f"<p>Solve the stub, then validate with one command "
            f"(runs <em>your</em> code):</p><div class=\"cmd\">"
            f"<span class=\"prompt\">$</span> {cmd}</div>")


def stub_title(section: str, stem: str) -> str:
    src = (ROOT / "exercises" / section / f"{stem}.py").read_text().splitlines()
    for line in src:
        s = line.strip().strip('"')
        if s.startswith("EXERCISE"):
            return s
    return stem


def test_names(test_file: str) -> list[str]:
    src = (ROOT / "tests" / test_file).read_text()
    return re.findall(r"^def (test_\w+)", src, flags=re.M)


def build():
    if DOCS.exists():
        for p in sorted(DOCS.rglob("*"), reverse=True):
            p.unlink() if p.is_file() or p.is_symlink() else None
        for p in sorted(DOCS.rglob("*"), reverse=True):
            if p.is_dir():
                p.rmdir()
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / ".nojekyll").write_text("")
    write(DOCS / "style.css", CSS)

    # ---- index (from README) ----
    index_body = md_to_html((ROOT / "README.md").read_text())
    write(DOCS / "index.html", page("ML practice gym", index_body, 0))

    # ---- guides ----
    for md in sorted((ROOT / "guides").glob("*.md")):
        body = md_to_html(md.read_text())
        write(DOCS / "guides" / f"{md.stem}.html",
              page(md.stem.replace("_", " "), body, 1))

    # ---- research log ----
    write(DOCS / "RESEARCH_LOG.html",
          page("Research log", md_to_html((ROOT / "RESEARCH_LOG.md").read_text()), 0))

    # ---- readings hub ----
    hub = ["<h1>Readings</h1>",
           "<p>Pre-digested notes per section and exercise: introduction, concept "
           "hierarchy, intuition with the key math, and required / recommended / "
           "suggested sources.</p>"]
    for section in sorted({e[0] for e in EXERCISES}):
        hub.append(f"<h2>{SECTION_TITLES[section]}</h2><ul>")
        hub.append(f"<li><a href=\"readings/{section}/index.html\">Section overview</a></li>")
        for s, stem, _, _ in [e for e in EXERCISES if e[0] == section]:
            hub.append(f"<li><a href=\"readings/{s}/{stem}.html\">"
                       f"{html.escape(stub_title(s, stem))}</a></li>")
        hub.append("</ul>")
    write(DOCS / "readings.html", page("Readings", "\n".join(hub), 0))

    # ---- readings pages ----
    for section in sorted({e[0] for e in EXERCISES}):
        src = ROOT / "readings" / section / "README.md"
        body = md_to_html(src.read_text()) if src.exists() else ""
        write(DOCS / "readings" / section / "index.html",
              page(f"{SECTION_TITLES[section]} · overview", body, 2))
        for s, stem, _, _ in [e for e in EXERCISES if e[0] == section]:
            note = ROOT / "readings" / s / f"{stem}.md"
            body = md_to_html(note.read_text())
            body += (f"<p class=\"meta\">Exercise stub: "
                     f"<a href=\"../../exercises/{s}/{stem}.html\">exercises/{s}/{stem}.py</a></p>")
            write(DOCS / "readings" / s / f"{stem}.html",
                  page(stub_title(s, stem), body, 2))

    # ---- exercises hub ----
    hub = ["<h1>Exercises</h1>",
           "<p>Implement the stub, then run its validate command. "
           "Reference solutions hide behind a disclosure on each page.</p>"]
    for section in sorted({e[0] for e in EXERCISES}):
        hub.append(f"<h2>{SECTION_TITLES[section]}</h2><ul>")
        for s, stem, test_file, kf in [e for e in EXERCISES if e[0] == section]:
            hub.append(f"<li><a href=\"exercises/{s}/{stem}.html\">"
                       f"{html.escape(stub_title(s, stem))}</a></li>")
        hub.append("</ul>")
    write(DOCS / "exercises.html", page("Exercises", "\n".join(hub), 0))

    # ---- exercise pages ----
    for section, stem, test_file, kf in EXERCISES:
        stub = (ROOT / "exercises" / section / f"{stem}.py").read_text()
        sol = (ROOT / "solutions" / section / f"{stem}.py").read_text()
        note = f"readings/{section}/{stem}.md".replace(".md", ".html")
        checks = "".join(f"<li><code>{t}</code></li>" for t in test_names(test_file))
        body = (f"<h1>{html.escape(stub_title(section, stem))}</h1>"
                f"<p class=\"meta\">Reading: <a href=\"../../{note}\">{note}</a> · "
                f"Suite: <code>tests/{test_file}</code></p>"
                + validate_cmd(test_file, kf)
                + f"<h2>Stub</h2>{code_block(stub)}"
                + f"<h2>Suite checks</h2><ul class=\"checks\">{checks}</ul>"
                + f"<details class=\"solution\"><summary>Reference solution "
                  f"(open after your version passes)</summary>"
                  f"<div class=\"body\">{code_block(sol)}</div></details>")
        write(DOCS / "exercises" / section / f"{stem}.html",
              page(stub_title(section, stem), body, 2))

    # ---- per-section exercise indexes ----
    for section in sorted({e[0] for e in EXERCISES}):
        items = [e for e in EXERCISES if e[0] == section]
        parts = [f"<h1>{SECTION_TITLES[section]}</h1>"]
        for s, stem, test_file, kf in items:
            parts.append(f"<h2>{html.escape(stub_title(s, stem))}</h2>"
                         + validate_cmd(test_file, kf)
                         + f"<p><a href=\"{stem}.html\">Open exercise page</a> · "
                         f"<a href=\"../../readings/{s}/{stem}.html\">Reading note</a></p>")
        # gather checks across the section's test files without duplicates
        seen, ordered = set(), []
        for _, _, tf, _ in items:
            for t in test_names(tf):
                if t not in seen:
                    seen.add(t)
                    ordered.append(t)
        checks = "".join(f"<li><code>{t}</code></li>" for t in ordered)
        parts.append(f"<h2>Suite checks in this module</h2><ul class=\"checks\">{checks}</ul>")
        write(DOCS / "exercises" / section / "index.html",
              page(SECTION_TITLES[section], "\n".join(parts), 2))

    count = sum(1 for _ in DOCS.rglob("*.html"))
    print(f"built {count} pages into docs/")


if __name__ == "__main__":
    build()
