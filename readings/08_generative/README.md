# Module 08 — Generative Modeling History (VAE → GAN → Diffusion)

## Where this sits

Discriminative training (Modules 01–07) learns boundaries; generative
modeling learns the distribution itself — harder, weirder, and the source
of every modern foundation model. Three paradigms, three pathologies:
VAEs ignore their latents, GANs collapse and saturate, diffusions cost
hundreds of steps (then DDIM removes most of them).

## Builds on

- Module 01 (cross-entropy, KL, the Gaussian facts in guides/01 §5).
- Module 04/07 (the backbones these losses train).
- Module 06 (spectral norm, gradient penalty stabilize the GAN).

## Builds toward

- Reading any modern generative paper: they all speak ELBO / minimax /
  score-matching dialects of the same ideas.
- Research taste: each paradigm won by fixing the previous one's *training
  dynamics*, not its *expressivity*.

## The arc

Autoencoders → VAE (2013, principled latents + collapse) → GAN (2014,
spectacular samples + minimax pathologies) → WGAN-GP/SN (honest distances)
→ score matching + DDPM (2020, stable likelihood training) → DDIM/CFG
(fast, steerable sampling). The field converged on diffusion the way it
converged on Adam: stability wins.

## Exercises

| Note | Exercise | Question it answers |
|------|----------|---------------------|
| [vae.md](vae.md) | `exercises/08_generative/vae.py` | ELBO, reparameterization, collapse |
| [gan.md](gan.md) | `exercises/08_generative/gan.py` | Minimax vs non-saturating, WGAN-GP, SN |
| [diffusion.md](diffusion.md) | `exercises/08_generative/diffusion.py` | Schedules, DDPM/DDIM steps, guidance |

## Section readings

- Required: Kingma & Welling, "Auto-Encoding Variational Bayes" —
  [arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)
- Recommended: Ho et al., "Denoising Diffusion Probabilistic Models" —
  [arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
- Suggested: Goodfellow et al., "Generative Adversarial Nets" —
  [arxiv.org/abs/1406.2661](https://arxiv.org/abs/1406.2661)
