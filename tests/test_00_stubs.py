"""Stubs fail gracefully: every exercise entry point raises NotImplementedError."""

from importlib import import_module

import pytest

# (module, attribute, constructor kwargs, method to invoke or None).
# A None method means constructing/calling the attribute itself must raise.
CASES = [
    ("exercises.01_foundations.linear_logistic", "stable_sigmoid", {"z": [0.0]}, None),
    ("exercises.01_foundations.linear_logistic", "LinearRegression", {"l2": 0.0},
     ("fit", {"X": [[1.0], [2.0]], "y": [1.0, 2.0]})),
    ("exercises.01_foundations.perceptron_mlp", "MLP", {"layer_dims": [2, 2, 1]}, None),
    ("exercises.01_foundations.losses", "mse", {"y_pred": [1.0], "y_true": [1.0]}, None),
    ("exercises.01_foundations.losses", "cross_entropy",
     {"logits": [[1.0]], "targets": [0]}, None),
    ("exercises.02_deep_learning.autograd", "Value", {"data": 1.0}, ("backward", {})),
    ("exercises.02_deep_learning.layers", "Linear",
     {"in_features": 2, "out_features": 2}, None),
    ("exercises.02_deep_learning.conv2d", "Conv2D",
     {"in_channels": 1, "out_channels": 1, "kernel_size": 1}, None),
    ("exercises.03_training_algos.optimizers", "SGD", {"params": []}, None),
    ("exercises.03_training_algos.optimizers", "Adam", {"params": []}, None),
    ("exercises.03_training_algos.schedulers", "CosineAnnealingWarmRestarts",
     {"base_lr": 0.1, "T_0": 4}, None),
    ("exercises.03_training_algos.schedulers", "WarmupStableDecay",
     {"base_lr": 0.1, "warmup_steps": 2, "stable_steps": 2, "decay_steps": 2}, None),
    ("exercises.03_training_algos.training_loop", "train_loop",
     {"model": None, "loss_fn": None, "optimizer": None, "X": None, "y": None}, None),
    ("exercises.04_architectures.resnet", "ResidualBlock",
     {"in_channels": 1, "out_channels": 1}, None),
    ("exercises.04_architectures.lstm", "LSTMCell",
     {"input_size": 2, "hidden_size": 2}, None),
    ("exercises.04_architectures.transformer", "scaled_dot_product_attention",
     {"Q": None, "K": None, "V": None}, None),
    ("exercises.04_architectures.transformer", "DecoderBlock",
     {"d_model": 8, "n_heads": 2, "d_ff": 16}, None),
    ("exercises.05_optimization.second_order", "lbfgs_direction",
     {"g": None, "S": [], "Y": []}, None),
    ("exercises.05_optimization.second_order", "DampedNewton", {"params": []}, None),
    ("exercises.05_optimization.adaptive_history", "AdamW", {"params": []}, None),
    ("exercises.05_optimization.adaptive_history", "Lion", {"params": []}, None),
    ("exercises.05_optimization.sharpness", "SAM",
     {"base_optimizer": None, "params": []}, None),
    ("exercises.06_regularization.regularizers", "mixup_criterion",
     {"criterion": None, "pred": None, "y_a": None, "y_b": None, "lam": 0.5}, None),
    ("exercises.06_regularization.normalization", "WeightNormLinear",
     {"in_features": 2, "out_features": 2}, None),
    ("exercises.06_regularization.normalization", "gradient_penalty",
     {"critic": None, "real": None, "fake": None}, None),
    ("exercises.07_sequences.rnn_history", "GRUCell",
     {"input_size": 2, "hidden_size": 2}, None),
    ("exercises.07_sequences.attention_evolution", "alibi_slopes", {"n_heads": 2}, None),
    ("exercises.07_sequences.ssm", "discretize_diag",
     {"A": None, "B": None, "dt": None}, None),
    ("exercises.08_generative.vae", "kl_normal", {"mu": None, "logvar": None}, None),
    ("exercises.08_generative.gan", "gan_d_loss",
     {"real_logits": None, "fake_logits": None}, None),
    ("exercises.08_generative.diffusion", "linear_beta_schedule", {"T": 10}, None),
    ("exercises.08_generative.diffusion", "DiffusionSchedule", {"T": 10}, None),
]


@pytest.mark.parametrize("module,name,kwargs,method", CASES)
def test_stub_raises_not_implemented(module, name, kwargs, method):
    mod = import_module(module)
    with pytest.raises(NotImplementedError):
        obj = getattr(mod, name)(**kwargs)
        if method is not None:
            meth, mkwargs = method
            getattr(obj, meth)(**mkwargs)
