"""EXERCISE 2.1 — Mini autograd engine (scalar Value + backprop).

Goals: dynamic graph construction; reverse-topological backward() applying
the chain rule; ops add/mul/pow/div/relu/tanh/exp/sigmoid.

Run: pytest tests/test_02_deep_learning.py -k autograd
"""

import math


class Value:
    def __init__(self, data: float, _children=(), _op: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def _wrap(self, other):
        # TODO: Wrap raw numbers as Value.
        raise NotImplementedError("# TODO: Implement this")

    def __add__(self, other):
        # TODO: Forward add + backward (grad flows unchanged to both).
        raise NotImplementedError("# TODO: Implement this")

    def __mul__(self, other):
        # TODO: Forward mul + backward (local grads are the swapped inputs).
        raise NotImplementedError("# TODO: Implement this")

    def __pow__(self, power):
        # TODO: Forward power + backward (power * x^(power-1)).
        raise NotImplementedError("# TODO: Implement this")

    def relu(self):
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")

    def tanh(self):
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")

    def exp(self):
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")

    def sigmoid(self):
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")

    def backward(self):
        # TODO: Topo-sort the graph, seed grad=1.0, apply _backward in reverse.
        raise NotImplementedError("# TODO: Implement this")
