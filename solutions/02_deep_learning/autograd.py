"""Mini scalar autograd engine (micrograd-style).

Concepts: dynamic computational graph of ``Value`` nodes; backpropagation
applies the chain rule in reverse-topological order. Supports the ops needed
to train a small MLP: add/mul/pow/div/neg/sub/relu/tanh/exp/sigmoid.
"""

import math


class Value:
    def __init__(self, data: float, _children=(), _op: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # -- construction helpers ------------------------------------------------
    def _wrap(self, other):
        return other if isinstance(other, Value) else Value(other)

    # -- forward ops (each closes over its local derivatives) ----------------
    def __add__(self, other):
        other = self._wrap(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = self._wrap(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, power):
        assert isinstance(power, (int, float)), "only scalar powers supported"
        out = Value(self.data ** power, (self,), f"**{power}")

        def _backward():
            self.grad += power * self.data ** (power - 1) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-self._wrap(other))

    def __truediv__(self, other):
        return self * self._wrap(other) ** -1

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1.0 - s) * out.grad

        out._backward = _backward
        return out

    # -- backprop -------------------------------------------------------------
    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
