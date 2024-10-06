from __future__ import annotations

from typing import Any
from collections.abc import Callable

import jax
import jax.numpy as jnp

from flax import nnx

from flax.nnx import Module  # , first_from
from flax.nnx import initializers
from flax.typing import (
    Dtype,
    Initializer,
)

Array = jax.Array
Carry = Any

class RNNCellBase(Module):
    """RNN cell base class."""

    def initialize_carry(
        self, rngs: nnx.Rngs | None, input_shape: tuple[int, ...]
      ) -> Carry:
        """Initialize the RNN cell carry.

        Args:
          rng: random number generator passed to the init_fn.
          input_shape: a tuple providing the shape of the input to the cell.

        Returns:
          An initialized carry for the given RNN cell.
        """
        raise NotImplementedError

    @property
    def num_feature_axes(self) -> int:
        """Returns the number of feature axes of the RNN cell."""
        raise NotImplementedError


class SimpleCell(RNNCellBase):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,  # not inferred from carry for now
        out_features: int,
        *,
        dtype: Dtype = jnp.float32,
        param_dtype: Dtype = jnp.float32,
        carry_init: Initializer = initializers.zeros_init(),
        residual: bool = False,
        activation_fn: Callable[..., Any] = jax.nn.tanh,
        kernel_init: Initializer = initializers.lecun_normal(),
        recurrent_kernel_init: Initializer = initializers.orthogonal(),
        bias_init: Initializer = initializers.zeros_init(),
        rngs: nnx.Rngs,
    ):
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.carry_init = carry_init
        self.residual = residual
        self.activation_fn = activation_fn
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.rngs = rngs

        # self.hidden_features = carry.shape[-1]
        # input and recurrent layers are summed so only one needs a bias.
        self.dense_h = nnx.Linear(
            in_features=self.hidden_features,
            out_features=self.hidden_features,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.recurrent_kernel_init,
            rngs=rngs,
        )
        self.dense_i = nnx.Linear(
            in_features=self.in_features,
            out_features=self.hidden_features,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            rngs=rngs,
        )

    def __call__(self, carry, inputs, *, rngs: nnx.Rngs | None = None):
        """Simple cell.

        Args:
          carry: the hidden state of the Simple cell,
            initialized using ``SimpleCell.initialize_carry``.
          inputs: an ndarray with the input for the current time step.
            All dimensions except the final are considered batch dimensions.

        Returns:
          A tuple with the new carry and the output.
        """

        new_carry = self.dense_i(inputs) + self.dense_h(carry)
        if self.residual:
            new_carry += carry
        new_carry = self.activation_fn(new_carry)
        return new_carry, new_carry

    def initialize_carry(self, rngs: nnx.Rngs | None, input_shape: tuple[int, ...]):
        """Initialize the RNN cell carry.

        Args:
          rng: random number generator passed to the init_fn.
          input_shape: a tuple providing the shape of the input to the cell.

        Returns:
          An initialized carry for the given RNN cell.
        """
        if rngs is None:
            rngs = self.rngs
        batch_dims = input_shape[:-1]
        mem_shape = batch_dims + (self.hidden_features,)
        return self.carry_init(rngs, mem_shape, self.param_dtype)

    @property
    def num_feature_axes(self) -> int:
        return 1


if __name__ == "__main__":
    rng = nnx.Rngs(0)
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 3))
    layer = SimpleCell(in_features=3, hidden_features=4, out_features=4, rngs=rng)
    carry = layer.initialize_carry(rng, x.shape)
    print(layer(carry, x))
    