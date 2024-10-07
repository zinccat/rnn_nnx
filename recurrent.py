from __future__ import annotations

from typing import (
    Any,
    TypeVar,
)
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp

from flax import nnx

from jax import random
from flax.nnx import Module  # , first_from
from flax.nnx import initializers
from flax.nnx import sigmoid, tanh
from flax.nnx import Carry
from flax.typing import (
    Dtype,
    Initializer,
)

default_kernel_init = initializers.lecun_normal()
default_bias_init = initializers.zeros_init()

A = TypeVar('A')
Array = jax.Array
# Carry = Any



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


class LSTMCell(RNNCellBase):
    r"""LSTM cell.

  The mathematical definition of the cell is as follows

  .. math::
      \begin{array}{ll}
      i = \sigma(W_{ii} x + W_{hi} h + b_{hi}) \\
      f = \sigma(W_{if} x + W_{hf} h + b_{hf}) \\
      g = \tanh(W_{ig} x + W_{hg} h + b_{hg}) \\
      o = \sigma(W_{io} x + W_{ho} h + b_{ho}) \\
      c' = f * c + i * g \\
      h' = o * \tanh(c') \\
      \end{array}

  where x is the input, h is the output of the previous time step, and c is
  the memory.
  """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        gate_fn: Callable[..., Any] = sigmoid,
        activation_fn: Callable[..., Any] = tanh,
        kernel_init: Initializer = default_kernel_init,
        recurrent_kernel_init: Initializer = initializers.orthogonal(),
        bias_init: Initializer = initializers.zeros_init(),
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        carry_init: Initializer = initializers.zeros_init(),
        rngs: nnx.Rngs,
    ):
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.gate_fn = gate_fn
        self.activation_fn = activation_fn
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.carry_init = carry_init
        self.rngs = rngs

        dense_h = partial(
            nnx.Linear,
            in_features=hidden_features,
            out_features=hidden_features,
            use_bias=True,
            kernel_init=self.recurrent_kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            rngs=rngs,
        )
        # input and recurrent layers are summed so only one needs a bias.
        dense_i = partial(
            nnx.Linear,
            in_features=in_features,
            out_features=hidden_features,
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            rngs=rngs,
        )

        self.ii = dense_i()
        self.if_ = dense_i()
        self.ig = dense_i()
        self.io = dense_i()
        self.hi = dense_h()
        self.hf = dense_h()
        self.hg = dense_h()
        self.ho = dense_h()

    def __call__(self, carry, inputs):
        r"""A long short-term memory (LSTM) cell.

        Args:
          carry: the hidden state of the LSTM cell,
            initialized using ``LSTMCell.initialize_carry``.
          inputs: an ndarray with the input for the current time step.
            All dimensions except the final are considered batch dimensions.

        Returns:
          A tuple with the new carry and the output.
        """
        c, h = carry

        i = self.gate_fn(self.ii(inputs) + self.hi(h))
        f = self.gate_fn(self.if_(inputs) + self.hf(h))
        g = self.activation_fn(self.ig(inputs) + self.hg(h))
        o = self.gate_fn(self.io(inputs) + self.ho(h))
        new_c = f * c + i * g
        new_h = o * self.activation_fn(new_c)
        return (new_c, new_h), new_h

    def initialize_carry(
        self, rngs: nnx.Rngs | None, input_shape: tuple[int, ...]
    ) -> tuple[Array, Array]:
        """Initialize the RNN cell carry.

        Args:
          rng: random number generator passed to the init_fn.
          input_shape: a tuple providing the shape of the input to the cell.
        Returns:
          An initialized carry for the given RNN cell.
        """
        batch_dims = input_shape[:-1]
        if rngs is None:
            rngs = self.rngs
        mem_shape = batch_dims + (self.hidden_features,)
        c = self.carry_init(rngs, mem_shape, self.param_dtype)
        h = self.carry_init(rngs, mem_shape, self.param_dtype)
        return (c, h)

    @property
    def num_feature_axes(self) -> int:
        return 1

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
        activation_fn: Callable[..., Any] = tanh,
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

class RNN(Module):
    """The ``RNN`` module takes any :class:`RNNCellBase` instance and applies it over a sequence

    using :func:`flax.linen.scan`.
    """

    def __init__(
        self,
        cell: RNNCellBase,
        time_major: bool = False,
        return_carry: bool = False,
        reverse: bool = False,
        keep_order: bool = False,
        unroll: int = 1,
        # variable_axes: Mapping[
        #     CollectionFilter, InOutScanAxis
        # ] = FrozenDict(),
        # variable_broadcast: CollectionFilter = "params",
        # variable_carry: CollectionFilter = False,
        # split_rngs: Mapping[PRNGSequenceFilter, bool] = FrozenDict(
        # {"params": False}
    ):
        self.cell = cell
        self.time_major = time_major
        self.return_carry = return_carry
        self.reverse = reverse
        self.keep_order = keep_order
        self.unroll = unroll

    def __call__(
        self,
        inputs: Array,
        *,
        initial_carry: Carry | None = None,
        # init_key: PRNGKey | None = None,
        seq_lengths: Array | None = None,
        return_carry: bool | None = None,
        time_major: bool | None = None,
        reverse: bool | None = None,
        keep_order: bool | None = None,
    ):
        if time_major is None:
            time_major = self.time_major
        if return_carry is None:
            return_carry = self.return_carry
        if reverse is None:
            reverse = self.reverse
        if keep_order is None:
            keep_order = self.keep_order
    
        # Infer the number of batch dimensions from the input shape.
        # Cells like ConvLSTM have additional spatial dimensions.
        time_axis = (
            0 if time_major else inputs.ndim - (self.cell.num_feature_axes + 1)
        )

        # make time_axis positive
        if time_axis < 0:
            time_axis += inputs.ndim

        if time_major:
        # we add +1 because we moved the time axis to the front
            batch_dims = inputs.shape[1 : -self.cell.num_feature_axes]
        else:
            batch_dims = inputs.shape[:time_axis]

        carry: Carry = self.cell.initialize_carry(nnx.Rngs(0), inputs.shape) if initial_carry is None else initial_carry
        slice_carry = seq_lengths is not None and return_carry

        def scan_fn(carry: Carry, x: Array) -> tuple[Carry, Array]:
            carry, y = self.cell(carry, x)
            if slice_carry:
               return carry, (carry, y)
            return carry, y
            
        scan = nnx.scan(
            scan_fn,
            # length=inputs.shape[time_axis],
            in_axes=(Carry, time_axis),
            out_axes=(Carry, (Carry, time_axis)) if slice_carry else (Carry, time_axis),
            unroll=self.unroll,
        )

        scan_output = scan(carry, inputs)

        # Next we select the final carry. If a segmentation mask was provided and
        # return_carry is True we slice the carry history and select the last valid
        # carry for each sequence. Otherwise we just use the last carry.
        if slice_carry:
            assert seq_lengths is not None
            _, (carries, outputs) = scan_output
            # seq_lengths[None] expands the shape of the mask to match the
            # number of dimensions of the carry.
            carry = _select_last_carry(carries, seq_lengths)
        else:
            carry, outputs = scan_output

        # if reverse and keep_order:
        #     outputs = jax.tree_util.tree_map(
        #         lambda x: flip_sequences(
        #             x,
        #             seq_lengths,
        #             num_batch_dims=len(batch_dims),
        #             time_major=time_major,  # type: ignore
        #         ),
        #         outputs,
        #     )

        if return_carry:
            return carry, outputs
        else:
            return outputs

def _select_last_carry(sequence: A, seq_lengths: jnp.ndarray) -> A:
    last_idx = seq_lengths - 1

    def _slice_array(x: jnp.ndarray):
        return x[last_idx, jnp.arange(x.shape[1])]

    return jax.tree_util.tree_map(_slice_array, sequence)

@nnx.jit
def run_layer(layer, inputs):
    carry = layer.initialize_carry(None, inputs.shape)
    carry, _ = layer(carry, inputs)
    return carry

@nnx.jit
def run_model(model, inputs):
    out = model(inputs)
    return out

if __name__ == "__main__":
    rngs = nnx.Rngs(0)
    batch_size, seq_len, feature_size, hidden_size = 48, 32, 16, 64
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, feature_size))
    layer = SimpleCell(in_features=feature_size, hidden_features=hidden_size, out_features=4, rngs=rngs)
    # layer = LSTMCell(in_features=feature_size, hidden_features=hidden_size, out_features=4, rngs=rngs)
    rnn = RNN(layer, time_major=True)
    from timeit import timeit

    print(run_model(rnn, x).shape)
    print(timeit(lambda: run_model(rnn, x), number=100))