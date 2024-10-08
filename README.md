# rnn_nnx
Implementation of RNN using Flax NNX, ported from Flax Linen.

## Usage
```python
@nnx.jit
def run_model(model, inputs):
    out = model(inputs)
    return out

if __name__ == "__main__":
    rngs = nnx.Rngs(0)
    batch_size, seq_len, feature_size, hidden_size = 48, 32, 16, 64
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, feature_size))
    # layer = SimpleCell(
    #     in_features=feature_size, hidden_features=hidden_size, rngs=rngs
    # )
    layer = LSTMCell(in_features=feature_size, hidden_features=hidden_size, rngs=rngs)
    rnn = RNN(layer, time_major=False, reverse=False, keep_order=False, unroll=1)

    print(run_model(rnn, x))

    bidirectional = Bidirectional(
        forward_rnn=rnn, backward_rnn=rnn, time_major=False, return_carry=True
    )
    ((c1, h1), (c2, h2)), output = run_model(bidirectional, x) # for lstm
    print(output.shape)
    print(c1.shape)
```