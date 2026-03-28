# toy training models using [JAX](https://docs.jax.dev/en/latest/)

JAX is a high-performance numerical computing library that combines NumPy-like APIs with automatic differentiation, vectorisation and Just in Time Compilation (JIT).

`vmap` automatically applies a function over batches, avoiding manual loops

```python
from jax import vmap

def single_loss(w, x, y):
    return (w * x - y) ** 2

batched_loss = vmap(single_loss, in_axes=(None, 0, 0))

print(batched_loss(w, x, y))
```

`jit` compiles your function to run much faster (using XLA under the hood).


```python
from jax import jit

@jit
def update(w, x, y, lr=0.1):
    g = grad(loss_fn)(w, x, y)
    return w - lr * g
```

The first call compiles the function, and later calls run much faster—great for training loops.


`grad` computes derivatives of a function
```python
import jax.numpy as jnp
from jax import grad

def loss_fn(w, x, y):
    preds = w * x
    return jnp.mean((preds - y) ** 2)

# Gradient w.r.t. parameter w
grad_loss = grad(loss_fn)

w = 2.0
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([2.0, 4.0, 6.0])

print(grad_loss(w, x, y))
```
this gives the gradient of a loss, which one would use in gradient descent
```
w = w - 0.1 * grad_loss(w, x, y)
```

