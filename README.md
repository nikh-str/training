# toy training models using [JAX](https://docs.jax.dev/en/latest/)
collection of toy models.

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

In standard NumPy, you can modify an array directly—for example, updating the first element with `b[0] = 1`. In JAX, this kind of in-place update isn’t allowed. JAX arrays (DeviceArrays) are immutable, meaning their contents cannot be changed after creation.

This design ties back to JAX’s requirement for pure functions: computations should not have side effects or modify existing data. Allowing in-place changes would make it much harder for JAX to analyze and optimize code, especially when using just-in-time compilation.

Instead, JAX provides a functional alternative. You can write `b.at[0].set(1)`, whichThe key distinction—and in many ways the foundation of everything else—is that JAX is built around a functional programming style. This design makes it much easier for JAX to apply powerful transformations like compilation and automatic differentiation. The core idea to understand is simple: avoid writing code with side effects. doesn’t modify b directly. Rather, it returns a new array that is identical to b, except that the first element is updated to 1.

The key distinction—and in many ways the foundation of everything else—is that JAX is built around a functional programming style. This design makes it much easier for JAX to apply powerful transformations like compilation and automatic differentiation. The core idea to understand is simple: avoid writing code with side effects.
In practice, this means structuring your main JAX computations as functions that depend only on their inputs and produce outputs, without modifying anything external.
