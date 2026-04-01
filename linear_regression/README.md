# Linear regression using JAX.

![demo](../assets/linreg.png)


### notes
To avoid loading the whole dataset we define the following class.

```
class BatchGenerator:

    def __init__(self, X, Y, batch_size):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.num_batches = (X.shape[0] - 1) // batch_size + 1

    def __iter__(self):
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = start + self.batch_size

            yield self.X[start:end], self.Y[start:end]

```
This enables mini-batch gradient descent. This class is not purely functional, but data loading can be imperative in JAX's framework. Model computation (`loss` and `update`) should be pure and JIT-compatible.

