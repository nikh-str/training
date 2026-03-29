# ResNet compared to NODE

Resnet layer
$$h_{t+1} = h_t + f(h_t,\theta_t)$$

Neural ODE

$$\frac{dh(t)}{dt} = f(h(t),t,\theta)$$

ResNet appears to be the forward Euler-discretisation of a Neural ODE.

- ResNet has a fixed number of layers (discrete steps) and a fixed compute cost
- NODE has a continuous depth (solved with an ODE solver) and its compute cost varies depending on difficulty

### memory
- ResNet stores activations for back propagation and memory grows with depth
- NODE uses adjoint method (recomputes states) and has overall lower memory usage 
