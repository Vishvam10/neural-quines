## Neural Quines

As [Wikipedia](https://en.wikipedia.org/wiki/Quine_(computing)) puts it, a **quine** is a computer program that takes no input and produces a copy of its own source code as its only output. A **neural quine** is the analogous concept for neural networks — a network that outputs its own weights. Something like this :

![Neural Quine](nnquine.png)

### Approaches Taken

#### Jacobian

Initially, I modeled the whole thing as a **regression problem** with multiple outputs. The idea was simple: treat the WHOLE neural network as a function $F_\theta(z)$ and try to make it output its own flattened parameters $\theta$:

$$
F(\theta) - \theta = 0
$$

The loss could then be written as:

$$
\mathcal{L} = \| F(\theta) - \theta \|
$$

and in principle, I could train the network using gradient descent or even **Newton’s method** if I computed the full Jacobian:

$$
J = \frac{\partial F(\theta)}{\partial \theta}, \quad
\theta \gets \theta - (J - I)^{-1} \big(F(\theta) - \theta\big)
$$


If we introduce a **scaling factor** $\alpha$ for the quine (to control the magnitude of self-replication), the loss becomes:

$$
\mathcal{L} = \| \alpha F(\theta) - \theta \|
$$

The Jacobian now includes the scaling :

$$
J = \frac{\partial (\alpha F(\theta))}{\partial \theta} = \alpha \frac{\partial F(\theta)}{\partial \theta}
$$

And the Newton-style update is updated accordingly:

$$
\theta \gets \theta - (J - I)^{-1} \big(\alpha F(\theta) - \theta \big)
$$


This allows you to control the "strength" of the quine with $\alpha$ while still using the Jacobian for Newton updates. This approach is mathematically neat, as it provides an exact fixed-point iteration — but in practice, there were several problems that I encountered later 🫠 (see [APPENDIX](#appendix--problems-with-the-naive-jacobian-approach)). As a result, I went for a non-Jacobian approach.

#### Non-Jacobian

Apparently, this topic was explored wayyy back in [Neural Network Quine, 2018](https://arxiv.org/abs/1803.05859) (I really should have read this first 😭). This paper proposes a clever alternative : **make the network weights queryable**. Instead of treating the problem as generic regression:

- Each input encodes which weight is being queried (usually via a one-hot vector) :
  
$$
\text{input}[i,j] \implies W_{ij} \approx F_\theta(\text{input})
$$


#### Improvements

Even with queryable inputs, the zero-quine problem remains. To bypass it, I followed [Evan Fletcher’s blog](https://evanfletcher42.com/2022/10/31/neural-quines/) (should've read this before too 😭 ... it's pretty good):

> We can forcibly avoid the zero quine by adding a parameter-free normalization layer after the trained dense layer. Both instance normalization and batch normalization (with all learned & running parameters disabled – no extra weights!) produce imperfect, but non-trivial, quines with relatively low error.

### How to run

```bash
# Clone the repo and go to root folder of repo
mkdir neural-quines
cd neural-quines
git clone https://github.com/Vishvam10/neural-quines


export PYTHONPATH=$(pwd)

# Jacobian
cd nnquine/jacobian
python train.py
python evaluate.py

# Non-Jacobian
cd nnquine/non_jacobian
python train.py
python evaluate.py
```

If you wish to contribute, kindly lint and format the code before raising a PR :

```bash
# Kindly lint and format the code if you plan to contribute
ruff check --config pyproject.toml
ruff check --select I --fix . --config pyproject.toml
ruff format --config pyproject.toml
```

### APPENDIX : Problems with the naive Jacobian approach

1. **The zero-quine problem**  
   - If all weights are zero, the network trivially satisfies $F(0) = 0$.  
   - Newton’s method or gradient descent will often converge to this trivial solution if the initialization isn’t perfect.  

2. **Jacobian scales poorly**  
   - For a network with $P$ parameters, the Jacobian $J$ is $P \times P$.  
   - Constructing, storing, and solving linear systems with $J - I$ becomes expensive and numerically unstable for even modest networks.  

3. **Single fixed input is limiting**  
   - The network needs some input to produce an output, but using a single “dummy” input only probes a tiny slice of the network’s behavior.  
   - This makes the reconstructed weights sensitive to the choice of input and can leave other weights poorly constrained.

4. **Nonlinear interactions and ill-conditioning**  
   - Neural quines are highly coupled : changing one weight affects multiple outputs.  
   - Newton’s method oscillates or overcorrects when the mapping is stiff, especially with normalization layers.  

