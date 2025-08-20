

# Generative Adversarial Networks (GANs)

## 1. Overview of Deep Generative Modeling
We be## 4. Practical Training and Loss Variant## 5. Architectural Considerations
The choice of## Reference## 1. Introduction
In many modern generative-modeling tasks, we wish to approximate an unknown data ## 7. Special Case: Generative Adversarial Networks (GANs)
For the **Jensen–Shannon divergence**, one sets
$$
  f(u)=u\ln u - (u+1)\ln\bigl(\tfrac{u+1}{2}\bigr),
  \quad
  f^*(t)=-\log\bigl(1-e^t\bigr),\; t<0,
  \quad
  \sigma_f(v)=-\log\bigl(1+e^{-v}\bigr).
$$
Identifying the sigmoid discriminator
$D_w(x)=1/(1+e^{-V_w(x)})$ gives the classical GAN minimax game:
$$
  \min_\theta\max_w\;\mathbb{E}_{x\sim p_x}[\log D_w(x)]
    + \mathbb{E}_{z\sim p_z}[\log\bigl(1-D_w(g_\theta(z))\bigr)].
$$
This recovers Goodfellow et al.'s original adversarial framework as a special case of VDM.x$ on a sample space $\mathcal{X}$ by a parameterized model $p_\theta$. We assume that we can:

- Draw i.i.d. samples $x\sim p_x$ (the training set), but cannot evaluate $p_x(x)$ in closed form.
- Generate samples from $p_\theta$ via a differentiable "generator" network
  $$
     g_\theta\colon \mathcal{Z}\to\mathcal{X},\quad z\sim p_z,
  $$
  so that $\hat{x} = g_\theta(z)$ has distribution $p_\theta$.

A powerful way to fit $p_\theta$ to $p_x$ is by minimizing an **f-divergence** between them:
$$
  \theta^* = \mathop{\arg\min}_\theta\;D_f\bigl(p_x\| p_\theta\bigr),
$$
where $D_f$ can be any member of the wide family of convex divergences. In practice, neither density is known analytically, so we rely on sample-based (Monte Carlo) approximations combined with a _variational_ (dual) formulation of the divergence., I. et al. "Generative Adversarial Nets." NeurIPS 2014.
- Nowozin, S., Cseke, B., & Tomioka, R. "f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization." NeurIPS 2016.
- Arjovsky, M., Chintala, S., & Bottou, L. "Wasserstein GAN." ICML 2017.
- Radford, A., Metz, L., & Chintala, S. "Unsupervised Representation Learning with Deep Convolutional GANs." ICLR 2016.ork architectures greatly affects sample quality and training stability:

- **Generator**  
   Typically a deep deconvolutional network (DCGAN) or style-based architecture (StyleGAN) that transforms noise $z\sim\mathcal{N}(0,I)$ into images.

- **Discriminator**  
   Usually a convolutional classifier that outputs a scalar "real vs. fake" score.

- **Conditional GAN (cGAN)**  
   Incorporates side information $y$ by feeding $(z,y)$ to the generator and $(x,y)$ to the discriminator, enabling class-conditional or image-to-image tasks.e original minimax loss provides a principled objective, in practice several variants improve gradient behavior:

- **Non-saturating generator loss**  
   $$
   \mathcal{L}_G(\theta)
   = -\mathbb{E}_{z\sim p_Z}[\ln D_\phi(G_\theta(z))],
   $$
   which mitigates vanishing gradients early in training.

- **Wasserstein GAN (WGAN)**  
   Replaces JS with the Earth-Mover's (Wasserstein) distance:
   $$
     W(p_X,p_\theta)
     = \inf_{\gamma\in\Pi(p_X,p_\theta)}\mathbb{E}_{(x,y)\sim\gamma}\|x-y\|,
   $$
   and enforces a 1-Lipschitz discriminator via weight clipping or gradient penalty (WGAN-GP).

- **f-GAN**  
   Generalizes adversarial learning to arbitrary $f$-divergences by maximizing the corresponding variational bound.ataset

$$D = \{x_1, x_2, \dots, x_n\}, \quad x_i\in\mathbb{R}^d,\;x_i\sim p_X(\text{unknown}).$$

A deep generative model aims to learn a parameterized distribution $p_\theta(x)$ that closely approximates the true data law $p_X(x)$ and allows efficient sampling. A canonical three-step recipe underlies most constructions:

1. Choose a simple latent prior $z\in\mathbb{R}^k$, e.g. $p_Z(z)=\mathcal{N}(0,I)$.
2. Define a deterministic generator

   $$g_\theta:\;\mathbb{R}^k\to\mathbb{R}^d,\quad \tilde{x} = g_\theta(z),$$

   which induces the (in general implicit) model distribution $p_\theta(x)$.
3. Select a divergence or distance $D(p_X\| p_\theta)\ge0$ with 

   $$D(p_X\| p_\theta)=0\iff p_X=p_\theta,$$

   and solve

   $$\theta^* = \arg\min_\theta D\bigl(p_X\| p_\theta\bigr).$$

Generative Adversarial Networks (GANs) instantiate this recipe by choosing the Jensen–Shannon divergence and optimizing it via a minimax game between two neural networks.


## 2. f-Divergences and Their Variational Forms
An $f$-divergence between two densities $p$ and $q$ on $\mathcal{X}$ is defined by

$$
D_f(p\| q)
= \int_{\mathcal{X}} q(x)\,f\bigl(\tfrac{p(x)}{q(x)}\bigr)\,dx,
$$

where $f:\,\mathbb{R}^+\to\mathbb{R}$ is convex with $f(1)=0$. Key examples:

- **Kullback–Leibler (KL) divergence**

   $$f(u)=u\ln u,\quad
   D_{\mathrm{KL}}(p\| q)
   = \int p(x)\ln\frac{p(x)}{q(x)}\,dx.
   $$

- **Jensen–Shannon (JS) divergence**

   $$D_{\mathrm{JS}}(p,q)
     = \tfrac{1}{2}D_{\mathrm{KL}}\Bigl(p\Big\|\tfrac{p+q}{2}\Bigr)
       + \tfrac{1}{2}D_{\mathrm{KL}}\Bigl(q\Big\|\tfrac{p+q}{2}\Bigr).
   $$

By convex duality each $f$-divergence admits the variational (Fenchel) representation

$$
D_f(p\| q)
= \sup_{T\in\mathcal{T}}\Bigl\{\mathbb{E}_{x\sim p}[T(x)]
- \mathbb{E}_{x\sim q}[f^*(T(x))]\Bigr\},
$$

where $f^*$ is the convex conjugate of $f$ and $\mathcal{T}$ is a rich class of functions (e.g., neural networks). This dual form paves the way for GAN-style adversarial optimization.


## 3. The GAN Minimax Formulation
Goodfellow et al. (2014) proposed to approximate the JS divergence by introducing:

- A **generator** $G_\theta(z)$ that maps noise $z\sim p_Z$ to synthetic samples.
- A **discriminator** $D_\phi(x)\in(0,1)$ that estimates the probability a sample is real.

They play the two-player zero-sum game with value function

$$
V(\theta,\phi)
= \mathbb{E}_{x\sim p_X}[\ln D_\phi(x)]
+ \mathbb{E}_{z\sim p_Z}[\ln\bigl(1 - D_\phi(G_\theta(z))\bigr)].
$$

Training alternates between

$$
\phi \leftarrow \arg\max_\phi V(\theta,\phi),
\quad
\theta \leftarrow \arg\min_\theta V(\theta,\phi).
$$

At the optimal discriminator

$$D_\phi^*(x)=\frac{p_X(x)}{p_X(x)+p_\theta(x)},$$

the minimax value reduces to

$$\min_\theta\max_\phi V(\theta,\phi)
= 2\,D_{\mathrm{JS}}\bigl(p_X\| p_\theta\bigr) - 2\ln2.$$

Thus, GAN training approximately minimizes the JS divergence between the real and generated distributions.


## 4. Practical Training and Loss Variants
While the original minimax loss provides a principled objective, in practice several variants improve gradient behavior:

-  **Non‐saturating generator loss**  
   \[
   \mathcal L_G(\theta)
   = -\mathbb E_{z\sim p_Z}[\ln D_\phi(G_\theta(z))],
   \]
   which mitigates vanishing gradients early in training.

-  **Wasserstein GAN (WGAN)**  
   Replaces JS with the Earth‐Mover’s (Wasserstein) distance:
   \[
     W(p_X,p_\theta)
     = \inf_{\gamma\in\Pi(p_X,p_\theta)}\mathbb E_{(x,y)\sim\gamma}\|x-y\|,
   \]
   and enforces a 1‐Lipschitz discriminator via weight clipping or gradient penalty (WGAN‐GP).

-  **f‐GAN**  
   Generalizes adversarial learning to arbitrary \(f\)-divergences by maximizing the corresponding variational bound.


## 5. Architectural Considerations
The choice of network architectures greatly affects sample quality and training stability:

-  **Generator**  
   Typically a deep deconvolutional network (DCGAN) or style‐based architecture (StyleGAN) that transforms noise \(z\sim\mathcal N(0,I)\) into images.

-  **Discriminator**  
   Usually a convolutional classifier that outputs a scalar “real vs. fake” score.

- **Conditional GAN (cGAN)**  
   Incorporates side information $y$ by feeding $(z,y)$ to the generator and $(x,y)$ to the discriminator, enabling class-conditional or image-to-image tasks.


## 6. Theoretical Insights and Challenges
Despite their empirical success, GANs face several theoretical and practical hurdles:

- **Nash equilibrium**  
   Existence is guaranteed under infinite capacity, but convergence to the equilibrium is not.

- **Mode collapse**  
   The generator may map many latent codes $z$ to a few distinct outputs. Remedies include minibatch discrimination, unrolled GANs, and PacGAN.

- **Training instability**  
   Sensitive to hyperparameters, learning rates, and loss balancing. Techniques such as spectral normalization and two-time-scale updates (TTUR) help stabilize learning.


## 7. Sampling and Applications
Once $\theta^*$ is obtained, generation is straightforward:

1. Sample $z\sim p_Z$.
2. Compute $x = G_{\theta^*}(z)$.

GANs excel at producing high-fidelity outputs across modalities:

- **Images:** DCGAN, StyleGAN architectures.  
- **Image-to-image translation:** pix2pix, CycleGAN.  
- **Super-resolution:** SRGAN.  
- **Audio and speech synthesis**, **text generation** (via sequence GANs).


## References
-  Goodfellow, I. et al. “Generative Adversarial Nets.” NeurIPS 2014.
-  Nowozin, S., Cseke, B., & Tomioka, R. “f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization.” NeurIPS 2016.
-  Arjovsky, M., Chintala, S., & Bottou, L. “Wasserstein GAN.” ICML 2017.
-  Radford, A., Metz, L., & Chintala, S. “Unsupervised Representation Learning with Deep Convolutional GANs.” ICLR 2016.

# Variational Auto-Encoders (VAEs)

## 1. Introduction
In many modern generative‐modeling tasks, we wish to approximate an unknown data distribution \(p_x\) on a sample space \(\mathcal X\) by a parameterized model \(p_\theta\).  We assume that we can:

- Draw i.i.d. samples \(x\sim p_x\) (the training set), but cannot evaluate \(p_x(x)\) in closed form.
- Generate samples from \(p_\theta\) via a differentiable “generator” network
  \[
     g_\theta\colon \mathcal Z\to\mathcal X,\quad z\sim p_z,
  \]
  so that \(\hat x = g_\theta(z)\) has distribution \(p_\theta\).

A powerful way to fit \(p_\theta\) to \(p_x\) is by minimizing an **f-divergence** between them:
\[
  \theta^* = \mathop{\arg\min}_\theta\;D_f\bigl(p_x\Vert p_\theta\bigr),
\]
where \(D_f\) can be any member of the wide family of convex divergences.  In practice, neither density is known analytically, so we rely on sample-based (Monte Carlo) approximations combined with a _variational_ (dual) formulation of the divergence.

## 2. Definition of f-Divergences
Given two distributions $P$ and $Q$ on $\mathcal{X}$ with densities $p(x)$ and $q(x)$, an **f-divergence** is defined by
$$
  D_f(P\| Q)
  = \int_{\mathcal{X}} q(x)\,f\bigl(p(x)/q(x)\bigr)\,dx,
$$
where $f:(0,\infty)\to\mathbb{R}$ is convex and satisfies $f(1)=0$. Important examples:

- **Kullback–Leibler (KL):** $f(u)=u\log u$. Measures relative entropy.
- **Pearson χ²:** $f(u)=(u-1)^2$. Emphasizes squared deviations.
- **Jensen–Shannon (JS):** $f(u)=u\ln u - (u+1)\ln\tfrac{u+1}{2}$. Symmetrized and smoothed version of KL.

Each choice of $f$ induces a different training objective and generative behavior.

## 3. Monte Carlo Estimation of Expectations
Any objective expressed as an expectation under $p_x$ or $p_\theta$ can be estimated via Monte Carlo sampling and the Law of Large Numbers (LLN). For example, if
$$
  I = \mathbb{E}_{x\sim p_x}\bigl[h(x)\bigr] = \int h(x)\,p_x(x)\,dx,
$$
then with i.i.d. samples $x_1,\dots,x_n\sim p_x$,
$$
  \frac{1}{n}\sum_{i=1}^n h(x_i) \xrightarrow[n\to\infty]{a.s.}\; I.
$$
Analogous sample averages approximate expectations under $p_\theta$. This makes it possible to optimize objectives that involve both distributions without ever computing their densities explicitly.

## 4. Variational (Dual) Representation via Convex Conjugate
The key to tractable optimization is a variational lower bound on $D_f$, obtained via the **Fenchel–Legendre transform**. Define the convex conjugate
$$
  f^*(t) = \sup_{u>0}\{\,u\,t - f(u)\}.
$$
By Fenchel–Young we have
$$
  f(u) = \sup_{t\in\mathrm{dom}(f^*)}\{\,u\,t - f^*(t)\}.
$$
Substitute $u = p_x(x)/p_\theta(x)$ into the f-divergence and interchange supremum and integral:
$$
  \begin{aligned}
  D_f(p_x\| p_\theta)
  &= \int p_\theta(x)\,f\bigl(p_x(x)/p_\theta(x)\bigr)\,dx \\
  &= \int p_\theta(x)\sup_t\{t\,\tfrac{p_x(x)}{p_\theta(x)} - f^*(t)\}\,dx \\
  &\ge \sup_{T}\Bigl[\mathbb{E}_{x\sim p_x}[T(x)] - \mathbb{E}_{x\sim p_\theta}[f^*(T(x))]\Bigr],
  \end{aligned}
$$
where the supremum is taken over all measurable functions $T:\mathcal{X}\to\mathrm{dom}(f^*)$. Equality holds if $T(x)=f'\bigl(p_x(x)/p_\theta(x)\bigr)$ lies in the function class.

## 5. Variational Divergence Minimization (VDM) Objective
To make the variational bound tractable, we parameterize $T$ by a neural network $T_w(x)$ with weights $w$. The **VDM objective** becomes:
$$
  \mathcal{J}(\theta,w)
  = \mathbb{E}_{x\sim p_x}\bigl[T_w(x)\bigr]
    - \mathbb{E}_{x\sim p_\theta}\bigl[f^*(T_w(x))\bigr].
$$
Fitting both generator and critic reduces to the saddle-point problem:
$$
  (\theta^*,w^*)
  = \arg\min_{\theta}\,\arg\max_{w}\;\mathcal{J}(\theta,w).
$$
At optimum, $\theta^*$ minimizes the true divergence and $T_{w^*}(x)\approx f'\bigl(p_x(x)/p_\theta(x)\bigr)$ recovers the density-ratio.

## 6. Implementation Details
- **Generator:** sample $z\sim p_z$ (e.g., Normal(0,I)), then compute $\tilde{x} = g_\theta(z)$. The resulting samples follow $p_\theta$.
- **Critic & Activation:** let $V_w(x)\in\mathbb{R}$ be the raw network output. Use a divergence-dependent activation $\sigma_f:\mathbb{R}\to\mathrm{dom}(f^*)$ so that
  $$
    T_w(x) = \sigma_f\bigl(V_w(x)\bigr) \in \mathrm{dom}(f^*).
  $$
- **Stochastic Optimization:** replace each expectation by a minibatch average and alternate
  $$\begin{aligned}
    w &\gets w + \eta_w\,\nabla_w\,\mathcal{J}(\theta,w), \\
    \theta &\gets \theta - \eta_\theta\,\nabla_\theta\,\mathcal{J}(\theta,w).
  \end{aligned}$$
This adversarial training drives the generator to produce samples indistinguishable from real data under the chosen divergence.

## 7. Special Case: Generative Adversarial Networks (GANs)
For the **Jensen–Shannon divergence**, one sets
\[
  f(u)=u\ln u - (u+1)\ln\bigl(\tfrac{u+1}2\bigr),
  \quad
  f^*(t)=-\log\bigl(1-e^t\bigr),\; t<0,
  \quad
  \sigma_f(v)=-\log\bigl(1+e^{-v}\bigr).
\]
Identifying the sigmoid discriminator
\(D_w(x)=1/(1+e^{-V_w(x)})\) gives the classical GAN minimax game:
\[
  \min_\theta\max_w\;\mathbb E_{x\sim p_x}[\log D_w(x)]
    + \mathbb E_{z\sim p_z}[\log\bigl(1-D_w(g_\theta(z))\bigr)].
\]
This recovers Goodfellow et al.’s original adversarial framework as a special case of VDM.

## References
- Csiszár, I. (1967). Information-type measures of difference of probability distributions and indirect observations. _Studia Sci. Math. Hungarica_.
- Nguyen, X., Wainwright, M. J., & Jordan, M. I. (2010). Estimating divergence functionals and the likelihood ratio by convex risk minimization. _IEEE Trans. Inform. Theory_.
- Nowozin, S., Cseke, B., & Tomioka, R. (2016). f-GAN: Training generative neural samplers using variational divergence minimization. _NeurIPS_.
- Goodfellow, I. et al. (2014). Generative adversarial nets. _NeurIPS_.


# Denoising Diffusion Probabilistic Models (DDPMs), score-based Models

# Denoising Diffusion Probabilistic Models (DDPMs), score-based Models

## Architecture Overview

Denoising Diffusion Probabilistic Models (DDPMs) represent a class of generative models that learn to reverse a gradual noising process. The key idea is to define a forward diffusion process that gradually adds noise to data, and then learn a reverse process that removes noise to generate new samples.

The forward process gradually corrupts the data $x_0$ by adding Gaussian noise over $T$ timesteps:
$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

The reverse process learns to denoise:
$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

## Noise Prior

We typically choose $p_z$ to be either:

- A standard multivariate normal:
  $$
    p_z(z) = \mathcal{N}(0, I),
  $$
  so $z\in\mathbb{R}^d$ has independent $\mathcal{N}(0,1)$ components, or

- A uniform distribution on the hypercube:
  $$
    p_z(z) = \mathrm{Uniform}([-1,1]^d).
  $$

This random vector $z$ provides the stochastic source for generating diverse samples via the generator network.

## Forward Diffusion Process

The forward process gradually adds noise to the data over $T$ timesteps:
$$q(x_{1:T}|x_0) = \prod_{t=1}^T q(x_t|x_{t-1})$$

where each step is a Gaussian transition:
$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

## Discriminator $D_{\omega}$

The discriminator is a binary classifier
\[
  D_{\omega}\colon\;\mathcal{X} \longrightarrow [0,1],
\]
parameterized by $\omega$.  For any input $x$,
\[
  D_{\omega}(x) \approx \Pr\bigl(x\sim p_{\mathrm{data}}\bigr).
\]
Concretely, one computes a logit $v_{\omega}(x)\in\mathbb{R}$ and applies the sigmoid activation:
\[
  D_{\omega}(x)
    = \sigma\bigl(v_{\omega}(x)\bigr)
    = \frac{1}{1 + \exp\bigl(-v_{\omega}(x)\bigr)}.
\]
The discriminator’s objective is to assign high scores to real data and low scores to generator outputs.

## Minimax Game and Loss Function

GAN training can be cast as the following minimax problem (Goodfellow et al., 2014):
\[
  \min_{\theta}\,\max_{\omega}\;J_{\mathrm{GAN}}(\theta,\omega)
  = \mathbb{E}_{x\sim p_{\mathrm{data}}}\bigl[\log D_{\omega}(x)\bigr]
    + \mathbb{E}_{z\sim p_z}\bigl[\log\bigl(1 - D_{\omega}(G_{\theta}(z))\bigr)\bigr].
\]

- **Discriminator** $D_{\omega}$ maximizes $J_{\mathrm{GAN}}$, boosting $D(x)$ on real samples and suppressing $D(\hat{x})$ on generated ones.
- **Generator** $G_{\theta}$ minimizes $J_{\mathrm{GAN}}$, seeking to produce $\hat{x}$ that drive $D_{\omega}(\hat{x})$ towards 1.

## Training Algorithm

We perform alternating updates using minibatches of size $B_1$ (real data) and $B_2$ (noise samples):

### 1. Discriminator Update

1. Sample real data $\{x_i\}_{i=1}^{B_1}\sim p_{\mathrm{data}}$ and noise $\{z_j\}_{j=1}^{B_2}\sim p_z$. 2. Compute generated samples $\hat{x}_j = G_{\theta}(z_j)$. 3. Form the stochastic gradient ascent objective
   \[
     \widehat{J}_D(\omega)
     = \frac{1}{B_1}\sum_{i=1}^{B_1}\log D_{\omega}(x_i)
       + \frac{1}{B_2}\sum_{j=1}^{B_2}\log\bigl(1 - D_{\omega}(\hat{x}_j)\bigr).
   \]
4. Update parameters:
   \[
     \omega \leftarrow \omega + \alpha_1\,\nabla_{\omega}\,\widehat{J}_D(\omega).
   \]

### 2. Generator Update

1. Keeping $\omega$ fixed, sample a fresh minibatch $\{z_j\}_{j=1}^{B_2}\sim p_z$.
2. Compute $\hat{x}_j = G_{\theta}(z_j)$ and evaluate
   \[
     \widehat{J}_G(\theta)
     = \frac{1}{B_2}\sum_{j=1}^{B_2}\log\bigl(1 - D_{\omega}(\hat{x}_j)\bigr).
   \]
3. Update parameters via gradient descent:
   \[
     \theta \leftarrow \theta - \alpha_2\,\nabla_{\theta}\,\widehat{J}_G(\theta).
   \]

**Practical tip:** to avoid vanishing gradients when the discriminator is strong, one often maximizes $\log D_{\omega}(G_{\theta}(z))$ instead of minimizing $\log(1 - D_{\omega}(G_{\theta}(z)))$.

## Divergence‐Minimization Perspective

The discriminator can be viewed as estimating the density ratio between real and generated data.  At optimality,
\[
  D^*(x) = \frac{p_{\mathrm{data}}(x)}{p_{\mathrm{data}}(x) + p_{\theta}(x)},
\]
and substituting into the GAN objective reveals that training minimizes the Jensen–Shannon divergence between $p_{\theta}$ and $p_{\mathrm{data}}$, driving the model distribution towards the data distribution.

## References

- I. Goodfellow, J. Pouget-Abadie, M. Mirza, et al. (2014). Generative Adversarial Networks. *NeurIPS*.  
- M. Arjovsky, S. Chintala, L. Bottou. (2017). Wasserstein GAN. *ICML*.  
- T. Salimans, I. Goodfellow, W. Zaremba, et al. (2016). Improved Techniques for Training GANs. *NeurIPS*.

# Auto-Regressive Models (AR), Large Language Models (LLMs)

## Minimax Formulation of Generative Adversarial Networks
Let \(p_{\text{data}}\) denote the true data distribution on \(\mathbb{R}^d\), and let \(p_z\) be a simple prior (for example, a standard normal \(\mathcal{N}(0,I_k)\)) over latent codes \(z\in\mathbb{R}^k\) with \(k\ll d\).  A generator network

  \(G_\theta\colon\mathbb{R}^k\to\mathbb{R}^d\)

pushes the prior forward to an implicit model distribution \(p_\theta\) by setting

  \(\hat x = G_\theta(z),\quad z\sim p_z.\)

A discriminator (or critic)

  \(D_\omega\colon\mathbb{R}^d\to[0,1]\)

estimates the probability that a sample is real.  The original GAN objective (Goodfellow et al., 2014) is the saddle-point problem

```math
\min_{\theta}\;\max_{\omega}\;J(\theta,\omega),
```
where

```math
J(\theta,\omega)
= \mathbb{E}_{x\sim p_{\text{data}}}\bigl[\log D_\omega(x)\bigr]
+ \mathbb{E}_{z\sim p_z}\bigl[\log\bigl(1 - D_\omega(G_\theta(z))\bigr)\bigr].
```

At the optimum:
- The discriminator \(D_\omega\) seeks to maximize \(J(\theta,\omega)\), distinguishing real from generated.
- The generator \(G_\theta\) seeks to minimize the second term, fooling \(D_\omega\) into assigning high probability to its samples.

Equivalently, one may view generator training as minimizing

```math
J_{G}(\theta)
= \mathbb{E}_{z\sim p_z}\bigl[\log\bigl(1 - D_{\omega^*}(G_\theta(z))\bigr)\bigr],
```
where \(\omega^*(\theta)=\arg\max_\omega J(\theta,\omega)\).  In practice GANs are trained by alternating gradient-based updates on \(\omega\) (ascent) and \(\theta\) (descent).

## Deep Convolutional GAN (DCGAN) Architecture
When modeling images \(x\in\mathbb{R}^{r\times c\times3}\), DCGAN replaces fully-connected layers with convolutional and transpose-convolutional ("deconvolutional") layers.  A typical **generator** pipeline is:

1. Dense layer mapping \(z\in\mathbb{R}^k\) to a low-resolution tensor, e.g.
   \(\mathbb{R}^k \to \mathbb{R}^{4\times4\times512}.\)
2. Sequence of transpose-convolutions that double spatial resolution at each stage:
   \[
     4\times4\times512
     \xrightarrow{\text{ConvTranspose}(256,4,2)} 8\times8\times256
     \xrightarrow{\text{ConvTranspose}(128,4,2)} \dots
     \xrightarrow{\text{ConvTranspose}(3,4,2)} r\times c\times 3.
   \]
3. Batch Normalization (Ioffe & Szegedy, 2015) and ReLU activations after every layer except the output, which uses \(\tanh\) to bound pixels in \([-1,1]\).
4. Weights initialized from \(\mathcal{N}(0,0.02^2)\).

The **discriminator** is a mirror image:
- Strided convolutions (no pooling), halving spatial resolution and doubling depth.
- BatchNorm + LeakyReLU (slope 0.2) in all hidden layers.
- Final sigmoid to produce \(D_\omega(x)\).

## Conditional GANs (cGANs)
When training data come with side information \(y\) (e.g. class labels), cGANs model the conditional distribution \(p(x\mid y)\).  Both generator and discriminator are fed \(y\) (via concatenation or learned embeddings):

```math
J(\theta,\omega)
= \mathbb{E}_{(x,y)\sim p_{x,y}}\bigl[\log D_\omega(x,y)\bigr]
+ \mathbb{E}_{z\sim p_z,\,y\sim p_y}\bigl[\log\bigl(1 - D_\omega(G_\theta(z,y),y)\bigr)\bigr].
```

At convergence the generator approximates the family \(p(x\mid y)\), enabling controlled sampling by fixing \(y\) and drawing \(z\).

## Inference and Sampling
Once \(\theta^*\) is learned, **unconditional sampling** simply draws

```math
z\sim p_z,\quad \hat x = G_{\theta^*}(z).
```

For **conditional sampling**, fix a label \(y^*\) and draw \(z\sim p_z\) to obtain

```math
\hat x\sim p_{\theta}(x\mid y^*)=G_{\theta^*}(z, y^*).
```

## Instability of f-Divergence-Based Training
Standard GANs minimize an \(f\)-divergence (e.g.\u00a0Jensen–Shannon) between \(p_{\text{data}}\) and \(p_\theta\).  Under the **manifold hypothesis**, real images concentrate on a low-dimensional set in \(\mathbb R^d\).  If the support of \(p_\theta\) lies on a separate manifold, the discriminator can perfectly separate real and fake, making gradients \(\nabla_\theta J(\theta,\omega^*)\) vanish and stalling training.

## Wasserstein GAN (WGAN) and Optimal Transport
To overcome saturation, Arjovsky et al. (2017) replace \(f\)-divergences with the **Earth-Mover (Wasserstein-1) distance**:

```math
W\bigl(p_{\text{data}},p_\theta\bigr)
= \inf_{\pi\in\Pi(p_{\text{data}},p_\theta)}
  \mathbb{E}_{(x,\tilde x)\sim\pi}\bigl[\|x-\tilde x\|_2\bigr],
```

where \(\Pi(\cdot,\cdot)\) is the set of joint distributions with given marginals.  By Kantorovich–Rubinstein duality,

```math
W(p_{\text{data}},p_\theta)
= \sup_{\|f\|_{L}\le1}
  \Bigl\{\mathbb{E}_{x\sim p_{\text{data}}}[f(x)]
        - \mathbb{E}_{z\sim p_z}[f(G_\theta(z))]\Bigr\},
```

where the supremum is over all 1-Lipschitz functions \(f\).  In practice, one implements a **critic** \(f_\omega\) and enforces the Lipschitz constraint by:

- **Weight clipping** (Arjovsky et al., 2017)
- **Gradient penalty** (Gulrajani et al., 2017): add
  \(\lambda\,\mathbb{E}_{\hat x}\bigl(\|\nabla_{\hat x}f_\omega(\hat x)\|_2-1\bigr)^2\).
- **Spectral normalization** (Miyato et al., 2018).

The resulting WGAN objective is

```math
\min_{\theta}\;\max_{\omega:\|f_\omega\|_{L}\le1}\;
  \mathbb{E}_{x\sim p_{\text{data}}}[f_\omega(x)]
- \mathbb{E}_{z\sim p_z}[f_\omega(G_\theta(z))].
```

## Transport-Plan Interpretation
For discrete supports \(\{x_i\}_{i=1}^m,\{\tilde x_j\}_{j=1}^n\) with masses \(p_i,\tilde p_j\), a transport plan is a nonnegative matrix \(\pi\in\mathbb{R}_+^{m\times n}\) satisfying

```math
\sum_j\pi_{ij}=p_i,
\quad
\sum_i\pi_{ij}=\tilde p_j.
```
The total cost is

```math
\sum_{i,j}\pi_{ij}\,\|x_i-\tilde x_j\|_2,
```
and optimizing \(\pi\) yields the minimal ``earth-moving'' effort.

## References
- Goodfellow, I.	et al.	(2014). Generative Adversarial Nets. NeurIPS.
- Radford, A., Metz, L., Chintala, S.	(2016). Unsupervised Representation Learning with DCGANs. arXiv:1511.06434.
- Arjovsky, M., Chintala, S., Bottou, L.	(2017). Wasserstein GAN. ICML.
- Gulrajani, I.	et al.	(2017). Improved Training of Wasserstein GANs. NeurIPS.
- Miyato, T.	et al.	(2018). Spectral Normalization for GANs. ICLR.

# State-space Models (SSMs), Su, Mamba

## Bi-Directional Generative Adversarial Networks (Bi-GANs)

Bi-Directional GANs, also known as Adversarially Learned Inference (ALI), extend the classical GAN framework by equipping the model with an explicit encoder that performs approximate inference in the latent space. This bidirectional architecture enables both synthesis of new data and inversion of observed samples back to latent representations, a feature critical for tasks such as representation learning, image editing, and anomaly detection.

### 1. Model Components

A Bi-GAN comprises three neural networks:

• **Generator** \(G_{\theta}: \mathcal{Z} \to \mathcal{X}\) with parameters \(\theta\).  
  – Takes a latent vector \(z\sim p_{z}(z)\) (e.g. a standard Gaussian) and produces a synthetic sample \(\tilde x = G_{\theta}(z)\).  

• **Encoder** \(E_{\phi}: \mathcal{X} \to \mathcal{Z}\) with parameters \(\phi\).  
  – Maps a real data point \(x\sim p_{\mathrm{data}}(x)\) to an inferred latent code \(\tilde z = E_{\phi}(x)\).  

• **Discriminator** \(D_{w}: \mathcal{X}\times\mathcal{Z} \to [0,1]\) with parameters \(w\).  
  – Receives joint pairs \((x,z)\) and outputs the probability that they originate from the real data–encoder distribution rather than the generator–prior distribution.

### 2. Joint Distributions and Adversarial Objective

Define two joint distributions over \((x,z)\):

\[
  p_{X,Z}^{\mathrm{real}}(x,z) 
  = p_{\mathrm{data}}(x)\,\delta\bigl(z - E_{\phi}(x)\bigr),
  \quad
  p_{X,Z}^{\mathrm{fake}}(x,z)
  = p_{z}(z)\,\delta\bigl(x - G_{\theta}(z)\bigr),
\]

where \(\delta(\cdot)\) is the Dirac delta enforcing consistency.  

The discriminator is trained to distinguish “real” joint samples \((x,E_{\phi}(x))\) from “fake” joint samples \((G_{\theta}(z),z)\).  The encoder and generator are trained jointly to fool the discriminator.  Concretely, the minimax game is:

\[
\min_{\theta,\phi}\;\max_{w}\;
\mathcal{L}(\theta,\phi,w)
= 
  \mathbb{E}_{x\sim p_{\mathrm{data}}} \bigl[\log D_{w}(x, E_{\phi}(x))\bigr]
+ \mathbb{E}_{z\sim p_{z}} \bigl[\log\bigl(1 - D_{w}(G_{\theta}(z), z)\bigr)\bigr].
\]

At the Nash equilibrium, the two joint distributions become indistinguishable, i.e.
\(p_{X,Z}^{\mathrm{real}}(x,z)=p_{X,Z}^{\mathrm{fake}}(x,z)\).

### 3. Inversion and Latent Consistency

By learning both \(G_{\theta}\) and \(E_{\phi}\) adversarially, a Bi-GAN approximates the inverse mapping

• **Generation:**
  \(z\sim p_{z}(z) \;\mapsto\; G_{\theta}(z)\) produces novel samples.  

• **Inference (Inversion):**
  \(x\sim p_{\mathrm{data}}(x) \;\mapsto\; E_{\phi}(x)\) yields an approximate latent code.

To further enforce cycle consistency and improve reconstruction fidelity, one often augments the adversarial loss with reconstruction terms:

\[
\mathcal{L}_{\mathrm{recon}}
= \lambda_x\, \mathbb{E}_{x\sim p_{\mathrm{data}}}\bigl[\|x - G_{\theta}(E_{\phi}(x))\|_{p}^{p}\bigr]
+ \lambda_z\, \mathbb{E}_{z\sim p_z}\bigl[\|z - E_{\phi}(G_{\theta}(z))\|_{p}^{p}\bigr],
\]

where \(\lambda_x,\,\lambda_z\ge0\) balance the data-space and latent-space reconstruction penalties.

### 4. Theoretical Properties

Under mild assumptions on model capacity and successful optimization, Bi-GANs provably align the joint data–latent distribution \(p_{\mathrm{data}}(x)\,q_{\phi}(z|x)\) with the model’s joint distribution \(p_{z}(z)\,p_{\theta}(x|z)\).  Key consequences include:

• **Meaningful Encodings:** Real data points are mapped to latent codes that respect the prior and semantic structure.  
• **Prior-Consistent Generation:** Generated samples have latent codes drawn from the prior \(p_{z}(z)\).  

### 5. Applications

Bi-GANs are particularly well suited to applications requiring bidirectional mappings:

• **Representation Learning & Disentanglement:** Learned encoder features \(E_{\phi}(x)\) can serve as inputs to downstream classifiers or for clustering.  

• **Image Editing & Latent Manipulation:** By interpolating or adding attribute vectors in \(\mathcal{Z}\), one can perform controlled modifications (e.g.
otatebox{0}{`smile'-vector}).  

• **Anomaly Detection:** Compare real samples \(x\) with reconstructions \(G_{\theta}(E_{\phi}(x))\). Large reconstruction error may indicate out-of-distribution or anomalous inputs.  

Example: In medical imaging, a Bi-GAN trained on healthy scans can highlight anomalous regions in new scans by measuring pixel-wise reconstruction residuals.

### 6. References

[1] Donahue, J., Krähenbühl, P., & Darrell, T. (2017). Adversarial Feature Learning. In *ICLR*.  
[2] Dumoulin, V., Shlens, J., & Kudlur, M. (2016). Adversarially Learned Inference. *arXiv:1606.00704*.  
[3] Creswell, A., & Bharath, A. A. (2018). Inverting the Generator of a Generative Adversarial Network. *IEEE Transactions on Circuits and Systems for Video Technology*.

# RL-based alignment for LLMs, RLHF, PPO, DPO

## GAN Inversion via Latent Regression
Generative adversarial networks (GANs) learn an implicit mapping

\[  G_{\theta}\colon \mathcal{Z}\to\mathcal{X},\quad z\sim p_z(z)=\mathcal{N}(0,I)\ \Longrightarrow\ x = G_{\theta}(z)\sim p_{\theta}(x),  \]

that approximates the true data distribution \(p_x\).  GAN inversion introduces a feed-forward encoder

\[  E_{\phi}\colon \mathcal{X}\to\mathcal{Z}  \]

such that for any sample (real or generated) \(x\), the recovered code

\(\hat z = E_{\phi}(x)\)

satisfies

\[  G_{\theta}(\hat z) \approx x.  \]

This inversion capability enables downstream tasks such as image editing, representation learning, style transfer and interpolation in latent space.

### Probabilistic Formulation
The generative model induces the marginal distribution

\[  p_{x}(x)
  = \int p_z(z)\,p_{\theta}(x\mid z)\,dz
  \approx \int p_z(z)\,\delta\bigl(x - G_{\theta}(z)\bigr)\,dz.  \]

Our goal is to approximate the (intractable) posterior \(p_{\theta}(z\mid x)\) with the deterministic encoder \(E_{\phi}(x)\).  **Latent regression** enforces that

\[  z,\ E_{\phi}\bigl(G_{\theta}(z)\bigr)  \]

are close in \(\ell_2\)-norm, i.e.

\[  \mathcal{L}_{\rm LR}(\theta,\phi)
  = \mathbb{E}_{z\sim p_z}\Bigl[\|z - E_{\phi}(G_{\theta}(z))\|_2^2\Bigr].  \]

### Model Architecture
We augment the standard GAN with an encoder:
- **Generator** \(G_{\theta}\): maps \(z\sim\mathcal{N}(0,I)\) to a synthetic sample \(\hat x=G_{\theta}(z)\).
- **Discriminator** \(D_{\omega}\): distinguishes real samples \(x\sim p_x\) from generated ones \(\hat x\sim p_{\theta}(x)\).
- **Encoder** \(E_{\phi}\): maps any sample \(x\) (real or generated) back to a latent estimate \(\hat z=E_{\phi}(x)\).

```mermaid
flowchart LR
    z ~ N(0,I) -->|G_θ| x̂ ~ p_θ(x)
    x̂       -->|E_φ| z̃
    z̃       -->|L_reg| ∥z − z̃∥²
    x_real   -->|D_ω| real/fake
```  

### Training Objective
The joint loss combines the adversarial GAN loss with the latent‐regression term:

\[
L(\theta,\omega,\phi)
= \underbrace{\mathbb{E}_{x\sim p_x}\bigl[\log D_{\omega}(x)\bigr]
  + \mathbb{E}_{z\sim p_z}\bigl[\log(1 - D_{\omega}(G_{\theta}(z)))\bigr]}_{\displaystyle \mathcal{L}_{\rm GAN}}
+ \lambda\;\underbrace{\mathbb{E}_{z\sim p_z}\bigl\|z - E_{\phi}(G_{\theta}(z))\bigr\|_2^2}_{\displaystyle \mathcal{L}_{\rm LR}}.
\]

Training proceeds by the saddle‐point optimization

\[  \min_{\theta,\phi}\;\max_{\omega}\;L(\theta,\omega,\phi).  \]

**Key details:**
- The adversarial terms encourage realistic sample generation.
- The regression term enforces \(E_{\phi}\circ G_{\theta}\approx \mathrm{Id}_{\mathcal{Z}}\), i.e.\, cycle consistency in latent space.
- Common architectural choices include spectral normalization in \(D_{\omega}\), residual or progressive growth for \(G_{\theta}\), and convolutional encoders for \(E_{\phi}\).
- The hyperparameter \(\lambda\) balances sample realism against inversion fidelity (typical values: 10–100).

### Inversion of Real Images
After training, a real image \(x_{\rm real}\) can be inverted by

\[
\hat z = E_{\phi}(x_{\rm real}),
\quad
x_{\rm rec} = G_{\theta}(\hat z),
\]

yielding a reconstruction \(x_{\rm rec}\) close to \(x_{\rm real}\) under pixel‐ or perceptual metrics.  The code \(\hat z\) can be manipulated (e.g.\, interpolated, edited) before decoding back through \(G_{\theta}\).

### Discussion and References
Latent regression provides a fast, feed‐forward inversion, avoiding costly per‐image optimization.  It typically achieves orders‐of‐magnitude speed‐ups over gradient‐based inversion methods (e.g. Zhu et al.\, 2016).

Key references:
- Creswell & Bharath (2016), “Inverting the generator of a GAN.”
- Perarnau et al. (2016), “Invertible Conditional GANs for image editing.”
- Donahue et al. (2017), “Adversarial Feature Learning.” ICLR.
- Dumoulin et al. (2017), “Adversarially Learned Inference.” ICLR.

---

## Adversarial Learning for Domain Adaptation
Unsupervised domain adaptation (UDA) seeks to leverage a labeled **source** dataset

\[  \mathcal{D}_S = \{(x_i,y_i)\}_{i=1}^n \sim p_S(x,y)  \]

and an unlabeled **target** dataset

\[  \mathcal{D}_T = \{\tilde x_j\}_{j=1}^m \sim p_T(x),  \]

with \(p_S \neq p_T\), to learn a classifier that generalizes to \(p_T\).  Covariate, prior or conditional shifts typically degrade performance when training only on \(\mathcal{D}_S\).

### Domain‐Adversarial Neural Networks (DANN)
Introduce:
- **Feature extractor** \(\Phi_{\phi}\colon X\to\mathbb{R}^d\).
- **Label predictor** \(C_{\psi}\colon\mathbb{R}^d\to\{1,\dots,C\}\) trained on source labels.
- **Domain discriminator** \(D_{w}\colon\mathbb{R}^d\to[0,1]\) trained to distinguish source vs.
  target features.

#### Losses
1. **Source classification** (cross‐entropy):

   \[
   L_{\rm cls}(\phi,\psi)
   = -\,\mathbb{E}_{(x,y)\sim p_S}\sum_{c=1}^C \mathbf{1}_{[y=c]}\,\log C_{\psi}(\Phi_{\phi}(x))_c.
   \]

2. **Domain adversarial** (binary cross‐entropy):

   \[
   L_{\rm adv}(\phi,w)
   = -\,\mathbb{E}_{x\sim p_S}[\log D_w(\Phi_{\phi}(x))]
     -\,\mathbb{E}_{\tilde x\sim p_T}[\log(1 - D_w(\Phi_{\phi}(\tilde x)))].
   \]

3. **Combined objective** (min–max):

   \[
   \min_{\phi,\psi}\;\max_{w}\;
   L_{\rm cls}(\phi,\psi)
   - \lambda_{\rm adv}\,L_{\rm adv}(\phi,w),
   \]

with \(\lambda_{\rm adv}>0\) balancing classification accuracy against domain invariance.

#### Optimization
- **Domain discriminator**: ascend in \(w\) to maximize \(L_{\rm adv}\).
- **Feature extractor & label predictor**: descend in \((\phi,\psi)\) to minimize \(L_{\rm cls} - \lambda_{\rm adv}L_{\rm adv}\).
- In practice, a **Gradient Reversal Layer (GRL)** implements the adversarial signal by multiplying the back‐propagated domain‐loss gradient by \(-\lambda_{\rm adv}\).

### Inference on Target
Given \(\tilde x_{\rm test}\sim p_T\):

1. Compute feature \(t = \Phi_{\phi^*}(\tilde x_{\rm test})\).
2. Predict label \(\hat y = \arg\max_c\,C_{\psi^*}(t)_c.\)

Since adversarial training aligns \(p_S(\Phi(x))\approx p_T(\Phi(x))\), the classifier generalizes across domains.

### Extensions
- **Conditional DANN**: class‐conditional domain discriminators.
- **Multi‐source/target** and **semi‐supervised** adaptation variants.
- **Reconstruction‐based alignment**: combine GAN inversion losses to translate images between domains with cycle consistency.

---

## Evaluation Metrics: Fréchet Inception Distance (FID)
To quantify how closely a generative model’s output matches real data, one often compares distributions in a high‐level feature space using the **2-Wasserstein distance** under a Gaussian assumption.

### Feature Extraction
Use a fixed, pretrained Inception v3 network \(I_{\varphi}\) (e.g.\, up to the “pool3” layer, \(d=2048\)) to map images to features:

\[
z = I_{\varphi}(x)\in\mathbb{R}^d.
\]

Collect two sets of features:
- Real: \(Z_{\rm real} = \{z_i\}_{i=1}^n\).
- Generated: \(Z_{\rm gen} = \{\hat z_j\}_{j=1}^m\).

### Gaussian Approximation
Estimate empirical moments:

\[
\mu_{\rm real} = \tfrac1n\sum_i z_i,
\quad
\Sigma_{\rm real} = \tfrac1n\sum_i (z_i-\mu_{\rm real})(z_i-\mu_{\rm real})^T,
\]

and similarly \(\mu_{\rm gen},\Sigma_{\rm gen}\).  Model each set by

\(\mathcal{N}(\mu,\Sigma)\.\)

### Fréchet Inception Distance
Under the closed-form for the 2-Wasserstein distance between Gaussians,

\[
\mathrm{FID}
= \|\mu_{\rm real}-\mu_{\rm gen}\|_2^2
+ \mathrm{Tr}\bigl(\Sigma_{\rm real} + \Sigma_{\rm gen}
  - 2(\Sigma_{\rm real}^{1/2}\,\Sigma_{\rm gen}\,\Sigma_{\rm real}^{1/2})^{1/2}\bigr).
\]

**Properties:**
- Symmetric, nonnegative, zero iff moments match exactly.
- Captures both shift in mean (quality) and differences in covariance (diversity).
- Robust to small pixel‐level distortions; reflects semantic alignment.

### Practical Tips
- Use large sample sizes (e.g.\, \(n,m\ge5\times10^4\)) for stable covariance estimates.
- Preprocess real and generated images identically (resize, crop, normalize).
- Compute matrix square roots via eigen‐ or singular‐value decomposition.
- Optionally add a small regularizer (e.g. \(\varepsilon I\)) to covariances for numerical stability.

### References
1. Heusel et al. (2017). “GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.” NeurIPS.
2. Dowson & Landau (1982). “The Fréchet Distance between Multivariate Normal Distributions.” J. Multivariate Anal.
3. Villani (2003). “Optimal Transport: Old and New.” Springer.
4. Salimans et al. (2016). “Improved Techniques for Training GANs.” NeurIPS.