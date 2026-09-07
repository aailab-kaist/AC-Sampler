# AC-Sampler: Accelerate And Correct Diffusion Sampling with Metropolis-Hastings Algorithm (ICLR 2026)

----

This repository contains the official PyTorch implementation of **"AC-Sampler: Accelerate And Correct Diffusion Sampling with Metropolis-Hastings Algorithm"** in [**ICLR 2026**](https://iclr.cc/Conferences/2026). | [**OpenReview**](https://openreview.net/forum?id=kWl13kRJTQ) |

[**Minsang Park**](https://sites.google.com/view/minsang-park/home)$^1$, [**Gyuwon Sim**](https://aai.kaist.ac.kr/bbs/board.php?bo_table=sub2_1&wr_id=27)$^1$, [**Hyeongho Na**](https://sites.google.com/view/asd-lab)$^2$, [**Jiseok Kwak**](https://aai.kaist.ac.kr/bbs/board.php?bo_table=sub2_1&wr_id=26)$^1$, [**Sumin Lee**](https://aai.kaist.ac.kr/bbs/board.php?bo_table=sub2_1&wr_id=18)$^1$, [**Richard Lee Kim**](https://aai.kaist.ac.kr/bbs/board.php?bo_table=sub2_1&wr_id=31)$^1$, [**Donghyeok Shin**](https://sdh0818.github.io/)$^1$, [**Byeonghu Na**](https://sites.google.com/view/byeonghu-na)$^1$, [**Yeongmin Kim**](https://sites.google.com/view/yeongmin-space/)$^1$, and [**Il-Chul Moon**](https://aai.kaist.ac.kr/bbs/board.php?bo_table=sub2_1&wr_id=3)$^{1,3}$

**KAIST**$^1$, **UNIST**$^2$, **summary.ai**$^3$

----

![main](./imgs/ACSampler_ver7.png)

----

This release covers the **CIFAR-10** experiments. AC-Sampler runs Metropolis-adjusted Langevin chains at intermediate noise levels of a pre-trained diffusion model. A time-dependent discriminator estimates the density ratio between the data and model marginals, and this ratio enters the Metropolis-Hastings acceptance step so that the chain corrects the generated marginal toward the data marginal. Accepted chain states are denoised to `t = 0` as additional trajectories, so many samples share the early (expensive) denoising steps and the average NFE per sample drops below that of the base sampler.

The code is built on [EDM](https://github.com/NVlabs/edm), [DG](https://github.com/aailabkaist/DG), and [DiffRS](https://github.com/aailabkaist/DiffRS).

## Overview of the sampler

Notation follows Sec. 4 of the paper. $\mathbf{s}^{\boldsymbol\theta}(\mathbf{x}_t, t) \approx \nabla_{\mathbf{x}_t} \log q_t(\mathbf{x}_t)$ is the pre-trained score network, $q_t$ and $p^{\boldsymbol\theta}_t$ are the data and model marginals at timestep $t$, and $d^{\boldsymbol\phi}(\mathbf{x}_t, t)$ is a time-dependent discriminator trained to separate $q_t$ from $p^{\boldsymbol\theta}_t$ (Eq. 8, same training scheme as DG). Its output gives the likelihood ratio

$$L^{\boldsymbol\phi}_t(\mathbf{x}_t, t) := \frac{d^{\boldsymbol\phi}(\mathbf{x}_t, t)}{1 - d^{\boldsymbol\phi}(\mathbf{x}_t, t)} \approx \frac{q_t(\mathbf{x}_t)}{p^{\boldsymbol\theta}_t(\mathbf{x}_t)} .$$

**Overall procedure (Sec. 4).** (i) Denoise from the prior down to a target timestep $\tau$ with the base sampler; each $\mathbf{x}_\tau$ is the initial state of an MCMC chain. (ii) Repeatedly draw candidates from a score-based proposal and apply MH correction (Algorithm 1). After a burn-in period the chain samples follow the true marginal $q_\tau$. (iii) Denoise every accepted sample from $\tau$ to $0$ to obtain the outputs. Step (i) is shared by all samples of a chain (*Acceleration Gain*), and step (ii) moves the samples toward $q_\tau$ (*Correction Gain*).

**Proposal distribution (Sec. 4.1, Eq. 5).** MALA with the pre-trained score:

$$p^{\boldsymbol\theta}_{\text{proposal},t}(\cdot \mid \mathbf{x}_t) = \mathcal{N}\!\left(\mathbf{x}_t + \tfrac{\eta}{2}\,\mathbf{s}^{\boldsymbol\theta}(\mathbf{x}_t, t),\ \eta\mathbf{I}\right),
\qquad \sqrt{\eta} = \text{SNR} \times \frac{2\,\lVert\boldsymbol\epsilon\rVert}{\lVert\mathbf{s}\rVert} \quad \text{(Eq. 68)},$$

where $\boldsymbol\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ is the injected noise. The same score evaluation serves both the denoising step and the proposal.

**Algorithm 1 `MALAOneStep`**

> **Input:** target timestep $\tau$, previous sample $\mathbf{x}_\tau$, score output $\mathbf{s} := \mathbf{s}^{\boldsymbol\theta}(\mathbf{x}_\tau, \tau)$, likelihood ratio $L^{\boldsymbol\phi}_\tau := \frac{d^{\boldsymbol\phi}(\mathbf{x}_\tau, \tau)}{1 - d^{\boldsymbol\phi}(\mathbf{x}_\tau, \tau)}$, score network $\mathbf{s}^{\boldsymbol\theta}$, discriminator $d^{\boldsymbol\phi}$
> **Output:** next sample $\tilde{\mathbf{x}}_\tau$
> 1. **repeat**
> 2. &nbsp;&nbsp;&nbsp;&nbsp;Propose $\tilde{\mathbf{x}}_\tau$ from proposal distribution $p^{\boldsymbol\theta}_{\text{proposal},\tau}(\cdot \mid \mathbf{x}_\tau)$ (Eq. 5)
> 3. &nbsp;&nbsp;&nbsp;&nbsp;Get score $\tilde{\mathbf{s}} \leftarrow \mathbf{s}^{\boldsymbol\theta}(\tilde{\mathbf{x}}_\tau, \tau)$, and likelihood ratio $\tilde{L}^{\boldsymbol\phi}_\tau \leftarrow \frac{d^{\boldsymbol\phi}(\tilde{\mathbf{x}}_\tau, \tau)}{1 - d^{\boldsymbol\phi}(\tilde{\mathbf{x}}_\tau, \tau)}$
> 4. &nbsp;&nbsp;&nbsp;&nbsp;Calculate acceptance probability $\alpha \leftarrow \hat\alpha(\mathbf{x}_\tau, \tilde{\mathbf{x}}_\tau, \mathbf{s}, \tilde{\mathbf{s}}, L^{\boldsymbol\phi}_\tau, \tilde{L}^{\boldsymbol\phi}_\tau)$ (Eq. 9)
> 5. &nbsp;&nbsp;&nbsp;&nbsp;Sample $u \sim \mathcal{U}(0, 1)$
> 6. **until** $u < \alpha$
> 7. **return** $\tilde{\mathbf{x}}_\tau, \tilde{\mathbf{s}}, \tilde{L}^{\boldsymbol\phi}_\tau$

**Acceptance probability (Sec. 4.2, Eq. 9).** Using Theorem 4.1 with $\hat{\mathbf{x}}_{t-1} := \tfrac12\big(\mu_t(\mathbf{x}_t, \mathbf{s}) + \mu_t(\tilde{\mathbf{x}}_t, \tilde{\mathbf{s}})\big)$, the intractable ratio $q_t(\tilde{\mathbf{x}}_t)/q_t(\mathbf{x}_t)$ becomes tractable:

$$\hat\alpha(\mathbf{x}_t, \tilde{\mathbf{x}}_t, \mathbf{s}, \tilde{\mathbf{s}}, L, \tilde{L}) = \min\!\left(1,\ \underbrace{\frac{q_{t|t-1}(\tilde{\mathbf{x}}_t \mid \hat{\mathbf{x}}_{t-1})}{q_{t|t-1}(\mathbf{x}_t \mid \hat{\mathbf{x}}_{t-1})}}_{\text{Forward term}} \cdot \underbrace{\frac{\tilde{L}}{L}}_{\text{Likelihood ratio}} \cdot \underbrace{\frac{p^{\boldsymbol\theta}_{\text{proposal},t}(\mathbf{x}_t \mid \tilde{\mathbf{x}}_t)}{p^{\boldsymbol\theta}_{\text{proposal},t}(\tilde{\mathbf{x}}_t \mid \mathbf{x}_t)}}_{\text{Proposal term}}\right).$$

The forward and proposal terms are Gaussians and the likelihood ratio comes from the discriminator, so $\hat\alpha$ is fully tractable.

**Propose-until-accept (Sec. 4.3).** Instead of keeping a rejected proposal as a repeated state, Algorithm 1 redraws proposals until one is accepted and records only accepted samples. This avoids duplicated states in the finite-sample empirical distribution.

**Correspondence with the code** (`ac_sampler` in `generate_ac_sampler.py`):

| Paper | Code | Notes |
| --- | --- | --- |
| Base sampler, $T$ steps | EDM Heun, `--steps` | Deterministic on CIFAR-10 (`--S_churn=0`). |
| Target timestep $\tau$ | `--branch_time` $= T - \tau$ | `--branch_time` is the number of denoising steps taken from the prior before the chain starts, i.e. $\tau$ steps of the schedule remain. Several values can be given as a comma-separated list. |
| SNR (Eq. 68) | `--snr` | Sets $\sqrt\eta$ per sample. |
| $n_{\text{burn-in}}$ | `--burn_in` | Accepted states discarded at the start of each chain. |
| $n_{\text{skip}}$ | `--num_skip` | Keep every $(n_{\text{skip}}+1)$-th accepted state. |
| $n_{\text{chain}}$ | `--num_samples_branch` | Chain length = number of samples obtained from one initial point (one value per `--branch_time`). |
| Eq. 5 proposal | `mh_branch` | $\tilde{\mathbf{x}} = \mathbf{x} + \tfrac{\eta}{2}\mathbf{s} + \sqrt\eta\,\boldsymbol\epsilon$; the score at $\tilde{\mathbf{x}}$ is re-used for the next denoising step. |
| Eq. 9 acceptance | `mh_branch` | $\log\hat\alpha = \log\tilde{L} - \log L + \tfrac{\eta}{8}\left(\lVert\mathbf{s}\rVert^2 - \lVert\tilde{\mathbf{s}}\rVert^2\right)$, where the last term is the closed form of the forward and proposal Gaussian terms. `--do_mh=0` removes the accept/reject test (plain Langevin). |
| DG$_p$ (Appendix, Table 16) | `--dg_proposal` | Discriminator-guided proposal $\mathbf{s} + w\,\nabla\log L^{\boldsymbol\phi}$; needs a gradient and is slower. |

Running without `--branch_time` reproduces the vanilla EDM sampler, which serves as the baseline and as the source of generated data for discriminator training.

## Requirements

Tested with Python 3.8, PyTorch 1.12.1, CUDA 11.3.

```bash
conda env create -f environment.yml
conda activate ac_sampler
# or
pip install -r requirements.txt
```

`tensorflow` is required only for `evaluations/evaluator.py` (Precision / Recall / Inception Score).

## Checkpoints and data

Place the following files under `checkpoints/` and `data/` (they are not tracked by git).

| File | Source |
| --- | --- |
| `checkpoints/pretrained_score/edm-cifar10-32x32-uncond-vp.pkl` | [EDM](https://github.com/NVlabs/edm) pre-trained score network |
| `checkpoints/ADM_classifier/32x32_classifier.pt` | [DG](https://github.com/aailabkaist/DG) (noise-conditioned ADM classifier used as a feature extractor) |
| `checkpoints/discriminator/cifar_uncond/discriminator_60.pt` | [DG](https://github.com/aailabkaist/DG), or train your own (Step 2) |
| `data/cifar10-32x32.npz` | [EDM FID reference statistics](https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz) |
| `data/true_data.npz` | CIFAR-10 training images as uint8 `(50000, 32, 32, 3)` (from DG). Only needed for discriminator training |

## Step 1. Baseline EDM samples

```bash
python generate_ac_sampler.py \
    --network=checkpoints/pretrained_score/edm-cifar10-32x32-uncond-vp.pkl \
    --discriminator_ckpt=checkpoints/discriminator/cifar_uncond/discriminator_60.pt \
    --outdir=samples/cifar10/edm_base --num_samples=50000
```

Samples are written as `samples_<i>.npz` (key `samples`, uint8 NHWC, 100 images per file) together with a preview grid `sample_<i>.png`. NFE statistics are written to `nfe_analysis.json`.

## Step 2. (Optional) Train the discriminator

The discriminator separates noisy real images from noisy generated images at random diffusion times. Use the baseline samples from Step 1 as the generated set.

```bash
python train.py \
    --savedir=checkpoints/discriminator/cifar_uncond \
    --gendir=samples/cifar10/edm_base \
    --datadir=data/true_data.npz \
    --epoch=60
```

## Step 3. AC-Sampler

Configurations from Table 21 of the paper (unconditional CIFAR-10, EDM + Heun). Remember `--branch_time = T - τ`.

| $T$ | SNR | $n_{\text{chain}}$ | $n_{\text{burn-in}}$ | $n_{\text{skip}}$ | $\tau$ | FID | NFE | Command flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 18 | 0.23 | 300 | 10 | 0 | 13 | 1.97 | 26.19 | `--steps=18 --branch_time=5 --num_samples_branch=300 --snr=0.23 --burn_in=10` |
| 18 | 0.23 | 50 | 10 | 0 | 11 | 2.10 | 22.78 | `--steps=18 --branch_time=7 --num_samples_branch=50 --snr=0.23 --burn_in=10` |
| 14 | 0.2 | 50 | 0 | 0 | 6 | 2.38 | 15.82 | `--steps=14 --branch_time=8 --num_samples_branch=50 --snr=0.2 --burn_in=0` |
| 10 | 0.2 | 11 | 0 | 0 | 5 | 3.24 | 10.57 | `--steps=10 --branch_time=5 --num_samples_branch=11 --snr=0.2 --burn_in=0` |

Example (first row):

```bash
python generate_ac_sampler.py \
    --network=checkpoints/pretrained_score/edm-cifar10-32x32-uncond-vp.pkl \
    --discriminator_ckpt=checkpoints/discriminator/cifar_uncond/discriminator_60.pt \
    --outdir=samples/cifar10/ac_T18_tau13 --num_samples=50000 \
    --steps=18 --branch_time=5 --num_samples_branch=300 --snr=0.23 --burn_in=10
```

Multiple target timesteps can be used at once, e.g. `--branch_time=3,4,5 --num_samples_branch=3,4,5` (Table 21, $\tau = 15, 14, 13$). The chain stops branching automatically once enough samples are queued to reach `--num_samples`.

Other options:

| Option | Meaning |
| --- | --- |
| `--do_mh` | `0` disables the accept/reject step (plain Langevin). |
| `--dg_proposal` | Weight of discriminator guidance inside the Langevin proposal (DG$_p$). |
| `--dg_weight` | Discriminator guidance on the ODE drift (DG baseline, Kim et al., 2023). |
| `--S_churn`, `--S_min`, `--S_max`, `--S_noise` | EDM stochastic sampler options (unused for CIFAR-10). |

## Step 4. Evaluation

FID (EDM protocol, 50k samples):

```bash
python fid_npzs.py --images=samples/cifar10/ac_T18_tau13 --ref=data/cifar10-32x32.npz
```

Precision / Recall / IS with the ADM evaluation suite:

```bash
python evaluations/merge_npz.py --samples_dir=samples/cifar10/ac_T18_tau13 --num_samples=10000
python evaluations/evaluator.py <reference_batch.npz> samples/cifar10/ac_T18_tau13/samples_merged.npz
```

Average NFE per sample:

```bash
python summarize_nfe.py samples/cifar10/edm_base samples/cifar10/ac_T18_tau13
```

## Repository layout

```
generate_ac_sampler.py   AC-Sampler (also the EDM baseline when no branch is given)
train.py                 Discriminator training
classifier_lib.py        Discriminator loading, log-ratio and VP-SDE utilities
fid_npzs.py              FID computation
summarize_nfe.py         NFE summary from nfe_analysis.json
evaluations/             ADM evaluation suite (Precision / Recall / IS) and npz merge helper
guided_diffusion/        ADM U-Net / classifier architecture (OpenAI)
dnnlib/, torch_utils/    EDM utilities (NVIDIA), required to unpickle EDM networks
```

## Acknowledgements

This code is heavily based on:

* Karras et al., *Elucidating the Design Space of Diffusion-Based Generative Models*, NeurIPS 2022. [code](https://github.com/NVlabs/edm)
* Kim et al., *Refining Generative Process with Discriminator Guidance in Score-based Diffusion Models*, ICML 2023. [code](https://github.com/aailabkaist/DG)
* Na et al., *Diffusion Rejection Sampling*, ICML 2024. [code](https://github.com/aailabkaist/DiffRS)
* Dhariwal & Nichol, *Diffusion Models Beat GANs on Image Synthesis*, NeurIPS 2021. [code](https://github.com/openai/guided-diffusion)

## Citation

```bibtex
@inproceedings{park2026acsampler,
  title     = {{AC-Sampler}: Accelerate And Correct Diffusion Sampling with Metropolis-Hastings Algorithm},
  author    = {Park, Minsang and Sim, Gyuwon and Na, Hyeongho and Kwak, Jiseok and Lee, Sumin and Kim, Richard Lee and Shin, Donghyeok and Na, Byeonghu and Kim, Yeongmin and Moon, Il-Chul},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
}
```

## License

Code derived from EDM (`generate_ac_sampler.py`, `fid_npzs.py`, `dnnlib/`, `torch_utils/`) is under NVIDIA's [CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/) license. The remaining code follows the Apache 2.0 license in `LICENSE`, inherited from DG.
