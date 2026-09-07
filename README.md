# AC-Sampler: Accelerate And Correct Diffusion Sampling with Metropolis-Hastings Algorithm (ICLR 2026)

----

This repository contains the official PyTorch implementation of **"AC-Sampler: Accelerate And Correct Diffusion Sampling with Metropolis-Hastings Algorithm"** in [**ICLR 2026**](https://iclr.cc/Conferences/2026).

[**Minsang Park**](https://sites.google.com/view/minsang-park/home)$^1$, [**Gyuwon Sim**](https://aai.kaist.ac.kr/bbs/board.php?bo_table=sub2_1&wr_id=27)$^1$, [**Hyeongho Na**](https://sites.google.com/view/asd-lab)$^2$, [**Jiseok Kwak**](https://aai.kaist.ac.kr/bbs/board.php?bo_table=sub2_1&wr_id=26)$^1$, [**Sumin Lee**](https://aai.kaist.ac.kr/bbs/board.php?bo_table=sub2_1&wr_id=18)$^1$, [**Richard Lee Kim**](https://aai.kaist.ac.kr/bbs/board.php?bo_table=sub2_1&wr_id=31)$^1$, [**Donghyeok Shin**](https://sdh0818.github.io/)$^1$, [**Byeonghu Na**](https://sites.google.com/view/byeonghu-na)$^1$, [**Yeongmin Kim**](https://sites.google.com/view/yeongmin-space/)$^1$, and [**Il-Chul Moon**](https://aai.kaist.ac.kr/bbs/board.php?bo_table=sub2_1&wr_id=3)$^{1,3}$

**KAIST**$^1$, **UNIST**$^2$, **summary.ai**$^3$

----

![main](./imgs/ACSampler_ver7.png)

----

This release covers the **CIFAR-10** experiments. AC-Sampler runs Metropolis-adjusted Langevin chains at intermediate noise levels of a pre-trained diffusion model. A time-dependent discriminator estimates the density ratio between the data and model marginals, and this ratio enters the Metropolis-Hastings acceptance step so that the chain corrects the generated marginal toward the data marginal. Accepted chain states are denoised to `t = 0` as additional trajectories, so many samples share the early (expensive) denoising steps and the average NFE per sample drops below that of the base sampler.

The code is built on [EDM](https://github.com/NVlabs/edm), [DG](https://github.com/aailabkaist/DG), and [DiffRS](https://github.com/aailabkaist/DiffRS).

## Overview of the sampler

Sampling follows the deterministic EDM Heun trajectory (18 steps, 35 NFE on CIFAR-10). At a chosen step index `k` (the *branch time*), each image `x_{t_k}` in the batch seeds a Markov chain at the fixed noise level `t_k`:

1. **Proposal.** A Langevin step `x' = x + (ε²/2)·s(x, t_k) + ε·z`, where `s` is the score from the diffusion model and the step size `ε` is set from a target signal-to-noise ratio (`--snr`).
2. **Accept / reject.** The MH log-acceptance combines the Langevin proposal-kernel ratio with the discriminator log-ratio `log r(x', t_k) - log r(x, t_k)`, where `r ≈ p_data(x_t) / p_model(x_t)`.
3. **Collect.** After `--burn_in` accepted moves, every `(--num_skip + 1)`-th accepted state is stored, until `--num_samples_branch` states per chain are collected.
4. **Denoise.** Collected states are queued and denoised from `t_k` to `0` with the same Heun solver.

Running the script without `--branch_time` reproduces the vanilla EDM sampler, which serves as the baseline and as the source of generated data for discriminator training.

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

Hyper-parameters used for CIFAR-10: `--snr=0.23`, `--burn_in=10`, one branch at step `8` collecting `500` states per chain. With a batch of 100 root images this yields `100 × 500 = 50,000` samples.

```bash
python generate_ac_sampler.py \
    --network=checkpoints/pretrained_score/edm-cifar10-32x32-uncond-vp.pkl \
    --discriminator_ckpt=checkpoints/discriminator/cifar_uncond/discriminator_60.pt \
    --outdir=samples/cifar10/ac_sampler_branch8 \
    --num_samples=50000 --branch_time=8 --num_samples_branch=500 --snr=0.23 --burn_in=10
```

Multiple branch times can be given as comma-separated lists, e.g. `--branch_time=5,10 --num_samples_branch=2,1`.

| Option | Meaning |
| --- | --- |
| `--branch_time` | Step indices (`0..steps-1`) at which MH chains are run. Smaller index = higher noise level. |
| `--num_samples_branch` | Number of chain states collected per root image at each branch time. |
| `--snr` | Target SNR for the Langevin step size. |
| `--burn_in` | Accepted proposals discarded at the start of each chain. |
| `--num_skip` | Thinning: keep every `(num_skip+1)`-th accepted state. |
| `--do_mh` | `0` disables the accept/reject step (plain Langevin). |
| `--dg_proposal` | Adds discriminator guidance to the score used inside the Langevin proposal. |
| `--dg_weight` | Discriminator guidance on the ODE drift (DG baseline). |
| `--steps`, `--S_churn`, ... | Standard EDM sampler options. |

## Step 4. Evaluation

FID (EDM protocol, 50k samples):

```bash
python fid_npzs.py --images=samples/cifar10/ac_sampler_branch8 --ref=data/cifar10-32x32.npz
```

Precision / Recall / IS with the ADM evaluation suite:

```bash
python evaluations/merge_npz.py --samples_dir=samples/cifar10/ac_sampler_branch8 --num_samples=10000
python evaluations/evaluator.py <reference_batch.npz> samples/cifar10/ac_sampler_branch8/samples_merged.npz
```

Average NFE per sample:

```bash
python summarize_nfe.py samples/cifar10/edm_base samples/cifar10/ac_sampler_branch8
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
