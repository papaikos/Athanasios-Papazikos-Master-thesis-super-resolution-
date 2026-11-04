# Baselines

Two baseline studies on real Sentinel-2 port scenes. Neither has a high-resolution reference, so no metric is claimed.

**Released pretrained models (15 scenes).** SEN2SR (WEO-SAS `mamba-main` / `referencex4`, ten bands, Hugging Face revision `5262f02bd454915de7a77cb8627f581255186719`) and DiffusionSat (fMoW Sentinel → RGB ControlNet, checkpoint 50,000, source commit `68259b074dfa567e6cc4667799dab0e37477fcf2`). Both models were used unmodified.
- Scripts: `dataset/scripts/acquire_pretrained_sentinel15.py`, `download_diffusionsat.py`, `infer_pretrained_weo.py`, `infer_pretrained_diffusionsat.py`, `render_pretrained_comparison.py`.
- Environments: `requirements-sen2sr.txt`, `requirements-diffusionsat.txt`. Method notes: `SEN2SR-method.md`, `DiffusionSat-method.md`.
- The weights are not redistributed here. Get them from their authors.
- DiffusionSat visibly invents or moves boats and harbour structures. Treat its outputs as generative, not as recovered detail.

**Fine-tuning pilot (30 scenes).** SEN2SR and the official pretrained SwinIR-M ×4 model, each fine-tuned for 2,000 updates on RGB. Scripts: `dataset/scripts/acquire_sentinel30_examples.py`, `infer_sentinel30_dual.py`, `render_rgb_comparisons.py`. Environment: `requirements-finetune-sen2sr-swinir.txt`. Both models sharpened strongly and clipped bright regions. That result motivated training from scratch on the maritime dataset.
