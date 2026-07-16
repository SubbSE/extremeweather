# ExtremeCast — Prithvi WxC 2.3B on Colab

Run the real NASA/IBM **Prithvi WxC 2.3B** weather foundation model on a free/Pro Colab GPU,
as part of the *ExtremeCast* subseasonal heatwave/coldwave forecasting project.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SubbSE/extremeweather/blob/main/Prithvi_WxC_Colab.ipynb)

## What the notebook does
1. Checks the Colab GPU (A100/L4/T4; auto-enables gradient checkpointing on <20 GB cards).
2. Installs `PrithviWxC` from the official repo.
3. Downloads the 2.3B weights + climatology + sample MERRA-2 + scalers from Hugging Face.
4. Loads the model and runs a **real forward pass → global forecast**, plotted as a 2 m-temperature map.
5. Provides an honest **fine-tuning scaffold** (frozen backbone + light head) for the heatwave task.

## Requirements
- A Colab GPU runtime (`Runtime → Change runtime type → GPU`). **CPU will not work** — the model is CUDA-only.
- ~9 GB download on first run (cached for the session).

## Honest scope
- ✅ Running Prithvi + a zero-shot forecast: works on Colab.
- ⚠️ Training it *very well* over many epochs needs the full MERRA-2 record (100s of GB, persistent storage)
  and long A100 time — Colab sessions time out and have ephemeral disk. Use Colab to run/prototype;
  use HPC or a rented cloud GPU for the full-scale fine-tune.

## Publish this so the badge works
```bash
# from this colab/ folder:
git add -A && git commit -m "Prithvi WxC Colab notebook"          # (already committed for you)
gh repo create YOUR_REPO --public --source=. --remote=origin --push   # if you have gh
# --- or without gh: create an empty repo on github.com, then: ---
git remote add origin https://github.com/YOUR_GH_USER/YOUR_REPO.git
git branch -M main
git push -u origin main
```
Then edit the badge link above, replacing `YOUR_GH_USER/YOUR_REPO`. Clicking it opens the notebook straight in Colab.

## References
- Code: https://github.com/NASA-IMPACT/Prithvi-WxC
- Weights: https://huggingface.co/ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M
- Paper: https://arxiv.org/abs/2409.13598
