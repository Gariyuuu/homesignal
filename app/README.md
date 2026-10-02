---
title: HomeSignal
emoji: 🏠
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: "6.29.1"
app_file: app.py
pinned: false
license: mit
---

# HomeSignal demo

Educational demo of a 12-month-forward ZIP-level home value growth model. **Not financial advice.**

## Run locally

```bash
make app            # from the repo root, after `make all` (or at least `make train`)
```

## Deploy to Hugging Face Spaces

1. Run `python app/prepare_app_data.py` to build `app/data/` (model + derived features + ZHVI history).
2. Create a Gradio Space and upload the contents of `app/` plus the `homesignal` package:

   ```bash
   pip install huggingface_hub
   python - <<'EOF'
   from huggingface_hub import HfApi
   api = HfApi()
   api.create_repo("<user>/homesignal", repo_type="space", space_sdk="gradio", exist_ok=True)
   api.upload_folder(folder_path="app", repo_id="<user>/homesignal", repo_type="space")
   api.upload_folder(folder_path="src/homesignal", path_in_repo="homesignal", repo_id="<user>/homesignal", repo_type="space")
   EOF
   ```

   `requirements.txt` in this folder lists the runtime dependencies; the Space runs on the free CPU tier.
3. Keep the data attribution shown in the app (Zillow Research, U.S. Census Bureau, Freddie Mac, BLS).

The bundle contains derived model features and the monthly ZHVI index per ZIP (needed for the history chart);
no other raw source files are redistributed. See `docs/sources.md` for terms.
