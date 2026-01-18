# Stable Diffusion 3.5 Comparison Dashboard

This dashboard allows you to compare the generation speed and quality of Stability AI's Stable Diffusion 3.5 models (Large, Large-Turbo, and Medium) on your local machine.

## Prerequisites

1.  **Hugging Face Account**: You need an account on [Hugging Face](https://huggingface.co/).
2.  **Access Token**: Create a User Access Token (Read permissions are sufficient). You can find this in your [Hugging Face Settings](https://huggingface.co/settings/tokens).
3.  **Model Access**: You must accept the license terms for the models to be able to download them. Visit the following pages and accept the terms:
    - [stable-diffusion-3.5-large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)
    - [stable-diffusion-3.5-large-turbo](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo)
    - [stable-diffusion-3.5-medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)

## Configuration

To run this dashboard, you need to provide your Hugging Face Access Token.

1.  Create a file named `.env` in the project root directory (same level as `src`).
2.  Add your token to the file:

```env
HF_TOKEN=hf_...
```

## Running the Dashboard

1.  Ensure your virtual environment is activated:
    ```powershell
    .\.venv\Scripts\activate
    ```
2.  Navigate to the project root directory (parent of `src`).
3.  Run the dashboard using Streamlit:
    ```powershell
    streamlit run src/comparison/dashboard.py
    ```


## Output

- Generated images will be saved to the `out/comparison` directory in the project root.
- Model weights are downloaded to the local `.cache` directory in the project root.

```powershell
.\.venv\Scripts\python -m streamlit run src/comparison/dashboard.py
.\.venv\Scripts\python -m streamlit run src/analysis/prompt_dashboard.py
```
