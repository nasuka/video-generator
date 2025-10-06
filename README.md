# Sora Streamlit Demo

Generate and download short videos with OpenAI's Sora 2 models from a Streamlit UI.

## Prerequisites
- [uv](https://docs.astral.sh/uv/) package manager
- Python 3.13+ (managed automatically by `uv`)
- An OpenAI API key with access to Sora 2

## Setup
1. Clone this repository and change into the project directory.
2. Create a `.env` file (optional) and set `OPENAI_API_KEY=your-key` or export it in your shell.
3. Install dependencies with `uv`:
   ```bash
   uv sync
   ```

## Run the Streamlit app
Use `uv run` so the virtual environment, dependencies, and Python version are handled automatically:

```bash
uv run streamlit run streamlit_app.py
```

The app lets you:
- Enter a prompt, choose Sora 2 or Sora 2 Pro, and set video length/size.
- Watch polling updates while the render job runs.
- Preview the finished MP4 directly in the browser and download it.
- Optionally save the MP4 to `downloads/` in the project workspace.
- Upload an optional reference image; the app resizes it to the selected resolution before sending it to Sora.

Download URLs from the API are short-lived (24 hours). Save important assets to your own storage promptly.
