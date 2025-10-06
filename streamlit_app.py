"""Streamlit app for generating Sora 2 videos and downloading the result."""
from __future__ import annotations

import os
import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from PIL import Image

# Load environment variables from a local .env file if present.
load_dotenv()

st.set_page_config(page_title="Sora 2 Video Generator", page_icon="🎬", layout="centered")

MODEL_OPTIONS = {
    "Sora 2 (faster, great for iteration)": "sora-2",
    "Sora 2 Pro (higher fidelity, slower)": "sora-2-pro",
}

SIZE_OPTIONS = {
    "HD 720p (1280x720)": "1280x720",
    "Square 1024 (1024x1024)": "1024x1024",
}

SECONDS_OPTIONS = [4, 8, 12]

DEFAULT_PROMPT = (
    "Wide shot of a child flying a red kite in a grassy park at golden hour, "
    "soft camera pan, cinematic lighting, natural ambient sound."
)

POLL_INTERVAL_SECONDS = 5
DOWNLOAD_DIR = Path("downloads")


def init_state() -> None:
    st.session_state.setdefault("video_bytes", None)
    st.session_state.setdefault("video_filename", None)
    st.session_state.setdefault("job_id", None)
    st.session_state.setdefault("download_path", None)
    st.session_state.setdefault("api_key", os.getenv("OPENAI_API_KEY", ""))
    st.session_state.setdefault("save_to_disk", False)
    st.session_state.setdefault("last_error", None)
    st.session_state.setdefault("resized_image_preview", None)
    st.session_state.setdefault("resized_image_size", None)


def clamp_progress(value: Optional[float]) -> int:
    if value is None:
        return 0
    return max(0, min(100, int(value)))


def binary_response_to_bytes(response) -> bytes:
    """Normalize the OpenAI binary response object to raw bytes."""

    read_method = getattr(response, "read", None)
    if callable(read_method):
        data = read_method()
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)

    content = getattr(response, "content", None)
    if isinstance(content, (bytes, bytearray)):
        return bytes(content)

    fd, temp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    temp_file = Path(temp_path)
    try:
        write_to_file = getattr(response, "write_to_file", None)
        if callable(write_to_file):
            write_to_file(temp_path)
            return temp_file.read_bytes()
        raise RuntimeError("Binary response did not expose a compatible interface.")
    finally:
        temp_file.unlink(missing_ok=True)


def parse_size(size: str) -> tuple[int, int]:
    try:
        width_str, height_str = size.lower().split("x", 1)
        return int(width_str), int(height_str)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"Invalid size '{size}', expected format WIDTHxHEIGHT.") from exc


def prepare_input_reference(uploaded_file, size: str) -> tuple[BytesIO, Image.Image]:
    """Resize the uploaded image to match the target size and return a file-like object."""

    width, height = parse_size(size)

    try:
        image = Image.open(uploaded_file)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Unable to open the uploaded image. Please upload a valid file.") from exc

    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    else:
        image = image.convert("RGB")

    resized = image.resize((width, height), Image.Resampling.LANCZOS)

    buffer = BytesIO()
    buffer.name = f"input_reference_{width}x{height}.png"
    resized.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer, resized


def main() -> None:
    init_state()

    st.title("🎬 Sora 2 Video Generator")
    st.write(
        "Generate short clips with OpenAI's Sora 2 video model and download the MP4 "
        "directly from this Streamlit app."
    )

    with st.expander("Usage Notes", expanded=False):
        st.markdown(
            "- Provide your OpenAI API key; the app never stores it.\n"
            "- Prompts must comply with OpenAI's Sora content policies.\n"
            "- Render jobs may take several minutes. This demo polls every "
            f"{POLL_INTERVAL_SECONDS} seconds."
        )

    api_key_input = st.text_input(
        "OpenAI API Key",
        value=st.session_state.api_key,
        type="password",
        help="You can also set OPENAI_API_KEY in a .env file.",
    )
    st.session_state.api_key = api_key_input

    prompt = st.text_area(
        "Prompt",
        value=DEFAULT_PROMPT,
        height=120,
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()), index=0)
        model = MODEL_OPTIONS[model_label]
    with col2:
        seconds_choice = st.selectbox(
            "Duration (seconds)",
            SECONDS_OPTIONS,
            index=SECONDS_OPTIONS.index(8),
            help="Sora 2 currently supports 4, 8, or 12 second clips.",
        )
        seconds = str(seconds_choice)
    with col3:
        size_label = st.selectbox("Frame Size", list(SIZE_OPTIONS.keys()), index=0)
        size = SIZE_OPTIONS[size_label]

    uploaded_image = st.file_uploader(
        "Optional reference image",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
        help="If provided, the image is resized to the selected frame size before upload.",
    )

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Original reference image", use_column_width=True)
        st.caption(f"The image will be resized to {size} before sending to Sora.")

    st.session_state.save_to_disk = st.checkbox(
        "Also save MP4 to local downloads/ directory", value=st.session_state.save_to_disk
    )

    if st.button("Generate video", type="primary"):
        if not st.session_state.api_key:
            st.error("Please provide your OpenAI API key.")
            return

        if not prompt.strip():
            st.error("Prompt cannot be empty.")
            return

        st.session_state.video_bytes = None
        st.session_state.video_filename = None
        st.session_state.download_path = None
        st.session_state.resized_image_preview = None
        st.session_state.resized_image_size = None
        st.session_state.last_error = None

        client = OpenAI(api_key=st.session_state.api_key)

        input_reference_file = None
        resized_preview = None

        if uploaded_image is not None:
            try:
                input_reference_file, resized_preview = prepare_input_reference(uploaded_image, size)
            except ValueError as exc:
                st.session_state.last_error = str(exc)
                st.error(str(exc))
                return

        try:
            with st.spinner("Starting render job..."):
                create_kwargs = {
                    "model": model,
                    "prompt": prompt,
                    "seconds": seconds,
                    "size": size,
                }
                if input_reference_file is not None:
                    create_kwargs["input_reference"] = input_reference_file

                job = client.videos.create(**create_kwargs)
        except OpenAIError as exc:
            st.session_state.last_error = str(exc)
            st.error(f"Failed to start video generation: {exc}")
            return

        status_placeholder = st.empty()
        progress_bar = st.progress(0)

        st.write(f"Job ID: `{job.id}`")

        try:
            while job.status in ("queued", "in_progress"):
                progress = clamp_progress(getattr(job, "progress", None))
                progress_bar.progress(progress)

                status_placeholder.info(f"Status: {job.status} • progress {progress}%")

                time.sleep(POLL_INTERVAL_SECONDS)
                job = client.videos.retrieve(job.id)

            status_placeholder.info(f"Status: {job.status}")
            progress_bar.progress(100)

            if job.status == "failed":
                error_message = getattr(getattr(job, "error", None), "message", "Unknown error")
                st.session_state.last_error = error_message
                st.error(f"Video generation failed: {error_message}")
                st.session_state.video_bytes = None
                st.session_state.video_filename = None
                st.session_state.download_path = None
                return

            if job.status != "completed":
                st.session_state.last_error = f"Unexpected job status: {job.status}"
                st.error(st.session_state.last_error)
                return

            with st.spinner("Downloading video content..."):
                content = client.videos.download_content(job.id, variant="video")
                video_bytes = binary_response_to_bytes(content)

            filename = f"{job.id}.mp4"
            st.session_state.video_bytes = video_bytes
            st.session_state.video_filename = filename
            st.session_state.job_id = job.id
            st.session_state.last_error = None
            if resized_preview is not None:
                st.session_state.resized_image_preview = resized_preview
                st.session_state.resized_image_size = size

            if st.session_state.save_to_disk:
                DOWNLOAD_DIR.mkdir(exist_ok=True)
                output_path = DOWNLOAD_DIR / filename
                output_path.write_bytes(video_bytes)
                st.session_state.download_path = output_path
            else:
                st.session_state.download_path = None

            st.success("Video generation completed! Preview below.")

        except OpenAIError as exc:
            st.session_state.last_error = str(exc)
            st.error(f"Error while polling or downloading video: {exc}")
            st.session_state.video_bytes = None
            st.session_state.video_filename = None
            st.session_state.download_path = None
            return

    if st.session_state.video_bytes:
        st.subheader("Preview")
        st.video(st.session_state.video_bytes)

        st.download_button(
            "Download MP4",
            data=st.session_state.video_bytes,
            file_name=st.session_state.video_filename or "sora-video.mp4",
            mime="video/mp4",
        )

        if st.session_state.download_path:
            st.caption(f"Saved locally to `{st.session_state.download_path}`.")

    if st.session_state.resized_image_preview is not None:
        st.subheader("Reference image (resized)")
        caption = "Resized to " + (st.session_state.resized_image_size or "selected frame size")
        st.image(st.session_state.resized_image_preview, caption=caption, use_column_width=True)

    if st.session_state.last_error:
        st.caption(f"Last error: {st.session_state.last_error}")


if __name__ == "__main__":
    main()
