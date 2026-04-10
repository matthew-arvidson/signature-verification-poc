"""
Gradio demo for the offline signature verifier POC.

The app accepts two signature images, preprocesses them into the same 224x224
representation used for training, and shows a similarity result with a simple
match / review / mismatch verdict.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from data_preprocessing import PreprocessConfig, preprocess_signature_pipeline
from inference import compare_signatures

CHECKPOINT_PATH = Path("checkpoint.pt")
MATCH_THRESHOLD = 0.45
REVIEW_THRESHOLD = 0.75
PREPROCESS_CONFIG = PreprocessConfig(signature_search_top=0.0)
DISCLAIMER_TEXT = (
    "POC only: this controlled demo compares signature similarity on limited sample data. "
    "It is not a production fraud decisioning system."
)


def ensure_bgr_uint8(image: np.ndarray) -> np.ndarray:
    """Convert Gradio image arrays into uint8 BGR for OpenCV-style preprocessing."""
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3:
        raise ValueError("Expected a 2D or 3D image array")

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    raise ValueError("Expected image with 1, 3, or 4 channels")


def preprocess_preview(image: np.ndarray) -> np.ndarray:
    """Return a view of the processed signature crop used by the model."""
    bgr = ensure_bgr_uint8(image)
    gray = preprocess_signature_pipeline(
        bgr,
        config=PREPROCESS_CONFIG,
        return_binary=False,
    )
    return np.stack([gray] * 3, axis=-1)


def verdict_html(verdict: str) -> str:
    """Render a simple colored verdict badge."""
    styles = {
        "match": ("#166534", "#dcfce7", "Likely Match"),
        "review": ("#92400e", "#fef3c7", "Needs Review"),
        "mismatch": ("#991b1b", "#fee2e2", "Likely Mismatch"),
    }
    text_color, bg_color, label = styles.get(verdict, ("#1f2937", "#e5e7eb", verdict.title()))
    return (
        f"<div style='display:inline-block;padding:10px 14px;border-radius:10px;"
        f"font-weight:600;color:{text_color};background:{bg_color};'>{label}</div>"
    )


def compare_for_demo(reference_image: np.ndarray, questioned_image: np.ndarray):
    """Gradio callback for comparing two uploaded signature images."""
    if reference_image is None or questioned_image is None:
        raise gr.Error("Please upload both a reference signature and a questioned signature.")

    reference_bgr = ensure_bgr_uint8(reference_image)
    questioned_bgr = ensure_bgr_uint8(questioned_image)

    reference_preview = preprocess_preview(reference_bgr)
    questioned_preview = preprocess_preview(questioned_bgr)

    result = compare_signatures(
        left_image_path=reference_bgr,
        right_image_path=questioned_bgr,
        checkpoint_path=CHECKPOINT_PATH,
        match_threshold=MATCH_THRESHOLD,
        review_threshold=REVIEW_THRESHOLD,
    )

    metrics_markdown = (
        f"### Similarity Result\n"
        f"- Similarity: **{result.similarity_percent:.1f}%**\n"
        f"- Distance: **{result.distance:.4f}**\n"
        f"- Thresholds: match `<= {MATCH_THRESHOLD:.2f}`, review `<= {REVIEW_THRESHOLD:.2f}`"
    )
    return (
        reference_preview,
        questioned_preview,
        metrics_markdown,
        verdict_html(result.verdict),
        DISCLAIMER_TEXT,
    )


def build_demo() -> gr.Blocks:
    """Construct the Gradio interface."""
    with gr.Blocks(title="Check Signature Verifier") as demo:
        gr.Markdown(
            """
            # Check Signature Verifier
            Upload a reference signature and a questioned signature to compare them
            using the trained Siamese embedding model.
            """
        )

        with gr.Row():
            reference_input = gr.Image(
                label="Reference Signature",
                type="numpy",
                image_mode="RGB",
            )
            questioned_input = gr.Image(
                label="Questioned Signature",
                type="numpy",
                image_mode="RGB",
            )

        compare_button = gr.Button("Compare", variant="primary")

        with gr.Row():
            reference_output = gr.Image(
                label="Processed Reference Crop",
                type="numpy",
                interactive=False,
            )
            questioned_output = gr.Image(
                label="Processed Questioned Crop",
                type="numpy",
                interactive=False,
            )

        metrics_output = gr.Markdown()
        verdict_output = gr.HTML()
        disclaimer_output = gr.Textbox(
            label="Disclaimer",
            interactive=False,
            lines=2,
        )

        compare_button.click(
            fn=compare_for_demo,
            inputs=[reference_input, questioned_input],
            outputs=[
                reference_output,
                questioned_output,
                metrics_output,
                verdict_output,
                disclaimer_output,
            ],
        )

        gr.Examples(
            examples=[
                ["data/samples/genuine/genuine_1.png", "data/samples/genuine/genuine_2.png"],
                ["data/samples/genuine/genuine_1.png", "data/samples/other/other_1.png"],
            ],
            inputs=[reference_input, questioned_input],
        )

    return demo


if __name__ == "__main__":
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {CHECKPOINT_PATH}. Run train.py before launching the demo."
        )
    app = build_demo()
    app.launch()
