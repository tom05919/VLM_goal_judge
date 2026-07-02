#!/usr/bin/env python3
"""Grounded DINO + SAM2 image segmentation. File path or PIL frame in, masks out."""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
GSAM_ROOT = SCRIPT_DIR / "Grounded-SAM-2"
sys.path.insert(0, str(GSAM_ROOT))

DEFAULT_IMAGE = SCRIPT_DIR.parent / "run_images/run3_no-stop/1_ex_omnivla_edge.jpg"
DEFAULT_GROUNDING_MODEL = "IDEA-Research/grounding-dino-base"
DEFAULT_SAM2_CKPT = GSAM_ROOT / "checkpoints/sam2.1_hiera_small.pt"
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
OUTPUT_DIR = SCRIPT_DIR / "outputs"


@dataclass
class SegmentationResult:
    masks: np.ndarray  # (n, H, W) bool
    boxes: np.ndarray  # (n, 4) xyxy
    labels: list[str]
    scores: list[float]

    def to_dict(self, image_source: str, width: int, height: int) -> dict:
        return {
            "image_source": image_source,
            "annotations": [
                {
                    "class_name": label,
                    "bbox": box,
                    "score": score,
                }
                for label, box, score in zip(
                    self.labels, self.boxes.tolist(), self.scores
                )
            ],
            "box_format": "xyxy",
            "img_width": width,
            "img_height": height,
        }


class GroundedSAM2Segmenter:
    def __init__(
        self,
        grounding_model: str = DEFAULT_GROUNDING_MODEL,
        sam2_checkpoint: Path = DEFAULT_SAM2_CKPT,
        sam2_config: str = DEFAULT_SAM2_CONFIG,
        device: str | None = None,
    ):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        if not sam2_checkpoint.is_file():
            raise FileNotFoundError(
                f"SAM2 checkpoint missing: {sam2_checkpoint}\n"
                "curl -L -o "
                f"{sam2_checkpoint} "
                "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(grounding_model, local_files_only=True)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            grounding_model, local_files_only=True
        ).to(self.device)

        sam2 = build_sam2(sam2_config, str(sam2_checkpoint), device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(sam2)
        self._autocast = torch.autocast(
            device_type=self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"
        )
        if self.device == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    @staticmethod
    def _normalize_prompt(text: str) -> str:
        text = text.strip().lower()
        return text if text.endswith(".") else f"{text}."

    def segment(self, image: Image.Image, text_prompt: str) -> SegmentationResult:
        """Run on a PIL RGB image. Call this from a live camera loop later."""
        text = self._normalize_prompt(text_prompt)
        rgb = np.array(image.convert("RGB"))

        with self._autocast:
            self.sam2_predictor.set_image(rgb)
            inputs = self.processor(images=image, text=text, return_tensors="pt").to(
                self.device
            )
            with torch.no_grad():
                outputs = self.grounding_model(**inputs)

            detections = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=0.4,
                text_threshold=0.3,
                target_sizes=[image.size[::-1]],
            )[0]

            if len(detections["boxes"]) == 0:
                h, w = rgb.shape[:2]
                return SegmentationResult(
                    masks=np.zeros((0, h, w), dtype=bool),
                    boxes=np.zeros((0, 4), dtype=np.float32),
                    labels=[],
                    scores=[],
                )

            boxes = detections["boxes"].cpu().numpy()
            masks, _, _ = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes,
                multimask_output=False,
            )
            if masks.ndim == 4:
                masks = masks.squeeze(1)

        return SegmentationResult(
            masks=masks.astype(bool),
            boxes=boxes,
            labels=detections["labels"],
            scores=detections["scores"].cpu().numpy().tolist(),
        )


def save_outputs(
    result: SegmentationResult,
    image_source: str,
    width: int,
    height: int,
    output_dir: Path,
    stem: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "masks": torch.from_numpy(result.masks),
            "boxes": torch.from_numpy(result.boxes),
            "labels": result.labels,
            "scores": result.scores,
        },
        output_dir / f"{stem}_masks.pt",
    )
    with open(output_dir / f"{stem}_segmentation.json", "w", encoding="utf-8") as f:
        json.dump(result.to_dict(image_source, width, height), f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grounded-SAM-2 image segmentation")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--text-prompt", default="door.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--grounding-model", default=DEFAULT_GROUNDING_MODEL)
    parser.add_argument("--sam2-checkpoint", type=Path, default=DEFAULT_SAM2_CKPT)
    parser.add_argument("--sam2-config", default=DEFAULT_SAM2_CONFIG)
    args = parser.parse_args()

    segmenter = GroundedSAM2Segmenter(
        grounding_model=args.grounding_model,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
    )

    image = Image.open(args.image)
    result = segmenter.segment(image, args.text_prompt)

    save_outputs(
        result,
        str(args.image),
        image.width,
        image.height,
        args.output_dir,
        args.image.stem,
    )

    print(f"image: {args.image}")
    print(f"prompt: {args.text_prompt}")
    print(f"detections: {len(result.labels)}")
    for label, score in zip(result.labels, result.scores):
        print(f"  {label}: {score:.3f}")
    print(f"masks shape: {result.masks.shape}")
    print(f"saved to {args.output_dir}")


if __name__ == "__main__":
    main()
