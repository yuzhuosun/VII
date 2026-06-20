"""Visual Instruction Grounding with Pillow."""

from __future__ import annotations

import random
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont

from .types import GroundedImage, ReprogrammedIntent

Position = Literal["top_left", "top", "top_right", "center", "bottom_left", "bottom", "bottom_right"]


@dataclass(slots=True)
class GroundingConfig:
    position: Position = "bottom"
    font_size: int = 28
    fill: tuple[int, int, int] = (255, 255, 255)
    alpha: int = 230
    border_fill: tuple[int, int, int] = (0, 0, 0)
    border_width: int = 2
    margin: int = 24
    max_width_ratio: float = 0.9
    random_jitter: int = 0
    seed: int | None = None


class VisualInstructionGrounder:
    """Render a visual instruction onto a safe reference image."""

    def __init__(self, config: GroundingConfig | None = None):
        self.config = config or GroundingConfig()

    def ground(
        self,
        image_path: str | Path,
        instruction: str | ReprogrammedIntent,
        output_path: str | Path,
        **overrides: object,
    ) -> GroundedImage:
        """Overlay ``instruction`` on ``image_path`` and save to ``output_path``."""

        cfg = self._merged_config(overrides)
        text = instruction.visual_instruction if isinstance(instruction, ReprogrammedIntent) else str(instruction)
        source = Path(image_path)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        rng = random.Random(cfg.seed)
        with Image.open(source).convert("RGBA") as image:
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            font = self._load_font(cfg.font_size)
            wrapped = self._wrap_text(text, font, int(image.width * cfg.max_width_ratio))
            bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, stroke_width=cfg.border_width, spacing=4)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x, y = self._position(cfg.position, image.size, (text_w, text_h), cfg.margin)
            if cfg.random_jitter:
                x += rng.randint(-cfg.random_jitter, cfg.random_jitter)
                y += rng.randint(-cfg.random_jitter, cfg.random_jitter)
            x = max(cfg.margin, min(x, image.width - text_w - cfg.margin))
            y = max(cfg.margin, min(y, image.height - text_h - cfg.margin))

            fill = (*cfg.fill, cfg.alpha)
            border_fill = (*cfg.border_fill, cfg.alpha)
            draw.multiline_text(
                (x, y),
                wrapped,
                font=font,
                fill=fill,
                spacing=4,
                stroke_width=cfg.border_width,
                stroke_fill=border_fill,
            )
            Image.alpha_composite(image, overlay).convert("RGB").save(output)

        return GroundedImage(
            source_path=str(source),
            output_path=str(output),
            instruction=text,
            position=(x, y),
            size=(text_w, text_h),
            metadata={"config": asdict(cfg)},
        )

    def _merged_config(self, overrides: dict[str, object]) -> GroundingConfig:
        values = asdict(self.config)
        values.update({k: v for k, v in overrides.items() if v is not None})
        return GroundingConfig(**values)

    @staticmethod
    def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        for font_name in ("DejaVuSans-Bold.ttf", "Arial.ttf"):
            try:
                return ImageFont.truetype(font_name, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    @staticmethod
    def _wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> str:
        avg_char_width = max(1, int(font.getlength("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") / 52))
        width = max(12, max_width // avg_char_width)
        return "\n".join(textwrap.wrap(text, width=width))

    @staticmethod
    def _position(position: Position, image_size: tuple[int, int], text_size: tuple[int, int], margin: int) -> tuple[int, int]:
        iw, ih = image_size
        tw, th = text_size
        positions = {
            "top_left": (margin, margin),
            "top": ((iw - tw) // 2, margin),
            "top_right": (iw - tw - margin, margin),
            "center": ((iw - tw) // 2, (ih - th) // 2),
            "bottom_left": (margin, ih - th - margin),
            "bottom": ((iw - tw) // 2, ih - th - margin),
            "bottom_right": (iw - tw - margin, ih - th - margin),
        }
        return positions[position]
