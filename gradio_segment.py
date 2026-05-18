"""Local Gradio UI for SAM2 image segmentation.

This UI project lives next to the local `sam2/` checkout and uses the sibling
SAM2.1 small checkpoint by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
import sys

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Sam2Paths:
    checkpoint: Path
    config: Path
    config_name: str


def default_sam2_paths(base_dir: Path | None = None) -> Sam2Paths:
    ui_dir = base_dir or Path(__file__).resolve().parent
    root = ui_dir.parent
    config_name = "configs/sam2.1/sam2.1_hiera_s.yaml"
    return Sam2Paths(
        checkpoint=root / "sam2" / "checkpoints" / "sam2.1_hiera_small.pt",
        config=root / "sam2" / "sam2" / config_name,
        config_name=config_name,
    )


def apply_mask_to_rgba(image: Image.Image, mask: np.ndarray) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    rgba.putalpha(alpha)
    return rgba


def normalize_select_point(evt: Any) -> tuple[int, int] | None:
    index = getattr(evt, "index", None)
    if not index or len(index) < 2:
        return None
    return int(index[0]), int(index[1])


def beginner_copy() -> dict[str, str]:
    return {
        "title": "AI 图片元素抠图助手",
        "intro": "不用懂抠图。按 1、2、3 做：上传图片、告诉 AI 要哪里、开始抠图。",
        "want_button": "我要这块",
        "not_button": "不要这块",
        "clear_button": "重新选",
        "run_button": "开始抠图",
        "candidate_label": "换个结果",
    }


def add_prompt_point(
    prompts: list[tuple[tuple[int, int], int]] | None,
    point: tuple[int, int] | None,
    mode: str,
) -> tuple[list[tuple[tuple[int, int], int]], str]:
    current = list(prompts or [])
    if point is None:
        return current, "请先在图片上点击一个位置。"

    label = 0 if "排除" in mode else 1
    current.append((point, label))
    positive = sum(1 for _, item_label in current if item_label == 1)
    negative = sum(1 for _, item_label in current if item_label == 0)
    return current, f"已添加 {mode}：({point[0]}, {point[1]})。当前目标点 {positive} 个，排除点 {negative} 个。"


def clear_prompt_points() -> tuple[list[tuple[tuple[int, int], int]], str]:
    return [], "已清空所有点。"


def prompt_arrays(prompts: list[tuple[tuple[int, int], int]]) -> tuple[np.ndarray, np.ndarray]:
    points = np.array([[point[0], point[1]] for point, _ in prompts], dtype=np.int32)
    labels = np.array([label for _, label in prompts], dtype=np.int32)
    return points, labels


def pick_candidate(
    masks: np.ndarray,
    scores: np.ndarray,
    candidate_number: int,
) -> tuple[np.ndarray, float, int]:
    if len(masks) == 0:
        raise ValueError("SAM2 没有返回候选 mask。")
    idx = max(0, min(int(candidate_number) - 1, len(masks) - 1))
    return masks[idx].astype(bool), float(scores[idx]), idx


def draw_prompt_overlay(
    image: Image.Image | None,
    prompts: list[tuple[tuple[int, int], int]] | None,
) -> Image.Image | None:
    if image is None:
        return None

    marked = image.convert("RGBA")
    draw = ImageDraw.Draw(marked)
    radius = max(5, min(marked.size) // 80)

    for point, label in prompts or []:
        x, y = point
        if label == 1:
            fill = (0, 220, 0, 255)
            outline = (255, 255, 255, 255)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=2)
        else:
            color = (255, 0, 0, 255)
            draw.line((x - radius, y - radius, x + radius, y + radius), fill=color, width=4)
            draw.line((x - radius, y + radius, x + radius, y - radius), fill=color, width=4)

    return marked


def prepare_uploaded_image(image: Image.Image | None):
    if image is None:
        return None, None, [], None, "先把图片放进左边的上传框里。"
    original = image.convert("RGB")
    preview = original.copy()
    return original, preview, [], None, "图片已放好。现在点一下你想抠出来的东西，再点“我要这块”。"


class Sam2Segmenter:
    def __init__(self, paths: Sam2Paths, device_preference: str = "auto") -> None:
        self.paths = paths
        self.device_preference = device_preference
        self.device = None
        self.predictor = None

    def _validate_paths(self) -> None:
        missing = []
        if not self.paths.checkpoint.exists():
            missing.append(str(self.paths.checkpoint))
        if not self.paths.config.exists():
            missing.append(str(self.paths.config))
        if missing:
            raise FileNotFoundError("缺少 SAM2 文件：\n" + "\n".join(missing))

    def _choose_device(self):
        import torch

        if self.device_preference == "cpu":
            return torch.device("cpu")
        if self.device_preference == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if self.device_preference == "auto" and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def load(self) -> None:
        if self.predictor is not None:
            return

        self._validate_paths()

        ui_dir = Path(__file__).resolve().parent
        removed_ui_dir = False
        if str(ui_dir) in sys.path:
            sys.path.remove(str(ui_dir))
            removed_ui_dir = True

        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            self.device = self._choose_device()
            model = build_sam2(
                self.paths.config_name,
                str(self.paths.checkpoint),
                device=self.device,
                apply_postprocessing=True,
            )
            self.predictor = SAM2ImagePredictor(model)

            if self.device.type == "mps":
                torch.mps.empty_cache()
        finally:
            if removed_ui_dir:
                sys.path.insert(0, str(ui_dir))

    def predict(
        self,
        image: Image.Image,
        prompts: list[tuple[tuple[int, int], int]],
        candidate_number: int,
    ) -> tuple[np.ndarray, float, int, int]:
        self.load()
        image_rgb = np.array(image.convert("RGB"))
        self.predictor.set_image(image_rgb)
        points, labels = prompt_arrays(prompts)

        masks, scores, _ = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        mask, score, idx = pick_candidate(masks, scores, candidate_number)
        return mask, score, idx, len(masks)


PATHS = default_sam2_paths()
SEGMENTER = Sam2Segmenter(PATHS)


def overlay_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    base = image.convert("RGBA")
    color = Image.new("RGBA", base.size, (48, 132, 255, 0))
    alpha = Image.fromarray((mask.astype(np.uint8) * 120), mode="L")
    color.putalpha(alpha)
    return Image.alpha_composite(base, color)


def save_png_temp(image: Image.Image) -> str:
    temp = NamedTemporaryFile(delete=False, suffix=".png")
    temp.close()
    image.save(temp.name)
    return temp.name


def run_segmentation(
    image: Image.Image | None,
    prompts: list[tuple[tuple[int, int], int]] | None,
    candidate_number: int,
):
    if image is None:
        return None, None, None, "先把图片放进左边的上传框里。"
    if not prompts:
        return None, None, None, "先在图片上点一下你想抠出来的东西，再点“我要这块”。"

    try:
        mask, score, candidate_idx, candidate_count = SEGMENTER.predict(image, prompts, candidate_number)
        cutout = apply_mask_to_rgba(image, mask)
        preview = overlay_mask(image, mask)
        output_file = save_png_temp(cutout)
        positive = sum(1 for _, label in prompts if label == 1)
        negative = sum(1 for _, label in prompts if label == 0)
        status = (
            f"抠好了。右边第一张蓝色区域是 AI 猜的范围，第二张是透明背景结果。"
            f"如果不准，可以试试“换个结果 {candidate_idx + 1}/{candidate_count}”，"
            f"或者在多出来的地方点一下“不要这块”。当前：我要这块 {positive} 个，不要这块 {negative} 个。"
        )
        return preview, cutout, output_file, status
    except RuntimeError as exc:
        if "mps" in str(exc).lower():
            SEGMENTER.device_preference = "cpu"
            SEGMENTER.predictor = None
            return None, None, None, "MPS 运行失败。已切换到 CPU，请再次点击分割。"
        return None, None, None, f"运行失败：{exc}"
    except Exception as exc:
        return None, None, None, f"运行失败：{exc}"


def add_prompt_and_overlay(
    original_image: Image.Image | None,
    prompts: list[tuple[tuple[int, int], int]] | None,
    point: tuple[int, int] | None,
    mode: str,
):
    updated, message = add_prompt_point(prompts, point, mode)
    return updated, draw_prompt_overlay(original_image, updated), message


def clear_prompt_and_overlay(original_image: Image.Image | None):
    updated, message = clear_prompt_points()
    return updated, draw_prompt_overlay(original_image, updated), None, message


def remember_click(evt: gr.SelectData):
    point = normalize_select_point(evt)
    if point is None:
        return None, "没有获取到点击坐标。"
    return point, f"已点到这里：({point[0]}, {point[1]})。如果这是你想要的地方，点“我要这块”；如果这是多出来的地方，点“不要这块”。"


def build_app() -> gr.Blocks:
    copy = beginner_copy()
    description = f"""
{copy["intro"]}

**你只需要记住：** 绿色圆点表示“我要这块”，红色叉表示“不要这块”。图片只在本机处理，不上传外网。
"""

    with gr.Blocks(title="SAM2 Local Segmenter") as app:
        gr.Markdown(f"# {copy['title']}")
        gr.Markdown(description)

        with gr.Row():
            gr.Markdown("""
### 1. 上传图片
把 PNG/JPG 放到左边。
""")
            gr.Markdown("""
### 2. 告诉 AI 要哪里
点图片里的目标，然后点“我要这块”。选多了就点多出来的位置，再点“不要这块”。
""")
            gr.Markdown("""
### 3. 开始抠图
点“开始抠图”。不准就试试“换个结果”。
""")

        point_state = gr.State(value=None)
        prompts_state = gr.State(value=[])
        original_image_state = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="左边点图：先点你想抠的地方",
                    type="pil",
                    interactive=True,
                    height=520,
                )
                click_status = gr.Textbox(label="小助手提示", interactive=False)
                with gr.Row():
                    add_positive_button = gr.Button(copy["want_button"], variant="primary")
                    add_negative_button = gr.Button(copy["not_button"])
                with gr.Row():
                    clear_points_button = gr.Button(copy["clear_button"])
                    segment_button = gr.Button(copy["run_button"], variant="primary")
                candidate_number = gr.Radio(
                    choices=[1, 2, 3],
                    value=1,
                    label=copy["candidate_label"],
                    info="AI 一次会猜 3 种结果。不准时，换 2 或 3 再点“开始抠图”。",
                )
            with gr.Column(scale=1):
                preview_image = gr.Image(label="AI 猜的范围（蓝色就是会被抠出来的地方）", type="pil", height=260)
                cutout_image = gr.Image(label="抠出来的透明图", type="pil", height=260)
                download_file = gr.File(label="下载 PNG")
                run_status = gr.Textbox(label="下一步怎么做", interactive=False)

        input_image.select(
            fn=remember_click,
            inputs=None,
            outputs=[point_state, click_status],
        )
        input_image.upload(
            fn=prepare_uploaded_image,
            inputs=[input_image],
            outputs=[original_image_state, input_image, prompts_state, point_state, click_status],
        )
        add_positive_button.click(
            fn=add_prompt_and_overlay,
            inputs=[original_image_state, prompts_state, point_state, gr.State("添加目标点")],
            outputs=[prompts_state, input_image, click_status],
        )
        add_negative_button.click(
            fn=add_prompt_and_overlay,
            inputs=[original_image_state, prompts_state, point_state, gr.State("添加排除点")],
            outputs=[prompts_state, input_image, click_status],
        )
        clear_points_button.click(
            fn=clear_prompt_and_overlay,
            inputs=[original_image_state],
            outputs=[prompts_state, input_image, point_state, click_status],
        )
        segment_button.click(
            fn=run_segmentation,
            inputs=[original_image_state, prompts_state, candidate_number],
            outputs=[preview_image, cutout_image, download_file, run_status],
        )

        gr.Markdown(
            """
## 看不准时怎么办？
1. 结果太大：在多出来的地方点一下，再点“不要这块”。
2. 结果太小：在漏掉的地方点一下，再点“我要这块”。
3. 形状不对：把“换个结果”切到 2 或 3，再点“开始抠图”。
4. 乱了：点“重新选”，从第一步再来。
"""
        )

    return app


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", server_port=7860)
