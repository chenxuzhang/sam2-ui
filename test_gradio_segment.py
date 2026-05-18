import importlib.util
import pathlib
import sys

import numpy as np
from PIL import Image


MODULE_PATH = pathlib.Path(__file__).with_name("gradio_segment.py")
spec = importlib.util.spec_from_file_location("gradio_segment", MODULE_PATH)
gradio_segment = importlib.util.module_from_spec(spec)
sys.modules["gradio_segment"] = gradio_segment
spec.loader.exec_module(gradio_segment)


def test_default_sam2_paths_point_to_sibling_small_model():
    paths = gradio_segment.default_sam2_paths(pathlib.Path("/tmp/ai-tools/sam2-ui"))

    assert paths.checkpoint == pathlib.Path("/tmp/ai-tools/sam2/checkpoints/sam2.1_hiera_small.pt")
    assert paths.config == pathlib.Path("/tmp/ai-tools/sam2/sam2/configs/sam2.1/sam2.1_hiera_s.yaml")
    assert paths.config_name == "configs/sam2.1/sam2.1_hiera_s.yaml"


def test_apply_mask_to_rgba_makes_unmasked_pixels_transparent():
    image = Image.new("RGB", (2, 2), (10, 20, 30))
    mask = np.array([[True, False], [False, True]])

    result = gradio_segment.apply_mask_to_rgba(image, mask)

    assert result.mode == "RGBA"
    assert result.getpixel((0, 0)) == (10, 20, 30, 255)
    assert result.getpixel((1, 0)) == (10, 20, 30, 0)
    assert result.getpixel((0, 1)) == (10, 20, 30, 0)
    assert result.getpixel((1, 1)) == (10, 20, 30, 255)


def test_normalize_select_data_extracts_point_tuple():
    class SelectData:
        index = (123, 456)

    assert gradio_segment.normalize_select_point(SelectData()) == (123, 456)


def test_add_prompt_point_appends_positive_and_negative_labels():
    prompts = []

    updated, message = gradio_segment.add_prompt_point(prompts, (10, 20), "添加目标点")
    updated, message = gradio_segment.add_prompt_point(updated, (30, 40), "添加排除点")

    assert updated == [((10, 20), 1), ((30, 40), 0)]
    assert "目标点 1 个" in message
    assert "排除点 1 个" in message


def test_prompt_arrays_convert_points_and_labels():
    points, labels = gradio_segment.prompt_arrays([((10, 20), 1), ((30, 40), 0)])

    assert points.tolist() == [[10, 20], [30, 40]]
    assert labels.tolist() == [1, 0]


def test_pick_candidate_clamps_to_available_masks():
    masks = np.array([
        [[True, False]],
        [[False, True]],
    ])
    scores = np.array([0.2, 0.8])

    mask, score, idx = gradio_segment.pick_candidate(masks, scores, 3)

    assert mask.tolist() == [[False, True]]
    assert score == 0.8
    assert idx == 1


def test_draw_prompt_overlay_marks_positive_and_negative_points():
    image = Image.new("RGB", (80, 80), (255, 255, 255))

    result = gradio_segment.draw_prompt_overlay(
        image,
        [((20, 20), 1), ((60, 60), 0)],
    )

    assert result.mode == "RGBA"
    assert result.getpixel((20, 20))[:3] == (0, 220, 0)
    assert result.getpixel((60, 60))[:3] == (255, 0, 0)


def test_beginner_copy_uses_plain_language():
    copy = gradio_segment.beginner_copy()

    assert "AI 图片元素抠图助手" in copy["title"]
    assert "不用懂抠图" in copy["intro"]
    assert copy["want_button"] == "我要这块"
    assert copy["not_button"] == "不要这块"
    assert copy["run_button"] == "开始抠图"
    assert copy["candidate_label"] == "换个结果"


def test_prepare_uploaded_image_keeps_original_clean_and_shows_preview():
    image = Image.new("RGB", (10, 10), (1, 2, 3))

    original, preview, prompts, point, message = gradio_segment.prepare_uploaded_image(image)

    assert original.getpixel((0, 0)) == (1, 2, 3)
    assert preview.getpixel((0, 0))[:3] == (1, 2, 3)
    assert prompts == []
    assert point is None
    assert "图片已放好" in message
