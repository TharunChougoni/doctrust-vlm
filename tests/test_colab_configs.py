from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SMOL_CONFIG = ROOT / "configs" / "colab_smolvlm_2b.yaml"
QWEN_CONFIG = ROOT / "configs" / "colab_qwen2_5_vl_3b.yaml"


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_colab_models_share_the_exact_prepared_inputs() -> None:
    smol = load(SMOL_CONFIG)
    qwen = load(QWEN_CONFIG)
    assert smol["data"] == qwen["data"]
    assert smol["variants"] == qwen["variants"]


def test_every_non_clean_colab_variant_declares_a_transform() -> None:
    for variant in load(SMOL_CONFIG)["variants"]:
        if variant["name"] != "clean":
            assert variant.get("transform")


def test_two_model_outputs_do_not_collide() -> None:
    smol = load(SMOL_CONFIG)
    qwen = load(QWEN_CONFIG)
    assert smol["output"]["predictions"] != qwen["output"]["predictions"]
    assert smol["output"]["metrics"] != qwen["output"]["metrics"]
