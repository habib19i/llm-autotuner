import pytest

from backend.model_repository import (
    _from_hf, _params_from_name, _active_params, _derive_provider, quant_from_filename,
    is_model_file, estimate_size_gb, get_models,
)
from backend.scoring import estimate_memory
from backend.selector import build_table, get_recommendation, pick_quant
from backend.utils import safe_model_path
from tests.conftest import make_hw


# ─── Parsing ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path,quant", [
    ("Qwen3-8B-Q4_K_M.gguf", "Q4_K_M"),
    ("Qwen3-8B-UD-Q4_K_XL.gguf", "UD-Q4_K_XL"),
    ("Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00001-of-00003.gguf", "Q4_K_M"),
    ("BF16/Model-BF16-00002-of-00010.gguf", "BF16"),
    ("gpt-oss-20b-MXFP4.gguf", "MXFP4"),
    ("Model-UD-IQ1_M.gguf", "UD-IQ1_M"),
    ("README.md", None),
    ("model.gguf", None),
])
def test_quant_from_filename(path, quant):
    assert quant_from_filename(path) == quant


def test_model_file_filter():
    assert is_model_file("Qwen3-8B-Q4_K_M.gguf")
    assert not is_model_file("mmproj-F16.gguf")
    assert not is_model_file("MTP/mtp-Qwen3.8-27B-Q4_0.gguf")
    assert not is_model_file("imatrix_unsloth.gguf_file")


@pytest.mark.parametrize("name,params", [
    ("Qwen3-235B-A22B", 235.0), ("Qwen3.5-0.8B", 0.8), ("gemma-4-31B-it", 31.0),
    ("SmolLM2-135M-Instruct", 0.135), ("Qwen3-1.7B", 1.7), ("Kimi-K2-Instruct", None),
])
def test_params_from_name(name, params):
    assert _params_from_name(name) == params


def test_active_params():
    assert _active_params("Qwen3-30B-A3B-Instruct", 30.5) == (3.0, True)
    assert _active_params("Llama-4-Scout-17B-16E-Instruct", 107.0) == (17.0, True)
    assert _active_params("gpt-oss-20b", 20.9) == (3.6, True)
    assert _active_params("Qwen3-8B", 8.2) == (8.2, False)


def test_provider_order():
    # Distills belong to DeepSeek, not to the base model's vendor
    assert _derive_provider("DeepSeek-R1-Distill-Qwen-14B") == "DeepSeek"
    assert _derive_provider("medgemma-4b-it") == "Google"
    assert _derive_provider("Qwen3-8B") == "Alibaba"


def test_size_estimate_is_realistic():
    # Qwen3-8B Q4_K_M is ~4.7 GiB on HuggingFace
    assert 4.3 < estimate_size_gb(8.19, "Q4_K_M") < 5.0


def test_from_hf_filters_non_llms():
    data = [
        {"id": "unsloth/Qwen3-8B-GGUF", "pipeline_tag": "text-generation",
         "gguf": {"total": 8190735360, "architecture": "qwen3", "context_length": 40960},
         "siblings": [{"rfilename": "Qwen3-8B-Q4_K_M.gguf"}, {"rfilename": "Qwen3-8B-Q8_0.gguf"}]},
        {"id": "unsloth/FLUX.1-dev-GGUF", "pipeline_tag": "text-to-image",
         "gguf": {"total": 11901408320, "architecture": "flux"},
         "siblings": [{"rfilename": "flux1-dev-Q4_K_M.gguf"}]},
        {"id": "unsloth/bge-small-en-v1.5-GGUF", "pipeline_tag": "feature-extraction",
         "gguf": {"total": 33212160, "architecture": "bert", "context_length": 512},
         "siblings": [{"rfilename": "bge-small-en-v1.5-Q4_K_M.gguf"}]},
        {"id": "unsloth/Qwen3-VL-8B-Instruct-GGUF", "pipeline_tag": "image-text-to-text",
         "gguf": {"total": 8190735360, "architecture": "qwen3vl", "context_length": 262144},
         "siblings": [{"rfilename": "Qwen3-VL-8B-Instruct-Q4_K_M.gguf"}, {"rfilename": "mmproj-F16.gguf"}]},
    ]
    models = _from_hf(data)
    assert [m.id for m in models] == ["unsloth/Qwen3-8B-GGUF", "unsloth/Qwen3-VL-8B-Instruct-GGUF"]
    assert models[0].params_b == pytest.approx(8.19, abs=0.01)
    assert models[0].context_k == 40
    assert {q.quant for q in models[0].quants} == {"Q4_K_M", "Q8_0"}
    assert models[1].is_vision and not models[0].is_vision


# ─── Scoring ───────────────────────────────────────────────────────────────────

def _model(rid):
    return next(m for m in get_models() if m.id == rid)


def test_small_model_fits_gpu():
    m = _model("unsloth/Qwen3-8B-GGUF")
    q = next(q for q in m.quants if q.quant == "Q4_K_M")
    r = estimate_memory(m, q, make_hw(vram=12))
    assert r.mode == "GPU" and r.fit in ("Perfect", "Good") and r.tok_s > 20


def test_huge_model_does_not_fit():
    m = _model("unsloth/Llama-3.3-70B-Instruct-GGUF")
    q, r = pick_quant(m, make_hw(vram=8, ram=16, avail=12))
    assert r.fit == "No Fit"


def test_split_offload_when_vram_too_small():
    m = _model("unsloth/Qwen3-14B-GGUF")
    q = next(q for q in m.quants if q.quant == "Q4_K_M")
    r = estimate_memory(m, q, make_hw(vram=6, ram=32, avail=24))
    assert r.mode == "CPU+GPU" and 0 < r.gpu_layers < 40


def test_integrated_gpu_does_not_double_count_memory():
    m = _model("unsloth/Qwen3-14B-GGUF")
    q = next(q for q in m.quants if q.quant == "Q8_0")  # ~14.6 GB
    igpu = make_hw(vram=8, ram=16, avail=12, vendor="AMD", integrated=True)
    assert estimate_memory(m, q, igpu).fit == "No Fit"


def test_cpu_only_path_reachable():
    m = _model("unsloth/Qwen3-1.7B-GGUF")
    q = next(q for q in m.quants if q.quant == "Q4_K_M")
    hw = make_hw(vram=0, vendor="Generic", integrated=True)
    hw.gpu.vulkan = False
    r = estimate_memory(m, q, hw)
    assert r.mode == "CPU" and r.gpu_layers == 0 and r.fit != "No Fit"


def test_moe_is_faster_than_dense_of_same_size():
    hw = make_hw(vram=8, ram=64, avail=56)
    moe, dense = _model("unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF"), _model("unsloth/Qwen3-32B-GGUF")
    qm = next(q for q in moe.quants if q.quant == "Q4_K_M")
    qd = next(q for q in dense.quants if q.quant == "Q4_K_M")
    assert estimate_memory(moe, qm, hw).tok_s > estimate_memory(dense, qd, hw).tok_s * 3


def test_table_sorted_by_fit_and_picks_best_quant():
    rows = build_table(make_hw(vram=24, ram=64, avail=48))
    order = {"Perfect": 0, "Good": 1, "Marginal": 2, "No Fit": 3}
    assert [order[r.fit] for r in rows] == sorted(order[r.fit] for r in rows)
    small = next(r for r in rows if r.id == "unsloth/Qwen3-1.7B-GGUF")
    assert small.quant == "Q8_0"  # plenty of VRAM -> highest quality non-F16 quant


def test_recommendation_prefers_4bit_or_better():
    rec = get_recommendation("unsloth/Qwen3-8B-GGUF", "coding", make_hw(vram=8))
    assert rec.row.quant not in ("Q2_K", "Q3_K_M")
    assert rec.launch_ctx >= 4096 and rec.why


# ─── Safety ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", ["../secret.gguf", "..\\..\\x.gguf", "/etc/passwd", "C:/Windows/x.gguf", "", "."])
def test_safe_model_path_rejects_escape(bad):
    with pytest.raises(ValueError):
        safe_model_path(bad)


def test_safe_model_path_accepts_nested():
    p = safe_model_path("Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf")
    assert p.name == "Qwen3-8B-Q4_K_M.gguf"


def test_two_bit_quants_never_count_as_clean_fit():
    # 30B MoE on a 12 GB machine only fits at ~2-bit: must be flagged Marginal, not Good
    m = _model("unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF")
    q, r = pick_quant(m, make_hw(vram=0, ram=16, avail=13.5, vendor="AMD", integrated=True))
    assert q.quality >= 86 or r.fit in ("Marginal", "No Fit")
