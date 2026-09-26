from src.propensity.inference.models import resolve_model, MODEL_SHORTCUTS, THINKING_MODELS


def test_resolve_model_expands_known_shortcut():
    resolved, is_thinking = resolve_model("llama33")
    assert resolved == MODEL_SHORTCUTS["llama33"]
    assert is_thinking is False


def test_resolve_model_flags_thinking_models():
    resolved, is_thinking = resolve_model("ds-r1-qwen32")
    assert resolved == MODEL_SHORTCUTS["ds-r1-qwen32"]
    assert is_thinking is True


def test_resolve_model_passes_through_unknown_key_as_full_repo_id():
    resolved, is_thinking = resolve_model("some-org/some-custom-model")
    assert resolved == "some-org/some-custom-model"
    assert is_thinking is False


def test_resolve_model_extra_shortcuts_override_registry():
    resolved, _ = resolve_model("llama33", extra_shortcuts={"llama33": "custom/override-repo"})
    assert resolved == "custom/override-repo"


def test_resolve_model_extra_shortcuts_add_new_entries():
    resolved, _ = resolve_model("my-model", extra_shortcuts={"my-model": "org/my-model-v1"})
    assert resolved == "org/my-model-v1"


def test_thinking_models_are_a_subset_of_shortcuts():
    assert THINKING_MODELS.issubset(set(MODEL_SHORTCUTS))
