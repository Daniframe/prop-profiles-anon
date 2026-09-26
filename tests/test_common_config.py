from src.propensity.common.config import load_config


def test_load_config_reads_sector_yaml(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "generation.yaml").write_text("n_per_level: 50\nname: RA\n")

    config = load_config("generation", config_dir=str(config_dir))

    assert config == {"n_per_level": 50, "name": "RA"}


def test_load_config_resolves_env_var_placeholders(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_TEST_SECRET", "shh")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "annotation.yaml").write_text("api_key: ${MY_TEST_SECRET}\n")

    config = load_config("annotation", config_dir=str(config_dir))

    assert config["api_key"] == "shh"


def test_load_config_resolves_nested_placeholders(tmp_path, monkeypatch):
    monkeypatch.setenv("NESTED_SECRET", "nested-value")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "inference.yaml").write_text(
        "azure:\n  api_key: ${NESTED_SECRET}\nlevels: [-1, 0, 1]\n"
    )

    config = load_config("inference", config_dir=str(config_dir))

    assert config["azure"]["api_key"] == "nested-value"
    assert config["levels"] == [-1, 0, 1]


def test_load_config_leaves_non_placeholder_strings_untouched(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "surfaces.yaml").write_text("data_path: 'data/inference/outcomes'\n")

    config = load_config("surfaces", config_dir=str(config_dir))

    assert config["data_path"] == "data/inference/outcomes"
