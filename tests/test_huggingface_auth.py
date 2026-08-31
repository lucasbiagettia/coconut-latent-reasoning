from coconut.huggingface_auth import load_huggingface_token


def test_dotenv_hf_token_is_optional_and_environment_wins(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text("# local secret\nHF_TOKEN='from-file'\n", encoding="utf-8")
    monkeypatch.delenv("HF_TOKEN", raising=False)

    assert load_huggingface_token(dotenv) == "from-file"

    monkeypatch.setenv("HF_TOKEN", "from-environment")
    assert load_huggingface_token(dotenv) == "from-environment"


def test_missing_dotenv_returns_none(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    assert load_huggingface_token(tmp_path / "missing.env") is None
