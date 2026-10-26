"""LLM 呼び出しのテスト（Gemini API は呼ばない）。"""

from types import SimpleNamespace

from app import ask as ask_module


def test_ask_displays_input_and_output_token_counts(monkeypatch, capsys):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=34,
        ),
        text="回答",
    )
    call_kwargs = {}

    def generate_content(**kwargs):
        call_kwargs.update(kwargs)
        return response

    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content)
    )
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")

    assert ask_module.ask("質問") == "回答"
    assert "[usage] 入力トークン数: 12, 出力トークン数: 34" in capsys.readouterr().out
    assert call_kwargs["config"].temperature == 0.0


def test_ask_reports_when_token_usage_is_unavailable(monkeypatch, capsys):
    response = SimpleNamespace(usage_metadata=None, text="回答")
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: response)
    )
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")

    assert ask_module.ask("質問") == "回答"
    assert "トークン数を取得できませんでした" in capsys.readouterr().out
