from cv_screener.shared import ollama_client


def test_chat_json_parses_object_embedded_in_extra_text(monkeypatch):
    class FakeClient:
        def chat(self, **kwargs):
            return {"message": {"content": 'Sure, here it is:\n{"a": 1, "b": "x"}\nDone.'}}

    monkeypatch.setattr(ollama_client, "raw_client", lambda: FakeClient())

    result = ollama_client.chat_json("prompt", system="sys")

    assert result == {"a": 1, "b": "x"}
