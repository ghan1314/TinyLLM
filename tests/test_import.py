def test_tinyllm_import() -> None:
    import tinyllm

    assert tinyllm.TinyLLMConfig().model_type == "tinyllm"
