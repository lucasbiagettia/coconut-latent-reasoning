from coconut.evaluation import extract_answer, normalize_answer


def test_extract_answer_uses_the_last_answer_marker():
    text = "Reasoning line.\nAnswer: provisional\nAnswer: Tom is a lempus.\n"
    assert extract_answer(text) == "Tom is a lempus."


def test_answer_normalization_is_case_and_whitespace_insensitive():
    assert normalize_answer("  Tom   IS a lempus. ") == normalize_answer(
        "tom is a lempus."
    )
