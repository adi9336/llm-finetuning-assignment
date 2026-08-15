"""M2 masking tests: char-span metadata + token-mask proof (answer-only training)."""
from src.masking import assemble, mask_metadata, token_mask_from_offsets


class TestAssemble:
    def test_spans_point_at_answer(self):
        prompt = "What is 2 + 2?"
        answer = "4"
        text, start, end = assemble(prompt, answer)
        assert text == f"{prompt}\n\n{answer}"
        assert text[start:end] == answer
        assert start == len(prompt) + 2
        assert end == len(text)

    def test_mask_metadata(self):
        md = mask_metadata("P", "A")
        assert md["answer_start_char"] == 3  # "P" + "\n\n"
        assert md["answer_end_char"] == 4
        assert md["answer_len"] == 1


class TestTokenMask:
    def test_only_answer_tokens_trained(self):
        """The core M2 claim: with real offsets, the mask covers exactly the
        answer's tokens and nothing before it."""
        prompt = "What is 2 + 2?"
        answer = "four"
        text, start, end = assemble(prompt, answer)
        # synthetic offsets: prompt as 4 tokens, answer "four" (chars 17..21) as 2 tokens
        offsets = [(0, 4), (5, 8), (9, 12), (13, 15), (17, 19), (19, 21)]
        mask = token_mask_from_offsets(offsets, start, end)
        assert mask == [False, False, False, False, True, True]
        assert sum(mask) == 2  # exactly the two answer tokens

    def test_no_answer_means_no_tokens(self):
        offsets = [(0, 3), (4, 7)]
        assert token_mask_from_offsets(offsets, 0, 0) == [False, False]

    def test_offsets_within_answer_all_flagged(self):
        offsets = [(0, 2), (2, 4), (4, 6)]
        # answer span covers chars 2..6 → tokens 1 and 2 (token 0 ends at 2, exclusive)
        assert token_mask_from_offsets(offsets, 2, 6) == [False, True, True]
