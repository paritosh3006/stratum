import pytest

from stratum.metrics import s2_retrieval as s2
from stratum.metrics import s4_rendering as s4


class TestRetrieval:
    def test_recall_partial(self):
        assert s2.recall_at_k(["a", "b", "c"], ["a", "z"], 5) == 0.5

    def test_recall_respects_k(self):
        assert s2.recall_at_k(["x", "y", "a"], ["a"], 2) == 0.0

    def test_no_gold_returns_none_not_zero(self):
        # Excluded from the mean rather than dragging it down.
        assert s2.recall_at_k(["a"], [], 5) is None

    def test_mrr_uses_first_hit(self):
        assert s2.mrr(["x", "a"], ["a"]) == 0.5

    def test_ndcg_perfect_ordering(self):
        assert s2.ndcg_at_k(["a", "b"], ["a", "b"], 5) == pytest.approx(1.0)


class TestPlaceholders:
    def test_preserved(self):
        r = s4.check_placeholders("EMI {amount} due {date}", "EMI {amount} due {date}")
        assert r.passed

    def test_translated_identifier_caught(self):
        r = s4.check_placeholders("EMI {amount}", "EMI {தொகை}")
        assert not r.passed
        assert "{amount}" in r.missing

    def test_dropped_placeholder_caught(self):
        r = s4.check_placeholders("EMI {amount}", "EMI amount")
        assert not r.passed

    def test_no_placeholders_is_not_a_failure(self):
        r = s4.check_placeholders("plain text", "plain text")
        assert r.passed and "no placeholders" in r.detail

    def test_multiple_syntaxes(self):
        found = s4.PLACEHOLDER_RE.findall("{a} {{b}} %(c)s ${d}")
        assert len(found) == 4


class TestNumerals:
    def test_devanagari_digits_fold(self):
        assert s4.fold_digits("४५") == "45"

    def test_tamil_digits_fold(self):
        assert s4.fold_digits("\u0BE7\u0BE8") == "12"

    def test_grouping_difference_is_not_an_error(self):
        # 1,50,000 vs 150000 is formatting, not corruption.
        assert s4.check_numerals("pay 1,50,000", "pay 150000").passed

    def test_altered_value_caught(self):
        r = s4.check_numerals("sum 500000", "sum 50000")
        assert not r.passed
        assert "500000" in r.missing

    def test_cross_script_numeral_preserved(self):
        assert s4.check_numerals("age 45", "வயது ४५").passed


class TestGlossary:
    @pytest.fixture
    def gl(self):
        return s4.Glossary(
            terms={"premium": {"ta-Taml": ["பிரீமியம்"]}},
            forbidden={"premium": {"ta-Taml": ["கட்டணம்"]}},
        )

    def test_approved_form_passes(self, gl):
        assert s4.check_glossary("premium amount", "பிரீமியம் தொகை", gl, "ta-Taml").passed

    def test_off_glossary_caught(self, gl):
        assert not s4.check_glossary("premium amount", "கட்டணம் தொகை", gl, "ta-Taml").passed

    def test_drift_caught_even_when_approved_present(self, gl):
        # Both forms in one answer: the user cannot map it back to the doc.
        r = s4.check_glossary("premium", "பிரீமியம் ... கட்டணம்", gl, "ta-Taml")
        assert not r.passed

    def test_term_not_in_source_is_out_of_scope(self, gl):
        assert s4.check_glossary("dental cover", "பல் சிகிச்சை", gl, "ta-Taml").passed
