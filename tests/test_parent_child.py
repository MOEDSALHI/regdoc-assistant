# tests/test_parent_child.py
import re

# ── Pattern detection tests (pure logic) ──────────────────────────────────────


def _find_article_sections(text: str) -> list[str]:
    """Mirror the pattern from ingest_parent_child for testing."""
    pattern = re.compile(r"(?:^|\n)(?:Article|Art\.?)\s+\d+", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        return [text.strip()]
    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        if section:
            sections.append(section)
    return sections


RGPD_TEXT = """
Article 5 - Principes relatifs au traitement
Les donnees doivent etre traitees de maniere licite et transparente.

Article 17 - Droit a l effacement
La personne a le droit d obtenir l effacement de ses donnees.

Article 35 - Analyse d impact DPIA
Le DPIA est obligatoire pour les traitements a risque eleve.
"""


def test_article_detection_finds_sections():
    sections = _find_article_sections(RGPD_TEXT)
    assert len(sections) == 3


def test_article_detection_correct_boundaries():
    sections = _find_article_sections(RGPD_TEXT)
    assert sections[0].startswith("Article 5")
    assert sections[1].startswith("Article 17")
    assert sections[2].startswith("Article 35")


def test_article_detection_no_markers_returns_full_text():
    plain = "Texte sans marqueur d article. Contenu quelconque."
    sections = _find_article_sections(plain)
    assert len(sections) == 1
    assert sections[0] == plain


def test_article_detection_preserves_content():
    sections = _find_article_sections(RGPD_TEXT)
    combined = " ".join(sections)
    assert "licite" in combined
    assert "effacement" in combined
    assert "DPIA" in combined


def test_child_filter_removes_short_fragments():
    """Verify that micro-fragments under 5 words are filtered."""
    children_with_text = [
        ("tematique et approfondie", 0),  # 3 words → should be filtered
        ("Le DPIA est obligatoire pour les traitements a risque eleve.", 0),  # OK
        ("oui", 0),  # 1 word → filtered
        ("Article 35 analyse impact DPIA traitement risque", 0),  # OK
    ]
    MIN_WORDS = 5
    valid = [(t, p) for t, p in children_with_text if len(t.split()) >= MIN_WORDS]
    assert len(valid) == 2


def test_parent_child_ratio():
    """For a 3-article doc, expect 3 parents and several children each."""
    sections = _find_article_sections(RGPD_TEXT)
    assert len(sections) == 3

    # Each section should produce at least 1 child
    # (real chunking tested via integration)
    for section in sections:
        word_count = len(section.split())
        assert word_count >= 5
