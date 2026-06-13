# tests/test_rag_prompts.py
from src.prompts.rag_prompts import (
    RAG_STRUCTURED_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT,
    build_cot_analysis_messages,
    build_few_shot_compliance_messages,
    build_rag_messages,
    build_rag_structured_messages,
    build_simple_messages,
    parse_structured_response,
)

# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------


def test_build_simple_messages_no_system():
    msgs = build_simple_messages("What is GDPR?")
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert "GDPR" in msgs[0]["content"]


def test_build_simple_messages_with_system():
    msgs = build_simple_messages("What is GDPR?", system_prompt="You are an expert.")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


def test_build_rag_messages_structure():
    chunks = ["Chunk 1 content", "Chunk 2 content"]
    msgs = build_rag_messages("What is GDPR?", context_chunks=chunks)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == RAG_SYSTEM_PROMPT
    assert "Chunk 1" in msgs[1]["content"]
    assert "Chunk 2" in msgs[1]["content"]
    assert "What is GDPR?" in msgs[1]["content"]


def test_build_rag_messages_custom_system_prompt():
    msgs = build_rag_messages(
        "Question?",
        context_chunks=["chunk"],
        system_prompt="Custom prompt",
    )
    assert msgs[0]["content"] == "Custom prompt"


def test_build_rag_structured_messages_uses_structured_prompt():
    msgs = build_rag_structured_messages("Question?", context_chunks=["chunk"])
    assert msgs[0]["content"] == RAG_STRUCTURED_SYSTEM_PROMPT


def test_build_few_shot_compliance_messages_structure():
    msgs = build_few_shot_compliance_messages("We keep logs for 3 years.")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    # Examples must be present in user message
    assert "COMPLIANT" in msgs[1]["content"]
    assert "NON-COMPLIANT" in msgs[1]["content"]
    assert "We keep logs for 3 years." in msgs[1]["content"]


def test_build_cot_analysis_messages_structure():
    msgs = build_cot_analysis_messages("We store data for 10 years.")
    assert len(msgs) == 2
    assert "Step 1" in msgs[0]["content"]
    assert "Step 5" in msgs[0]["content"]
    assert "10 years" in msgs[1]["content"]


# ---------------------------------------------------------------------------
# JSON parser
# ---------------------------------------------------------------------------


def test_parse_structured_response_valid_json():
    raw = '{"answer": "6 months", "confidence": "HIGH", "citations": [], "cannot_answer": false}'
    result = parse_structured_response(raw)
    assert result["answer"] == "6 months"
    assert result["confidence"] == "HIGH"
    assert result["cannot_answer"] is False


def test_parse_structured_response_strips_markdown_fences():
    raw = '```json\n{"answer": "test", "confidence": "LOW", "citations": [], "cannot_answer": false}\n```'
    result = parse_structured_response(raw)
    assert result["answer"] == "test"


def test_parse_structured_response_invalid_json_returns_fallback():
    raw = "This is not valid JSON at all"
    result = parse_structured_response(raw)
    assert result["cannot_answer"] is True
    assert result["confidence"] == "LOW"
    assert "_parse_error" in result


def test_parse_structured_response_cannot_answer_schema():
    raw = '{"answer": null, "confidence": "LOW", "citations": [], "cannot_answer": true}'
    result = parse_structured_response(raw)
    assert result["answer"] is None
    assert result["cannot_answer"] is True
