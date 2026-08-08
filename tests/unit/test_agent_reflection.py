from sky_v1.agent.base import ToolContext
from sky_v1.agent.reflection import ReflectionEngine


def _ctx():
    return ToolContext(session_id="r")


def test_reflection_normal_long_answer():
    engine = ReflectionEngine()
    long_answer = (
        "这是一个非常详细和完整的回答，覆盖了问题的各个方面。"
        "首先我们需要了解背景知识，然后逐步分析各个组成部分。"
        "最后给出结论以及未来的研究方向。"
    )
    needs_rewrite, feedback = engine.review(None, _ctx(), long_answer, [])
    assert needs_rewrite is False
    assert feedback == "OK"


def test_reflection_hallucination_keyword():
    engine = ReflectionEngine()
    answer = "我确定100%不存在这个东西，你可以放心。"
    needs_rewrite, feedback = engine.review(None, _ctx(), answer, [])
    assert needs_rewrite is True
