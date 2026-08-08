import pytest

from sky_v1.agent.memory import ShortTermMemory, LongTermMemory


def test_short_term_rolling():
    stm = ShortTermMemory(max_turns=3)
    stm.add("user", "u1")
    stm.add("assistant", "a1")
    stm.add("user", "u2")
    stm.add("assistant", "a2")
    assert len(stm) == 3


def test_short_term_invalid_role():
    stm = ShortTermMemory(max_turns=5)
    with pytest.raises(ValueError):
        stm.add("bot", "hello")


def test_long_term_remember_recall():
    ltm = LongTermMemory()
    mem_id = ltm.remember("用户喜欢吃苹果和香蕉，还有橘子。", {"topic": "preference"})
    assert isinstance(mem_id, str)
    results = ltm.recall("苹果 香蕉", top_k=3)
    assert isinstance(results, list)
    assert len(results) >= 1
