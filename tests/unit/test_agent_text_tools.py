from sky_v1.agent.base import ToolContext
from sky_v1.agent.tools.text_tools import ChatTool, CodeTool, RagTool


def _ctx():
    return ToolContext(session_id="t")


def test_chat_returns_simulated():
    result = ChatTool().run(_ctx(), prompt="hi")
    assert result.success is True
    assert "[SIM]" in result.output or "[SIMULATED]" in result.output


def test_code_run_hello():
    result = CodeTool().run(_ctx(), language="python", code="print('hello')")
    assert result.success is True
    assert "hello" in result.output


def test_rag_no_kb():
    result = RagTool().run(_ctx(), query="X")
    assert result.success is True
    lowered = result.output.lower()
    has_no_kb = "没有知识库" in result.output
    has_general = "通用摘要" in result.output
    has_sim = "sim" in lowered
    assert has_no_kb or has_general or has_sim
