from sky_v1.agent.base import ToolContext
from sky_v1.agent.planner import PlannerLLM


def _ctx():
    return ToolContext(session_id="p")


def test_plan_generate_cat_image():
    p = PlannerLLM()
    plan = p.plan(_ctx(), user_message="生成一张猫的图片")
    assert plan.tool_name == "tool_image_generate"


def test_plan_run_code():
    p = PlannerLLM()
    plan = p.plan(_ctx(), user_message="运行代码: 1+1", attachments=[{"type": "text"}])
    assert plan.tool_name == "tool_code"


def test_plan_image_attachment():
    p = PlannerLLM()
    plan = p.plan(_ctx(), user_message="", attachments=[{"type": "image_url"}])
    assert plan.tool_name == "tool_image_understand"


def test_plan_empty_message_no_error():
    p = PlannerLLM()
    plan = p.plan(_ctx(), user_message="")
    assert plan is not None
    assert isinstance(plan.tool_name, str)
    assert len(plan.tool_name) > 0
