"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import os
import re
from typing import Dict, Any, List
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket
from google import genai
from google.genai import types

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
# PERSONA
Bạn là VinAssistant, trợ lý khách hàng của Vingroup. Trả lời bằng tiếng Việt,
ngắn gọn, lịch sự và dựa trên dữ liệu thực tế.

# AVAILABLE TOOLS
- search_product_catalog: tra cứu sản phẩm theo danh mục và giá tối đa.
- submit_support_ticket: ghi nhận yêu cầu hỗ trợ của khách hàng.

# CORE RULES
Không bịa tên sản phẩm, giá, mã ticket hoặc chính sách. Dùng đúng tool khi
người dùng cần dữ liệu catalog hoặc cần gửi yêu cầu hỗ trợ.

# OPERATIONAL BOUNDARIES
Chỉ hỗ trợ sản phẩm, dịch vụ và chăm sóc khách hàng của Vingroup. Với câu hỏi
ngoài phạm vi, nói rõ giới hạn thay vì suy đoán.

# OUTPUT CONTRACT
Pipeline xử lý là Intent Detection → tool theo thứ tự → tổng hợp Final Answer.
Không hiển thị suy nghĩ nội bộ; chỉ trả về câu trả lời cuối cùng cho người dùng.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

def call_gemini_api(user_input: str) -> str:
    """
    Gọi Gemini API 1 lượt (không dùng tool).
    
    API key được đọc từ biến môi trường ``GEMINI_API_KEY``. Hàm dùng REST
    endpoint trực tiếp để giữ cho starter project không cần thêm dependency.

    Raises:
        ValueError: Nếu input rỗng hoặc chưa cấu hình API key.
        RuntimeError: Nếu Gemini trả về lỗi hoặc response không có nội dung.
    """
    if not isinstance(user_input, str) or not user_input.strip():
        raise ValueError("user_input must be a non-empty string.")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")

    client = genai.Client(api_key=api_key)

    interaction = client.interactions.create(
        model = model,
        input = user_input
    )

    return interaction.output_text or "[Gemini API] Không có nội dung trả về."

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        # TODO 2: Trả về câu trả lời tĩnh (mock) hoặc gọi Gemini API 1 lượt (không dùng tool)
        # Mục tiêu: Quan sát hiện tượng bịa thông tin (hallucination)
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {call_gemini_api(user_input)}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("user_input must be a non-empty string.")

        self.trace = []
        self.trace.append({"step": "init", "user_input": user_input})

        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            try:
                return self._run_with_gemini(user_input, api_key)
            except Exception as exc:
                # Keep the lab runnable without a network connection or a
                # model that does not support function calling.
                self.trace.append({"step": "provider_fallback", "error": str(exc)})

        return self._run_offline(user_input)

    def _run_with_gemini(self, prompt: str, api_key: str) -> Dict[str, Any]:
        """Run a bounded ReAct loop using Gemini for thought/action selection."""
        function_declarations = [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
            }
            for tool in TOOL_DEFINITIONS
            if tool.get("name") and tool.get("parameters")
        ]
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT or None,
            tools=[{"function_declarations": function_declarations}]
            if function_declarations
            else None,
            temperature=0.2,
        )
        client = genai.Client(api_key=api_key)
        current_prompt = prompt
        for iteration in range(1, self.max_iterations + 1):
            response = client.models.generate_content(
                model=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
                contents=current_prompt,
                config=config,
            )
            function_calls = getattr(response, "function_calls", None)
            if not function_calls:
                answer = getattr(response, "text", "") or ""
                self.trace.append(
                    {"step": "final", "iteration": iteration, "type": "text", "content": answer}
                )
                return {
                    "answer": answer,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "completed",
                }

            call = function_calls[0]
            tool_name = call.name
            arguments = dict(call.args) if getattr(call, "args", None) else {}
            tool = TOOL_MAP.get(tool_name)
            if tool is None:
                raise ValueError(f"Gemini requested an unknown tool: {tool_name}")
            self.trace.append(
                {
                    "step": "thought",
                    "iteration": iteration,
                    "content": f"Gemini chọn công cụ {tool_name}.",
                }
            )
            result = tool(**arguments)
            observation = self._format_tool_answer(tool_name, result)
            self.trace.append(
                {
                    "step": "action",
                    "iteration": iteration,
                    "tool": tool_name,
                    "arguments": arguments,
                }
            )
            self.trace.append(
                {"step": "observation", "iteration": iteration, "content": observation}
            )
            current_prompt = (
                f"{prompt}\n\nObservation từ {tool_name}: {observation}\n"
                "Hãy tiếp tục ReAct. Nếu đã đủ thông tin, trả lời cuối cùng bằng văn bản; "
                "nếu chưa đủ, hãy gọi công cụ tiếp theo."
            )

        error = f"Lỗi: Agent đã vượt quá số vòng lặp tối đa ({self.max_iterations})."
        self.trace.append({"step": "error", "content": error})
        return {
            "answer": error,
            "trace": self.trace,
            "iterations": self.max_iterations,
            "status": "max_iterations_reached",
        }

    def _run_offline(self, user_input: str) -> Dict[str, Any]:
        """Run the explicit Intent Detection -> tools -> synthesis pipeline."""
        text = user_input.lower()
        needs_catalog = any(
            word in text
            for word in ("xe điện", "xe vinfast", "du lịch", "vinpearl", "resort")
        )
        needs_ticket = any(
            word in text for word in ("bị lỗi", "hỗ trợ", "khiếu nại", "ticket", "sự cố")
        )
        is_faq = not needs_catalog and not needs_ticket and "bảo hành" in text
        intent = {
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "is_faq": is_faq,
        }
        self.trace.append({"step": "intent_detection", "intent": intent})

        if is_faq:
            answer = "Pin xe điện VinFast được bảo hành 10 năm."
            self.trace.append({"step": "final", "iteration": 1, "content": answer})
            return {"answer": answer, "trace": self.trace, "iterations": 1, "status": "completed"}

        observations = []
        if needs_catalog:
            if self.max_iterations < 1:
                return self._iteration_limit_error(1, ["search_product_catalog"])
            observations.append(self._run_catalog(user_input, text, 1))
        if needs_ticket:
            if self.max_iterations < 2:
                return self._iteration_limit_error(2, ["submit_support_ticket"])
            observations.append(self._run_ticket(user_input, text, 2))

        if not observations:
            answer = "Tôi chưa tìm thấy thông tin phù hợp. Vui lòng cung cấp thêm chi tiết."
            final_iteration = 1
        elif len(observations) == 1:
            answer = observations[0]
            final_iteration = 1 if needs_catalog else 2
        else:
            if self.max_iterations < 3:
                return self._iteration_limit_error(3, [])
            final_iteration = 3
            answer = "Tổng hợp thông tin:\n" + "\n".join(observations)

        self.trace.append({"step": "final", "iteration": final_iteration, "content": answer})
        return {
            "answer": answer,
            "trace": self.trace,
            "iterations": final_iteration,
            "status": "completed",
        }

    def _run_catalog(self, user_input: str, text: str, iteration: int) -> str:
        self.trace.append({"step": "thought", "iteration": iteration, "content": "Intent cần tra cứu catalog."})
        category = "du_lich" if any(word in text for word in ("du lịch", "vinpearl", "resort")) else "xe_dien"
        max_price = 999999999999
        price_match = re.search(r"(\d+(?:[.,]\d+)?)\s*(tỷ|triệu)", text)
        if price_match:
            amount = float(price_match.group(1).replace(",", "."))
            max_price = int(amount * (1000000000 if price_match.group(2) == "tỷ" else 1000000))
        arguments = {"category": category, "max_price": max_price}
        result = search_product_catalog(**arguments)
        return self._record_tool_result("search_product_catalog", arguments, result, iteration)

    def _run_ticket(self, user_input: str, text: str, iteration: int) -> str:
        self.trace.append({"step": "thought", "iteration": iteration, "content": "Intent cần tạo support ticket."})
        name_match = re.search(r"(?:tôi tên|tên tôi là)\s+([^,.]+)", user_input, re.IGNORECASE)
        customer_name = name_match.group(1).strip() if name_match else "Khách hàng"
        priority = "high" if any(word in text for word in ("gấp", "nghiêm trọng", "khẩn")) else "medium"
        arguments = {"customer_name": customer_name, "issue_description": user_input, "priority": priority}
        result = submit_support_ticket(**arguments)
        return self._record_tool_result("submit_support_ticket", arguments, result, iteration)

    def _record_tool_result(self, tool_name: str, arguments: Dict[str, Any], result: Any, iteration: int) -> str:
        observation = self._format_tool_answer(tool_name, result)
        self.trace.append({"step": "action", "iteration": iteration, "tool": tool_name, "arguments": arguments})
        self.trace.append({"step": "observation", "iteration": iteration, "content": observation})
        return observation

    def _iteration_limit_error(self, iteration: int, pending: List[str]) -> Dict[str, Any]:
        suffix = f" Chưa thực hiện: {', '.join(pending)}." if pending else ""
        answer = f"Lỗi: Agent đã vượt quá số vòng lặp tối đa ({self.max_iterations}).{suffix}"
        self.trace.append({"step": "error", "iteration": iteration, "content": answer})
        return {"answer": answer, "trace": self.trace, "iterations": self.max_iterations, "status": "max_iterations_reached"}

    @staticmethod
    def _format_tool_answer(tool_name: str, result: Any) -> str:
        if tool_name == "search_product_catalog":
            if not result:
                return "Rất tiếc, không tìm thấy sản phẩm phù hợp."
            return "Kết quả tìm kiếm: " + ", ".join(
                f"{item.get('name', 'Sản phẩm')} ({item.get('price_vnd', 0):,} VNĐ)"
                for item in result
            )
        if tool_name == "submit_support_ticket":
            return (
                f"Đã tạo ticket {result['ticket_id']} cho {result['customer_name']} "
                f"với mức ưu tiên {result['priority']}."
            )
        return json.dumps(result, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    from dotenv import load_dotenv
    load_dotenv()  # Load GEMINI_API_KEY từ .env nếu có

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
