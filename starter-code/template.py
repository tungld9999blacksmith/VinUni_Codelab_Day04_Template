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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from tools import (
    TOOL_DEFINITIONS,
    TOOL_MAP,
    detect_intent,
    search_product_catalog,
    submit_support_ticket,
    INTENT_DETECTION_DEFINITION
)
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
Pipeline xử lý là LLM Intent Detection → tool được chọn → tổng hợp Final Answer.
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
        """Ask the LLM for intent, then dispatch the selected business tools."""
        client = genai.Client(api_key=api_key)
        intent_definition = INTENT_DETECTION_DEFINITION
        intent_config = types.GenerateContentConfig(
            system_instruction=(
                "Bạn là bộ phân loại intent. Bắt buộc gọi function intent_detection. "
                "Trả JSON arguments gồm need_catalog, need_ticket, direct_answer. "
                "Có thể chọn đồng thời need_catalog và need_ticket; direct_answer "
                "chỉ đúng khi không cần tool."
            ),
            tools=[{"function_declarations": [intent_definition]}],
            temperature=0,
        )
        response = client.models.generate_content(
            model=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            contents=prompt,
            config=intent_config,
        )
        function_calls = getattr(response, "function_calls", None) or []
        if not function_calls or function_calls[0].name != "intent_detection":
            raise ValueError("Intent model did not return intent_detection function call.")

        arguments = dict(function_calls[0].args or {})
        intent = detect_intent(**arguments)
        self.trace.append({"step": "intent_detection", "source": "llm", "intent": intent})

        if intent["direct_answer"] and not intent["need_catalog"] and not intent["need_ticket"]:
            answer = self._ask_direct_answer(client, prompt)
            self.trace.append({"step": "final", "iteration": 2, "content": answer})
            return {"answer": answer, "trace": self.trace, "iterations": 2, "status": "completed"}

        return self._execute_intent(prompt, intent)

    def _ask_direct_answer(self, client: Any, user_input: str) -> str:
        instructions = (
            "Trả lời trực tiếp user input bằng tiếng Việt. "
            "Không gọi tool, không trả về JSON, không mô tả quy trình nội bộ. "
            "Nếu câu hỏi cần dữ liệu mà bạn không có, hãy nói rõ giới hạn."
        )

        if not client:
            api_key = os.environ.get("GEMINI_API_KEY", None)
            client = genai.Client(api_key=api_key)
        
        response = client.models.generate_content(
            model=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            contents=f"Instructions:{instructions}\n\nUser input:\n{user_input}",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=None,
                temperature=0.2,
            ),
        )
        answer = getattr(response, "text", "") or ""
        if not answer:
            raise ValueError("Direct-answer model returned empty content.")
        return answer

    def _run_offline(self, user_input: str) -> Dict[str, Any]:
        """Fallback intent detection used when the LLM is unavailable."""
        text = user_input.lower()
        intent = detect_intent(
            need_catalog=any(
                word in text
                for word in ("xe điện", "xe vinfast", "du lịch", "vinpearl", "resort")
            ),
            need_ticket=any(
                word in text
                for word in ("bị lỗi", "hỗ trợ", "khiếu nại", "ticket", "sự cố")
            ),
            direct_answer="bảo hành" in text,
        )
        self.trace.append({"step": "intent_detection", "intent": intent})

        return self._execute_intent(user_input, intent)

    def _execute_intent(self, user_input: str, intent: Dict[str, bool]) -> Dict[str, Any]:
        text = user_input.lower()
        if intent["direct_answer"] and not intent["need_catalog"] and not intent["need_ticket"]:
            answer = self._ask_direct_answer(None, user_input)
            self.trace.append({"step": "final", "iteration": 1, "content": answer})
            return {"answer": answer, "trace": self.trace, "iterations": 1, "status": "completed"}

        actions = []
        if intent["need_catalog"]:
            actions.append(("search_product_catalog", lambda: self._run_catalog(user_input, text, 2)))
        if intent["need_ticket"]:
            actions.append(("submit_support_ticket", lambda: self._run_ticket(user_input, text, 2)))

        if not actions:
            answer = "Tôi chưa tìm thấy thông tin phù hợp. Vui lòng cung cấp thêm chi tiết."
            final_iteration = 1
        else:
            if self.max_iterations < 1:
                return self._iteration_limit_error(2, [name for name, _ in actions])
            observations = self._run_tools_in_parallel(actions)
            answer = observations[0] if len(observations) == 1 else "Tổng hợp thông tin:\n" + "\n".join(observations)
            final_iteration = 2 if len(observations) == 1 else 3

        self.trace.append({"step": "final", "iteration": final_iteration, "content": answer})
        return {
            "answer": answer,
            "trace": self.trace,
            "iterations": final_iteration,
            "status": "completed",
        }

    def _run_tools_in_parallel(self, actions: List[Any]) -> List[str]:
        """Execute all selected business tools concurrently and preserve order."""
        self.trace.append(
            {
                "step": "parallel_tool_execution",
                "tools": [name for name, _ in actions],
            }
        )
        observations = [None] * len(actions)
        with ThreadPoolExecutor(max_workers=len(actions)) as executor:
            futures = {
                executor.submit(action): index
                for index, (_, action) in enumerate(actions)
            }
            for future in as_completed(futures):
                observations[futures[future]] = future.result()
        return observations

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


LoopingAgent = ToolCallingAgent

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
