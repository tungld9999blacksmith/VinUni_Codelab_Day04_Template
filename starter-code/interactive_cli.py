"""Interactive CLI for the ToolCallingAgent."""

import json
import sys

from template import LoopingAgent


HELP_TEXT = """
Nhập câu hỏi để gửi cho LoopingAgent.

Lệnh:
  :trace  Hiển thị trace của lần chạy gần nhất
  :json   Hiển thị đầy đủ kết quả dạng JSON
  :help   Hiển thị hướng dẫn này
  :quit   Thoát chương trình
"""


def main() -> None:
    agent = LoopingAgent()
    last_result = None

    print("=== VinAssistant Interactive CLI ===")
    print("Gõ :help để xem hướng dẫn, hoặc :quit để thoát.")

    while True:
        try:
            user_input = input("\nBạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTạm biệt!")
            return

        if not user_input:
            continue
        if user_input == ":quit":
            print("Tạm biệt!")
            return
        if user_input == ":help":
            print(HELP_TEXT)
            continue
        if user_input == ":trace":
            if last_result is None:
                print("Chưa có lần chạy nào.")
            else:
                print(json.dumps(last_result["trace"], ensure_ascii=False, indent=2))
            continue
        if user_input == ":json":
            if last_result is None:
                print("Chưa có lần chạy nào.")
            else:
                print(json.dumps(last_result, ensure_ascii=False, indent=2))
            continue

        try:
            last_result = agent.run(user_input)
        except Exception as exc:
            print(f"Lỗi khi xử lý yêu cầu: {exc}", file=sys.stderr)
            continue

        print(f"\nVinAssistant: {last_result['answer']}")
        print(
            f"[status={last_result['status']}, "
            f"iterations={last_result.get('iterations', 'n/a')}]"
        )


if __name__ == "__main__":
    main()
