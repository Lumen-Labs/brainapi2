import unittest

from src.services.mcp.prompt import build_mcp_tools_instructions


class ChatbotMcpPromptTests(unittest.TestCase):
    def test_build_mcp_tools_instructions_includes_brain_id(self):
        text = build_mcp_tools_instructions("my-brain")
        self.assertIn("my-brain", text)
        self.assertIn("search_semantically", text)
        self.assertIn("traverse_graph", text)
        self.assertIn("tool_name", text)

    def test_prepend_wraps_user_prompt(self):
        user_prompt = "User question"
        combined = f"{build_mcp_tools_instructions('brain-a')}\n\n{user_prompt}"
        self.assertTrue(combined.endswith(user_prompt))
        self.assertIn("brain-a", combined)


if __name__ == "__main__":
    unittest.main()
