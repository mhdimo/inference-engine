// Standalone test for the minimal Jinja2 chat-template renderer.
// Build with:
//   c++ -std=c++17 -Iengine tests/test_chat_template.cpp engine/core/chat_template.cpp -o /tmp/ct_test && /tmp/ct_test
#include "core/chat_template.hpp"

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace core;

static int g_failures = 0;

// Helper to escape non-printable chars for readable failure messages.
static std::string esc(const std::string& s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            case '\r': out += "\\r"; break;
            default: out += c;
        }
    }
    return out;
}

static void check(const std::string& label,
                  const std::string& actual,
                  const std::string& expected) {
    if (actual == expected) {
        std::cout << "[PASS] " << label << "\n";
    } else {
        ++g_failures;
        std::cout << "[FAIL] " << label << "\n";
        std::cout << "  expected (" << expected.size() << " bytes): \"" << esc(expected) << "\"\n";
        std::cout << "  actual   (" << actual.size() << " bytes): \"" << esc(actual) << "\"\n";
        // First-difference diagnostic.
        size_t i = 0;
        while (i < expected.size() && i < actual.size() && expected[i] == actual[i]) ++i;
        if (i < expected.size() || i < actual.size()) {
            std::cout << "  first diff at byte " << i << "\n";
        }
    }
}

int main() {
    std::cout << "==========================================================\n";
    std::cout << "Chat Template Renderer Test\n";
    std::cout << "==========================================================\n";

    // --- SmolLM2 template ---------------------------------------------------
    const std::string smollm =
        "{% for message in messages %}"
        "{% if loop.first and messages[0]['role'] != 'system' %}"
        "{{ '<|im_start|>system\\nYou are a helpful AI assistant named SmolLM, "
        "trained by Hugging Face<|im_end|>\\n' }}"
        "{% endif %}"
        "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

    // --- Qwen2.5 template ---------------------------------------------------
    // Note: \\n inside C++ string literals become \n (single newline) in the
    // actual template string, matching the real Jinja template.
    const std::string qwen =
        "{%- if tools %}\n"
        " {{- '<|im_start|>system\\n' }}\n"
        " {%- if messages[0]['role'] == 'system' %}\n"
        " {{- messages[0]['content'] }}\n"
        " {%- else %}\n"
        " {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n"
        " {%- endif %}\n"
        " {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the "
        "user query.\\n\\nYou are provided with function signatures within "
        "<tools></tools> XML tags:\\n<tools>\" }}\n"
        " {%- for tool in tools %}\n"
        " {{- \"\\n\" }}\n"
        " {{- tool | tojson }}\n"
        " {%- endfor %}\n"
        " {{- \"\\n</tools>\\n\\nFor each function call, return a json object with "
        "function name and arguments within <tool_call>\\n</tool_call> XML tags:\\n"
        "<tool_call>\\n{\\\"name\\\": <name>, \\\"arguments\\\": <args>}\\n"
        "</tool_call><|im_end|>\\n\" }}\n"
        "{%- else %}\n"
        " {%- if messages[0]['role'] == 'system' %}\n"
        " {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n"
        " {%- else %}\n"
        " {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a "
        "helpful assistant.<|im_end|>\\n' }}\n"
        " {%- endif %}\n"
        "{%- endif %}\n"
        "{%- for message in messages %}\n"
        " {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) "
        "or (message.role == \"assistant\" and not message.tool_calls) %}\n"
        " {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n"
        " {%- elif message.role == \"assistant\" %}\n"
        " {{- '<|im_start|>' + message.role }}\n"
        " {%- if message.content %}\n"
        " {{- '\\n' + message.content }}\n"
        " {%- endif %}\n"
        " {%- for tool_call in message.tool_calls %}\n"
        " {%- if tool_call.function is defined %}\n"
        " {%- set tool_call = tool_call.function %}\n"
        " {%- endif %}\n"
        " {{- '\\n\\n{\\\"name\\\": \\\"' }}\n"
        " {{- tool_call.name }}\n"
        " {{- '\\\", \\\"arguments\\\": ' }}\n"
        " {{- tool_call.arguments | tojson }}\n"
        " {{- '}\\n' }}\n"
        " {%- endfor %}\n"
        " {{- '<|im_end|>\\n' }}\n"
        " {%- elif message.role == \"tool\" %}\n"
        " {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n"
        " {{- '<|im_start|>user' }}\n"
        " {%- endif %}\n"
        " {{- '\\n\\n' + message.content }}\n"
        " {{- '\\n' }}\n"
        " {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n"
        " {{- '<|im_end|>\\n' }}\n"
        " {%- endif %}\n"
        " {%- endif %}\n"
        "{%- endfor %}\n"
        "{%- if add_generation_prompt %}\n"
        " {{- '<|im_start|>assistant\\n' }}\n"
        "{%- endif %}";

    // ----------------------------------------------------------------------
    // Test 1: SmolLM2, single user message, add_generation_prompt=true
    // ----------------------------------------------------------------------
    {
        std::vector<ChatMessage> msgs = {{"user", "hi"}};
        std::string out = apply_chat_template(smollm, msgs, true);
        std::string expected =
            "<|im_start|>system\n"
            "You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n"
            "<|im_start|>user\n"
            "hi<|im_end|>\n"
            "<|im_start|>assistant\n";
        check("SmolLM2 single user + gen prompt", out, expected);
    }

    // ----------------------------------------------------------------------
    // Test 2: Qwen2.5, single user message, add_generation_prompt=true
    // ----------------------------------------------------------------------
    {
        std::vector<ChatMessage> msgs = {{"user", "hi"}};
        std::string out = apply_chat_template(qwen, msgs, true);
        std::string expected =
            "<|im_start|>system\n"
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "hi<|im_end|>\n"
            "<|im_start|>assistant\n";
        check("Qwen2.5 single user + gen prompt", out, expected);
    }

    // ----------------------------------------------------------------------
    // Test 3: SmolLM2 multi-turn (user, assistant, user) + gen prompt
    // ----------------------------------------------------------------------
    {
        std::vector<ChatMessage> msgs = {
            {"user", "Hello"},
            {"assistant", "Hi there! How can I help?"},
            {"user", "Write a haiku"},
        };
        std::string out = apply_chat_template(smollm, msgs, true);
        std::string expected =
            "<|im_start|>system\n"
            "You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n"
            "<|im_start|>user\n"
            "Hello<|im_end|>\n"
            "<|im_start|>assistant\n"
            "Hi there! How can I help?<|im_end|>\n"
            "<|im_start|>user\n"
            "Write a haiku<|im_end|>\n"
            "<|im_start|>assistant\n";
        check("SmolLM2 multi-turn + gen prompt", out, expected);
    }

    // ----------------------------------------------------------------------
    // Test 4: SmolLM2 single user, add_generation_prompt=false (no trailer)
    // ----------------------------------------------------------------------
    {
        std::vector<ChatMessage> msgs = {{"user", "hi"}};
        std::string out = apply_chat_template(smollm, msgs, false);
        std::string expected =
            "<|im_start|>system\n"
            "You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n"
            "<|im_start|>user\n"
            "hi<|im_end|>\n";
        check("SmolLM2 single user, no gen prompt", out, expected);
    }

    // ----------------------------------------------------------------------
    // Extra: SmolLM2 with an explicit system message (no synthetic header)
    // ----------------------------------------------------------------------
    {
        std::vector<ChatMessage> msgs = {
            {"system", "Be brief."},
            {"user", "ping"},
        };
        std::string out = apply_chat_template(smollm, msgs, false);
        std::string expected =
            "<|im_start|>system\n"
            "Be brief.<|im_end|>\n"
            "<|im_start|>user\n"
            "ping<|im_end|>\n";
        check("SmolLM2 explicit system message", out, expected);
    }

    std::cout << "==========================================================\n";
    if (g_failures == 0) {
        std::cout << "[SUCCESS] All chat-template tests passed.\n";
    } else {
        std::cout << "[FAILURE] " << g_failures << " test(s) failed.\n";
    }
    std::cout << "==========================================================\n";
    return g_failures == 0 ? 0 : 1;
}
