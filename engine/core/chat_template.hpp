#pragma once
#include <string>
#include <vector>

namespace core {

struct ChatMessage {
    std::string role;
    std::string content;
};

// Render a Jinja2 (minimal subset) chat_template.
// - messages: the conversation turns.
// - add_generation_prompt: if true, the template's add_generation_prompt branch
//   appends the assistant prompt header.
// `tools` is always treated as an empty list (no tool-calling support), but the
// interpreter still evaluates any `tools` branches as empty/false so templates
// that reference `tools` (e.g. Qwen2.5) take the no-tools path without crashing.
std::string apply_chat_template(const std::string& tmpl,
                                const std::vector<ChatMessage>& messages,
                                bool add_generation_prompt);

} // namespace core
