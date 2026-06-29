#include "engine/core/types.hpp"
#include "engine/core/logger.hpp"
#include "engine/core/tokenizer.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <cassert>

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "Byte-Pair Encoding (BPE) Tokenizer Test (Phase 8)" << std::endl;
    std::cout << "==========================================================" << std::endl;

    using namespace core;

    Tokenizer tokenizer;

    // 1. Register 256 basic byte tokens (ensures complete vocabulary coverage)
    for (int i = 0; i < 256; ++i) {
        tokenizer.add_token(std::string(1, static_cast<char>(i)), i);
    }

    // 2. Add subwords and words vocab mapping
    tokenizer.add_token("ro", 256);
    tokenizer.add_token("bot", 257);
    tokenizer.add_token("robot", 258);
    tokenizer.add_token(" learns", 259);
    tokenizer.add_token(" to", 260);
    tokenizer.add_token(" code", 261);

    // 3. Add merge priorities to compile letters into words
    // Rank index values (lower ranks execute merges first)
    tokenizer.add_merge("r", "o", 0);
    tokenizer.add_merge("b", "o", 1);
    tokenizer.add_merge("bo", "t", 2);
    tokenizer.add_merge("ro", "bot", 3);
    
    // merges for " learns"
    tokenizer.add_merge(" ", "l", 4);
    tokenizer.add_merge(" l", "e", 5);
    tokenizer.add_merge(" le", "a", 6);
    tokenizer.add_merge(" lea", "r", 7);
    tokenizer.add_merge(" lear", "n", 8);
    tokenizer.add_merge(" learn", "s", 9);

    // merges for " to"
    tokenizer.add_merge(" ", "t", 10);
    tokenizer.add_merge(" t", "o", 11);

    // merges for " code"
    tokenizer.add_merge(" ", "c", 12);
    tokenizer.add_merge(" c", "o", 13);
    tokenizer.add_merge(" co", "d", 14);
    tokenizer.add_merge(" cod", "e", 15);

    // 4. Test String Encoding
    std::string prompt = "robot learns to code";
    std::cout << "[TEST] Input prompt string: \"" << prompt << "\"" << std::endl;

    std::vector<int32_t> tokens = tokenizer.encode(prompt);
    std::cout << "[TEST] Encoded Token IDs:   [ ";
    for (int32_t t : tokens) {
        std::cout << t << " ";
    }
    std::cout << "]" << std::endl;

    // We expect: [258, 259, 260, 261]
    // 258: "robot", 259: " learns", 260: " to", 261: " code"
    assert(tokens.size() == 4);
    assert(tokens[0] == 258);
    assert(tokens[1] == 259);
    assert(tokens[2] == 260);
    assert(tokens[3] == 261);
    std::cout << "[SUCCESS] Prompt successfully encoded to expected word tokens!" << std::endl;

    // 5. Test Token Decoding
    std::string decoded = tokenizer.decode(tokens);
    std::cout << "[TEST] Decoded output text: \"" << decoded << "\"" << std::endl;
    assert(decoded == prompt);
    std::cout << "[SUCCESS] Decoded prompt string matches input perfectly!" << std::endl;

    // 6. Test fallback (byte-level parsing of unregistered vocabulary characters)
    std::string mixed_prompt = "robot learns to code! #xyz";
    std::cout << "[TEST] Input mixed prompt:  \"" << mixed_prompt << "\"" << std::endl;
    std::vector<int32_t> mixed_tokens = tokenizer.encode(mixed_prompt);
    std::string mixed_decoded = tokenizer.decode(mixed_tokens);
    std::cout << "[TEST] Decoded mixed text:  \"" << mixed_decoded << "\"" << std::endl;
    assert(mixed_decoded == mixed_prompt);
    std::cout << "[SUCCESS] Mixed byte-level fallback verified successfully!" << std::endl;

    std::cout << "==========================================================" << std::endl;
    std::cout << "[SUCCESS] BPE Tokenizer integration test PASSED!" << std::endl;
    std::cout << "==========================================================" << std::endl;

    return 0;
}
