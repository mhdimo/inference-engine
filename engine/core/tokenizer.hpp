#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

namespace core {

// GPT-2 style byte-level BPE tokenizer.
//
// Vocabulary keys and BPE merge keys are stored in *byte-level-encoded*
// space: each raw byte of the source text is first mapped through
// byte_to_unicode() (e.g. ASCII space 0x20 -> U+0120 "Ġ") before being
// used as a symbol. This matches the format used by GPT-2 / SmolLM2
// tokenizers stored in GGUF files (tokenizer.ggml.tokens / .merges).
class Tokenizer {
public:
    Tokenizer();

    // Register a vocabulary entry: the byte-level-encoded token string -> id.
    void add_token(const std::string& token, int32_t id);

    // Register a BPE merge priority rank. `first` and `second` are
    // byte-level-encoded symbol strings; lower rank merges first.
    void add_merge(const std::string& first, const std::string& second, int rank);

    // Register a special token (e.g. "<|im_start|>") that must be emitted
    // as a single id and never participates in BPE merging.
    void add_special(const std::string& token, int32_t id);

    // Encode raw text to token IDs using GPT-2 byte-level BPE.
    std::vector<int32_t> encode(const std::string& text) const;

    // Decode token IDs back to the original text (inverting byte_to_unicode).
    std::string decode(const std::vector<int32_t>& ids) const;

private:
    std::unordered_map<std::string, int32_t> vocab_;
    std::unordered_map<int32_t, std::string> rev_vocab_;

    // Special tokens that short-circuit BPE (longest-match, left-to-right).
    std::unordered_map<std::string, int32_t> specials_;
    size_t max_special_len_ = 0;

    struct pair_hash {
        size_t operator()(const std::pair<std::string, std::string>& p) const {
            return std::hash<std::string>{}(p.first) ^ (std::hash<std::string>{}(p.second) << 1);
        }
    };
    std::unordered_map<std::pair<std::string, std::string>, int, pair_hash> merge_ranks_;

    // ---- byte_to_unicode machinery ----
    // Maps a raw byte value (0..255) to a unicode codepoint (as a UTF-8 string
    // holding a single character). Built once at construction.
    std::string byte_to_unicode_[256];
    // Inverse map: unicode codepoint (>=0x100 for the remapped bytes) -> byte.
    std::unordered_map<int, unsigned char> unicode_to_byte_;

    void build_byte_tables_();

    // GPT-2 pre-tokenization (ASCII approximation). Splits a segment into
    // raw-text pieces; each piece is later byte-level-encoded for BPE.
    std::vector<std::string> pretokenize_(const std::string& text) const;

    // Apply BPE merges to a byte-level-encoded word; returns final symbols.
    std::vector<std::string> bpe_(const std::string& word) const;

    // Encode a single non-special pre-token to ids (with byte fallback).
    void encode_pretoken_(const std::string& raw_piece,
                          std::vector<int32_t>& out) const;
};

} // namespace core
