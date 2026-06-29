#include "tokenizer.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>

namespace core {

Tokenizer::Tokenizer() {
    build_byte_tables_();
}

// Build the GPT-2 byte <-> unicode bijection.
//
// Bytes in [0x21,0x7E] U [0xA1,0xAC] U [0xAE,0xFF] map to chr(b) (identity).
// All other bytes, in ascending order, map to chr(256 + n) for a running n.
void Tokenizer::build_byte_tables_() {
    auto is_identity = [](unsigned char b) {
        return (b >= 0x21 && b <= 0x7E) ||
               (b >= 0xA1 && b <= 0xAC) ||
               (b >= 0xAE && b <= 0xFF);
    };

    int n = 0;
    for (int b = 0; b < 256; ++b) {
        int codepoint;
        if (is_identity(static_cast<unsigned char>(b))) {
            codepoint = b;
        } else {
            codepoint = 256 + n;
            ++n;
        }

        // UTF-8 encode the single codepoint into byte_to_unicode_[b].
        std::string& s = byte_to_unicode_[b];
        if (codepoint < 0x80) {
            s.push_back(static_cast<char>(codepoint));
        } else if (codepoint < 0x800) {
            s.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
            s.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        } else { // U+0800..U+FFFF (covers 256+n up to ~U+017F+range)
            s.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
            s.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
            s.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        }
        unicode_to_byte_[codepoint] = static_cast<unsigned char>(b);
    }
}

void Tokenizer::add_token(const std::string& token, int32_t id) {
    vocab_[token] = id;
    rev_vocab_[id] = token;
}

void Tokenizer::add_merge(const std::string& first, const std::string& second, int rank) {
    merge_ranks_[std::make_pair(first, second)] = rank;
}

void Tokenizer::add_special(const std::string& token, int32_t id) {
    specials_[token] = id;
    rev_vocab_[id] = token;
    max_special_len_ = std::max(max_special_len_, token.size());
}

// Decode a single unicode codepoint from a UTF-8 stream starting at `pos`.
// Returns the codepoint and advances pos past the character.
static int decode_utf8_char(const std::string& s, size_t& pos) {
    if (pos >= s.size()) return -1;
    unsigned char c0 = static_cast<unsigned char>(s[pos]);
    if (c0 < 0x80) {
        int cp = c0;
        pos += 1;
        return cp;
    } else if ((c0 & 0xE0) == 0xC0) {
        if (pos + 1 >= s.size()) { pos = s.size(); return c0; }
        unsigned char c1 = static_cast<unsigned char>(s[pos + 1]);
        int cp = ((c0 & 0x1F) << 6) | (c1 & 0x3F);
        pos += 2;
        return cp;
    } else if ((c0 & 0xF0) == 0xE0) {
        if (pos + 2 >= s.size()) { pos = s.size(); return c0; }
        unsigned char c1 = static_cast<unsigned char>(s[pos + 1]);
        unsigned char c2 = static_cast<unsigned char>(s[pos + 2]);
        int cp = ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
        pos += 3;
        return cp;
    } else if ((c0 & 0xF8) == 0xF0) {
        if (pos + 3 >= s.size()) { pos = s.size(); return c0; }
        unsigned char c1 = static_cast<unsigned char>(s[pos + 1]);
        unsigned char c2 = static_cast<unsigned char>(s[pos + 2]);
        unsigned char c3 = static_cast<unsigned char>(s[pos + 3]);
        int cp = ((c0 & 0x07) << 18) | ((c1 & 0x3F) << 12) |
                 ((c2 & 0x3F) << 6) | (c3 & 0x3F);
        pos += 4;
        return cp;
    }
    // Invalid lead byte; emit it raw.
    pos += 1;
    return c0;
}

// GPT-2 pre-tokenization using an ASCII hand-written scanner.
//std::regex does not support \p{L}, so we scan manually.
//
// Rules (left to right over raw bytes):
//   - A single leading space attaches to the following token (" ?" semantics).
//   - Runs of letters [A-Za-z], digits [0-9], or other non-space bytes.
//   - Trailing whitespace forms its own piece(s).
std::vector<std::string> Tokenizer::pretokenize_(const std::string& text) const {
    std::vector<std::string> out;
    const size_t n = text.size();
    size_t i = 0;

    auto is_letter = [](unsigned char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); };
    auto is_digit  = [](unsigned char c) { return (c >= '0' && c <= '9'); };
    auto is_space  = [](unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'; };

    while (i < n) {
        unsigned char c = static_cast<unsigned char>(text[i]);

        if (is_space(c)) {
            // Consume a run of whitespace. A single leading space attaches to
            // the following word; the rest (or trailing whitespace) is emitted
            // as whitespace pieces.
            size_t start = i;
            // Take the first space, then see what follows.
            // If the next non-consumed char starts a letter/digit/other run,
            // we attach this one leading space to it.
            if (c == ' ' && i + 1 < n) {
                unsigned char nxt = static_cast<unsigned char>(text[i + 1]);
                if (is_letter(nxt) || is_digit(nxt) || !is_space(nxt)) {
                    // Attach the single leading space to the following run.
                    size_t j = i + 1;
                    if (is_letter(nxt)) {
                        while (j < n && is_letter(static_cast<unsigned char>(text[j]))) ++j;
                    } else if (is_digit(nxt)) {
                        while (j < n && is_digit(static_cast<unsigned char>(text[j]))) ++j;
                    } else {
                        while (j < n) {
                            unsigned char cc = static_cast<unsigned char>(text[j]);
                            if (is_space(cc) || is_letter(cc) || is_digit(cc)) break;
                            ++j;
                        }
                    }
                    out.emplace_back(text.substr(start, j - start));
                    i = j;
                    continue;
                }
            }
            // Otherwise, this is pure trailing/inter-whitespace: consume the
            // whole run as one piece.
            size_t j = i;
            while (j < n && is_space(static_cast<unsigned char>(text[j]))) ++j;
            out.emplace_back(text.substr(start, j - start));
            i = j;
            continue;
        }

        // Non-space: consume a maximal run of the same class (no leading space).
        size_t start = i;
        if (is_letter(c)) {
            while (i < n && is_letter(static_cast<unsigned char>(text[i]))) ++i;
        } else if (is_digit(c)) {
            while (i < n && is_digit(static_cast<unsigned char>(text[i]))) ++i;
        } else {
            while (i < n) {
                unsigned char cc = static_cast<unsigned char>(text[i]);
                if (is_space(cc) || is_letter(cc) || is_digit(cc)) break;
                ++i;
            }
        }
        out.emplace_back(text.substr(start, i - start));
    }

    return out;
}

// Apply BPE to a byte-level-encoded word. `word` is the concatenation of
// single-char symbols. We merge the lowest-rank adjacent pair repeatedly.
std::vector<std::string> Tokenizer::bpe_(const std::string& word) const {
    // Split into single-character (single-symbol) pieces. Because byte-level
    // encoding produces only single-codepoint symbols, decode per character.
    std::vector<std::string> symbols;
    {
        size_t pos = 0;
        while (pos < word.size()) {
            size_t start = pos;
            decode_utf8_char(word, pos); // advances pos by one char
            symbols.emplace_back(word.substr(start, pos - start));
        }
    }

    if (symbols.size() < 2) return symbols;

    const int INF = std::numeric_limits<int>::max();

    while (symbols.size() > 1) {
        // Find the adjacent pair with the lowest merge rank.
        int best_rank = INF;
        size_t best_idx = 0;
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            auto it = merge_ranks_.find(std::make_pair(symbols[i], symbols[i + 1]));
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
            }
        }
        if (best_rank == INF) break;

        // Merge ALL non-overlapping left-to-right occurrences of that exact pair.
        const std::string& a = symbols[best_idx];
        const std::string& b = symbols[best_idx + 1];
        std::vector<std::string> merged;
        merged.reserve(symbols.size());
        for (size_t i = 0; i < symbols.size(); ) {
            if (i + 1 < symbols.size() && symbols[i] == a && symbols[i + 1] == b) {
                merged.push_back(a + b);
                i += 2;
            } else {
                merged.push_back(symbols[i]);
                i += 1;
            }
        }
        symbols = std::move(merged);
    }

    return symbols;
}

void Tokenizer::encode_pretoken_(const std::string& raw_piece,
                                 std::vector<int32_t>& out) const {
    if (raw_piece.empty()) return;

    // Map each raw byte through byte_to_unicode to get the encoded word.
    std::string word;
    word.reserve(raw_piece.size() * 2);
    for (unsigned char b : raw_piece) {
        word += byte_to_unicode_[b];
    }

    // BPE-merge in byte-level-encoded space.
    std::vector<std::string> symbols = bpe_(word);

    for (const std::string& sym : symbols) {
        auto it = vocab_.find(sym);
        if (it != vocab_.end()) {
            out.push_back(it->second);
            continue;
        }
        // Fallback: emit each individual byte-level single-char token.
        size_t pos = 0;
        while (pos < sym.size()) {
            size_t start = pos;
            decode_utf8_char(sym, pos);
            std::string single = sym.substr(start, pos - start);
            auto bit = vocab_.find(single);
            if (bit != vocab_.end()) {
                out.push_back(bit->second);
            }
            // If even the single byte token is missing, skip (no crash).
        }
    }
}

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    std::vector<int32_t> ids;
    if (text.empty()) return ids;

    const size_t n = text.size();

    // 1. Split on special tokens with longest-match, left-to-right.
    //    Segments between specials are BPE-encoded.
    size_t i = 0;
    std::string buf; // accumulates non-special text between specials

    auto flush_buf = [&]() {
        if (buf.empty()) return;
        std::vector<std::string> pieces = pretokenize_(buf);
        for (const std::string& p : pieces) {
            encode_pretoken_(p, ids);
        }
        buf.clear();
    };

    while (i < n) {
        // Try to match a special token starting at i (longest first).
        bool matched = false;
        if (max_special_len_ > 0 && n - i >= 1) {
            size_t max_len = std::min(max_special_len_, n - i);
            for (size_t len = max_len; len >= 1; --len) {
                std::string cand = text.substr(i, len);
                auto sit = specials_.find(cand);
                if (sit != specials_.end()) {
                    flush_buf();
                    ids.push_back(sit->second);
                    i += len;
                    matched = true;
                    break;
                }
            }
        }
        if (matched) continue;

        buf.push_back(text[i]);
        ++i;
    }
    flush_buf();

    return ids;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    // First, reconstruct the byte-level-encoded string by concatenating
    // each id's vocab string.
    std::string encoded;
    for (int32_t id : ids) {
        auto it = rev_vocab_.find(id);
        if (it != rev_vocab_.end()) {
            encoded += it->second;
        }
    }

    // Walk the encoded string codepoint by codepoint; map each codepoint back
    // to its raw byte via the inverse byte_to_unicode table.
    std::string out;
    size_t pos = 0;
    while (pos < encoded.size()) {
        int cp = decode_utf8_char(encoded, pos);
        if (cp < 0) break;
        auto uit = unicode_to_byte_.find(cp);
        if (uit != unicode_to_byte_.end()) {
            out.push_back(static_cast<char>(uit->second));
        } else {
            // Codepoint not in the remap table (identity-mapped byte or stray):
            // re-emit its UTF-8 bytes so special tokens / multibyte vocab
            // strings still round-trip.
            if (cp < 0x80) {
                out.push_back(static_cast<char>(cp));
            } else if (cp < 0x800) {
                out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            } else {
                out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            }
        }
    }
    return out;
}

} // namespace core
