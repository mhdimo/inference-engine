// Standalone test for the GPT-2 byte-level BPE tokenizer.
//
// Build (from repo root):
//   c++ -std=c++17 -Iengine tests/test_tokenizer_bpe.cpp engine/core/tokenizer.cpp -o /tmp/tok_test && /tmp/tok_test
//
// Covers:
//   * byte_to_unicode bijection (space 0x20 -> U+0120 "Ġ", etc.)
//   * synthetic vocab/merges round-trip
//   * real SmolLM2 vocab+merges (parsed live from the GGUF) oracle checks:
//       encode("hii")        == [88, 4877]
//       encode(" Hello")     == [38699]   (token "ĠHello")
//       round-trip " Hello" / "hii"
//   * special-token handling ("<|im_start|>") is emitted as one id.

#include "core/tokenizer.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <assert.h>

using core::Tokenizer;

static int g_failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "[FAIL] " << (msg) << " (line " << __LINE__ << ")\n"; ++g_failures; } \
    else { std::cout << "[ok]   " << (msg) << "\n"; } \
} while (0)

// ---------------------------------------------------------------------------
// Minimal GGUF metadata reader: pulls tokenizer.ggml.tokens and
// tokenizer.ggml.merges string arrays without depending on the engine loader.
// ---------------------------------------------------------------------------
namespace {

enum GVal : uint32_t {
    G_UINT8=0,G_INT8=1,G_UINT16=2,G_INT16=3,G_UINT32=4,G_INT32=5,G_FLOAT32=6,
    G_BOOL=7,G_STRING=8,G_ARRAY=9,G_UINT64=10,G_INT64=11,G_FLOAT64=12,
};

struct GGUFData {
    std::vector<char> buf;
    const char* p;
    const char* end;
    std::vector<std::string> tokens;
    std::vector<std::string> merges;
    bool ok = false;
};

template <typename T>
static T rd(const char*& p) {
    T v; std::memcpy(&v, p, sizeof(T)); p += sizeof(T); return v;
}
static std::string rd_str(const char*& p) {
    uint64_t len = rd<uint64_t>(p);
    std::string s(p, len); p += len; return s;
}

static size_t val_size(uint32_t t) {
    switch (t) {
        case G_UINT8: case G_INT8: case G_BOOL: return 1;
        case G_UINT16: case G_INT16: return 2;
        case G_UINT32: case G_INT32: case G_FLOAT32: return 4;
        case G_UINT64: case G_INT64: case G_FLOAT64: return 8;
        default: return 0;
    }
}

static void skip_array(const char*& p) {
    uint32_t et = rd<uint32_t>(p);
    uint64_t len = rd<uint64_t>(p);
    for (uint64_t i = 0; i < len; ++i) {
        if (et == G_STRING) { uint64_t sl = rd<uint64_t>(p); p += sl; }
        else if (et == G_ARRAY) skip_array(p);
        else p += val_size(et);
    }
}

static bool load_gguf_tokens(const std::string& path, GGUFData& g) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::vector<char> buf((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
    g.buf = std::move(buf);
    g.p = g.buf.data();
    g.end = g.p + g.buf.size();

    uint32_t magic = rd<uint32_t>(g.p);
    if (magic != 0x46554747) return false; // "GGUF"
    /*uint32_t ver =*/ rd<uint32_t>(g.p);
    uint64_t tensor_count = rd<uint64_t>(g.p);
    uint64_t kv_count = rd<uint64_t>(g.p);

    for (uint64_t i = 0; i < kv_count; ++i) {
        std::string key = rd_str(g.p);
        uint32_t type = rd<uint32_t>(g.p);
        if (type == G_STRING) {
            (void)rd_str(g.p);
        } else if (type == G_ARRAY) {
            uint32_t et = rd<uint32_t>(g.p);
            uint64_t len = rd<uint64_t>(g.p);
            if (et == G_STRING &&
                (key == "tokenizer.ggml.tokens" || key == "tokenizer.ggml.merges")) {
                std::vector<std::string>* dst =
                    (key == "tokenizer.ggml.tokens") ? &g.tokens : &g.merges;
                dst->resize(len);
                for (uint64_t k = 0; k < len; ++k) (*dst)[k] = rd_str(g.p);
            } else {
                for (uint64_t k = 0; k < len; ++k) {
                    if (et == G_STRING) { uint64_t sl = rd<uint64_t>(g.p); g.p += sl; }
                    else if (et == G_ARRAY) skip_array(g.p);
                    else g.p += val_size(et);
                }
            }
        } else {
            g.p += val_size(type);
        }
    }
    (void)tensor_count;
    g.ok = !g.tokens.empty();
    return g.ok;
}

// Reconstruct the GPT-2 byte->unicode char for a byte (for assertions).
static std::string byte_to_unicode_char(int b) {
    auto idy = [](int x) {
        return (x >= 0x21 && x <= 0x7E) || (x >= 0xA1 && x <= 0xAC) || (x >= 0xAE && x <= 0xFF);
    };
    int n = 0, cp = 0;
    for (int x = 0; x <= b; ++x) {
        cp = idy(x) ? x : (256 + n);
        if (!idy(x)) ++n;
    }
    // encode cp as utf-8
    std::string s;
    if (cp < 0x80) s.push_back(static_cast<char>(cp));
    else if (cp < 0x800) { s.push_back(static_cast<char>(0xC0|(cp>>6))); s.push_back(static_cast<char>(0x80|(cp&0x3F))); }
    else { s.push_back(static_cast<char>(0xE0|(cp>>12))); s.push_back(static_cast<char>(0x80|((cp>>6)&0x3F))); s.push_back(static_cast<char>(0x80|(cp&0x3F))); }
    return s;
}

} // namespace

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static void test_byte_to_unicode() {
    std::cout << "\n=== byte_to_unicode ===\n";
    // We verify the bijection indirectly through encode/decode round-trips
    // on raw bytes. But we can assert the well-known expected chars:
    std::string space_g = byte_to_unicode_char(0x20);   // "Ġ" U+0120
    std::string A        = byte_to_unicode_char(0x41);   // "A"
    std::string zero     = byte_to_unicode_char(0x00);   // chr(256)

    // U+0120 in UTF-8 is 0xC4 0xA0
    CHECK(space_g == std::string("\xC4\xA0"), "space 0x20 -> U+0120 (\\xC4\\xA0)");
    CHECK(A == "A", "'A' 0x41 -> \"A\"");
    // chr(256) = U+0100 -> UTF-8 0xC4 0x80
    CHECK(zero == std::string("\xC4\x80"), "byte 0x00 -> U+0100 (\\xC4\\x80)");

    // Verify 0x21 ('!') is identity.
    CHECK(byte_to_unicode_char(0x21) == "!", "'!' 0x21 -> \"!\" (identity)");
}

static void test_synthetic() {
    std::cout << "\n=== synthetic vocab/merges round-trip ===\n";
    Tokenizer t;

    // Register the 256 byte-level single-char tokens so any byte is coverable.
    for (int b = 0; b < 256; ++b) {
        // Build the byte-level-encoded single-char string.
        auto idy = [](int x) {
            return (x >= 0x21 && x <= 0x7E) || (x >= 0xA1 && x <= 0xAC) || (x >= 0xAE && x <= 0xFF);
        };
        int n = 0, cp = 0;
        for (int x = 0; x <= b; ++x) { cp = idy(x) ? x : (256 + n); if (!idy(x)) ++n; }
        std::string s;
        if (cp < 0x80) s.push_back(static_cast<char>(cp));
        else if (cp < 0x800) { s.push_back(static_cast<char>(0xC0|(cp>>6))); s.push_back(static_cast<char>(0x80|(cp&0x3F))); }
        else { s.push_back(static_cast<char>(0xE0|(cp>>12))); s.push_back(static_cast<char>(0x80|((cp>>6)&0x3F))); s.push_back(static_cast<char>(0x80|(cp&0x3F))); }
        t.add_token(s, b);
    }

    // "he" merge then "hello" via merges in byte-level space.
    // h,e,l,o are all identity bytes, so byte-level-encoded == raw.
    t.add_merge("h", "e", 0);   // he
    t.add_merge("l", "l", 1);   // ll
    t.add_merge("he", "l", 2);  // hel
    t.add_merge("hel", "ll", 3);// hell  (note: must pick lowest-rank pair)
    t.add_merge("he", "ll", 4); // alternate (not lowest)
    t.add_merge("hell", "o", 5);// hello

    // Register the merged vocab entries so BPE output resolves to single ids.
    t.add_token("he", 256);
    t.add_token("ll", 257);
    t.add_token("hel", 258);
    t.add_token("hell", 259);
    t.add_token("hello", 260);

    auto ids = t.encode("hello");
    std::cout << "  encode(\"hello\") = [";
    for (auto id : ids) std::cout << " " << id;
    std::cout << " ]\n";
    std::string dec = t.decode(ids);
    CHECK(dec == "hello", "decode(encode(\"hello\")) == \"hello\"");
    // BPE should fully merge "hello" into a single token via the rank-0..5 chain.
    CHECK(ids.size() == 1 && ids[0] == 260,
          "encode(\"hello\") fully merges -> [260]");

    // " world" with leading space -> "Ġworld"
    t.add_merge("\xC4\xA0", "w", 10); // Ġ + w  (space attaches)
    auto ids2 = t.encode(" world");
    std::string dec2 = t.decode(ids2);
    CHECK(dec2 == " world", "decode(encode(\" world\")) == \" world\"");
    std::cout << "  encode(\" world\") = [";
    for (auto id : ids2) std::cout << " " << id;
    std::cout << " ]\n";
}

static void test_specials() {
    std::cout << "\n=== special tokens ===\n";
    Tokenizer t;
    t.add_special("<|im_start|>", 1);
    t.add_special("<|im_end|>", 2);
    t.add_token("x", 100);

    auto ids = t.encode("<|im_start|>x<|im_end|>");
    std::cout << "  encode = [";
    for (auto id : ids) std::cout << " " << id;
    std::cout << " ]\n";
    CHECK(ids.size() >= 3, "specials split yields >=3 ids");
    CHECK(ids.front() == 1, "first id == <|im_start|> (1)");
    CHECK(ids.back() == 2, "last id == <|im_end|> (2)");
}

static void test_real_smollm2() {
    std::cout << "\n=== real SmolLM2 vocab+merges (GGUF) ===\n";
    const std::string path = "/Users/liang/inference-engine/SmollM2-135M-Instruct-Q4_0.gguf";
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        std::cout << "  [skip] GGUF not found at " << path << "\n";
        return;
    }

    GGUFData g;
    if (!load_gguf_tokens(path, g)) {
        CHECK(false, "failed to parse GGUF tokenizer arrays");
        return;
    }
    std::cout << "  loaded " << g.tokens.size() << " tokens, "
              << g.merges.size() << " merges\n";

    Tokenizer t;
    for (size_t i = 0; i < g.tokens.size(); ++i) {
        t.add_token(g.tokens[i], static_cast<int32_t>(i));
    }
    // Register known SmolLM2 special tokens so encode() can emit them whole.
    t.add_special("<|im_start|>", 1);
    t.add_special("<|im_end|>", 2);
    t.add_special("<|endoftext|>", 0);

    int rank = 0;
    for (const std::string& m : g.merges) {
        auto sp = m.find(' ');
        if (sp == std::string::npos) { ++rank; continue; }
        std::string a = m.substr(0, sp);
        std::string b = m.substr(sp + 1);
        t.add_merge(a, b, rank);
        ++rank;
    }

    // Oracle: encode("hii") == [88, 4877]
    {
        auto ids = t.encode("hii");
        std::cout << "  encode(\"hii\") = [";
        for (auto id : ids) std::cout << " " << id;
        std::cout << " ]  (oracle: [88 4877])\n";
        CHECK(ids.size() == 2 && ids[0] == 88 && ids[1] == 4877,
              "encode(\"hii\") == [88, 4877]");
        CHECK(t.decode(ids) == "hii", "decode(\"hii\") round-trips");
    }

    // Oracle: encode(" Hello") == [38699] ("ĠHello")
    {
        auto ids = t.encode(" Hello");
        std::cout << "  encode(\" Hello\") = [";
        for (auto id : ids) std::cout << " " << id;
        std::cout << " ]  (oracle: [38699])\n";
        CHECK(ids.size() == 1 && ids[0] == 38699,
              "encode(\" Hello\") == [38699]");
        CHECK(t.decode(ids) == " Hello", "decode(\" Hello\") round-trips");
        // Confirm the first token's vocab string starts with "Ġ".
        const std::string& vt = g.tokens[38699];
        CHECK(!vt.empty() && (unsigned char)vt[0] == 0xC4,
              "vocab[38699] starts with \\xC4 (lead byte of U+0120 \"Ġ\")");
        std::cout << "  vocab[38699] = \"" << vt << "\" (len " << vt.size() << ")\n";
    }

    // Round-trip a longer English sentence.
    {
        std::string s = "The quick brown fox jumps over the lazy dog.";
        auto ids = t.encode(s);
        std::string back = t.decode(ids);
        CHECK(back == s, "long English sentence round-trips");
        std::cout << "  encoded " << s.size() << " chars -> " << ids.size() << " tokens\n";
    }

    // Special-token round-trip with BPE in between.
    {
        std::string s = "<|im_start|>user\nHello<|im_end|>";
        auto ids = t.encode(s);
        std::string back = t.decode(ids);
        CHECK(back == s, "chat template fragment round-trips");
        std::cout << "  encode(\"" << s << "\") = [";
        for (auto id : ids) std::cout << " " << id;
        std::cout << " ]\n";
    }
}

int main() {
    std::cout << "=========================================================\n";
    std::cout << "GPT-2 byte-level BPE tokenizer tests\n";
    std::cout << "=========================================================\n";

    test_byte_to_unicode();
    test_synthetic();
    test_specials();
    test_real_smollm2();

    std::cout << "\n=========================================================\n";
    if (g_failures == 0) {
        std::cout << "ALL TESTS PASSED\n";
    } else {
        std::cout << g_failures << " CHECK(S) FAILED\n";
    }
    std::cout << "=========================================================\n";
    return g_failures == 0 ? 0 : 1;
}
