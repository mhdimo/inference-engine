#include "chat_template.hpp"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace core {
namespace {

// ---------------------------------------------------------------------------
// Runtime value type
// ---------------------------------------------------------------------------
//
// A Value is one of: bool, int64, string, list, map, or "undefined".
// Attribute/subscript access on a missing key returns undefined so that
// `X is defined` and truthiness behave per Jinja semantics.

class Value;
using List = std::vector<Value>;
using Map = std::map<std::string, Value>; // ordered for deterministic JSON

class Value {
public:
    enum class Kind { Null, Bool, Int, String, List, Map, Undefined };

    Value() : kind_(Kind::Null) {}

    static Value makeUndefined() { Value v; v.kind_ = Kind::Undefined; return v; }
    static Value makeBool(bool b) { Value v; v.kind_ = Kind::Bool; v.b_ = b; return v; }
    static Value makeInt(std::int64_t i) { Value v; v.kind_ = Kind::Int; v.i_ = i; return v; }
    static Value makeString(std::string s) { Value v; v.kind_ = Kind::String; v.s_ = std::move(s); return v; }
    static Value makeList(List l) { Value v; v.kind_ = Kind::List; v.list_ = std::make_shared<List>(std::move(l)); return v; }
    static Value makeMap(Map m) { Value v; v.kind_ = Kind::Map; v.map_ = std::make_shared<Map>(std::move(m)); return v; }

    Kind kind() const { return kind_; }
    bool asBool() const { return b_; }
    std::int64_t asInt() const { return i_; }
    const std::string& asString() const { return s_; }
    const List& asList() const { return *list_; }
    const Map& asMap() const { return *map_; }

    bool truthy() const {
        switch (kind_) {
            case Kind::Undefined:
            case Kind::Null: return false;
            case Kind::Bool: return b_;
            case Kind::Int: return i_ != 0;
            case Kind::String: return !s_.empty();
            case Kind::List: return list_ && !list_->empty();
            case Kind::Map: return map_ && !map_->empty();
        }
        return false;
    }

    // `X is defined`: true iff the value is not undefined/null.
    bool defined() const { return kind_ != Kind::Undefined && kind_ != Kind::Null; }

    Value get(const std::string& key) const {
        if (kind_ == Kind::Map) {
            auto it = map_->find(key);
            if (it != map_->end()) return it->second;
        }
        return makeUndefined();
    }

private:
    Kind kind_ = Kind::Null;
    bool b_ = false;
    std::int64_t i_ = 0;
    std::string s_;
    std::shared_ptr<List> list_;
    std::shared_ptr<Map> map_;
};

// ---------------------------------------------------------------------------
// JSON serialization (for the `tojson` filter)
// ---------------------------------------------------------------------------

void toJson(const Value& v, std::string& out);

std::string toStr(const Value& v) {
    switch (v.kind()) {
        case Value::Kind::String: return v.asString();
        case Value::Kind::Int: return std::to_string(v.asInt());
        case Value::Kind::Bool: return v.asBool() ? "true" : "false";
        case Value::Kind::Undefined:
        case Value::Kind::Null: return "";
        default: {
            std::string s;
            toJson(v, s);
            return s;
        }
    }
}

void toJson(const Value& v, std::string& out) {
    switch (v.kind()) {
        case Value::Kind::Undefined:
        case Value::Kind::Null:
            out += "null";
            break;
        case Value::Kind::Bool:
            out += v.asBool() ? "true" : "false";
            break;
        case Value::Kind::Int:
            out += std::to_string(v.asInt());
            break;
        case Value::Kind::String: {
            const std::string& s = v.asString();
            out += '"';
            for (char c : s) {
                switch (c) {
                    case '"':  out += "\\\""; break;
                    case '\\': out += "\\\\"; break;
                    case '\n': out += "\\n"; break;
                    case '\r': out += "\\r"; break;
                    case '\t': out += "\\t"; break;
                    case '\b': out += "\\b"; break;
                    case '\f': out += "\\f"; break;
                    default:
                        if (static_cast<unsigned char>(c) < 0x20) {
                            char buf[8];
                            std::snprintf(buf, sizeof(buf), "\\u%04x",
                                          static_cast<unsigned char>(c));
                            out += buf;
                        } else {
                            out += c;
                        }
                }
            }
            out += '"';
            break;
        }
        case Value::Kind::List: {
            out += '[';
            bool first = true;
            for (const Value& e : v.asList()) {
                if (!first) out += ", ";
                first = false;
                toJson(e, out);
            }
            out += ']';
            break;
        }
        case Value::Kind::Map: {
            out += '{';
            bool first = true;
            for (const auto& kv : v.asMap()) {
                if (!first) out += ", ";
                first = false;
                toJson(Value::makeString(kv.first), out);
                out += ": ";
                toJson(kv.second, out);
            }
            out += '}';
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Scope: a chained name->Value lookup used during rendering.
// ---------------------------------------------------------------------------
//
// The interpreter renders nodes against an Env. An Env is a (parent, locals)
// pair: name lookups first consult `locals`, then walk up to `parent`.
// `set` writes into the *innermost* locals (so a `set` inside a loop body
// shadows/overrides for the remainder of that iteration's body). This matches
// Jinja's scoping closely enough for the templates we support.

struct Env {
    const Env* parent;
    const Map* locals; // borrowed; lifetime is the surrounding render call

    Value lookup(const std::string& name) const {
        if (locals) {
            auto it = locals->find(name);
            if (it != locals->end()) return it->second;
        }
        if (parent) return parent->lookup(name);
        return Value::makeUndefined();
    }
};

// ---------------------------------------------------------------------------
// Expression AST
// ---------------------------------------------------------------------------

struct Expr {
    virtual ~Expr() = default;
    virtual Value eval(const Env& env) const = 0;
};
using ExprPtr = std::unique_ptr<Expr>;

struct LitExpr : Expr {
    Value value;
    explicit LitExpr(Value v) : value(std::move(v)) {}
    Value eval(const Env&) const override { return value; }
};

struct VarExpr : Expr {
    std::string name;
    explicit VarExpr(std::string n) : name(std::move(n)) {}
    Value eval(const Env& env) const override { return env.lookup(name); }
};

struct AttrExpr : Expr {
    ExprPtr base;
    std::string attr;
    AttrExpr(ExprPtr b, std::string a) : base(std::move(b)), attr(std::move(a)) {}
    Value eval(const Env& env) const override {
        Value b = base->eval(env);
        // Attribute and subscript access are interchangeable on map values.
        if (b.kind() == Value::Kind::Map) return b.get(attr);
        return Value::makeUndefined();
    }
};

struct SubscriptExpr : Expr {
    ExprPtr base;
    ExprPtr index;
    SubscriptExpr(ExprPtr b, ExprPtr i) : base(std::move(b)), index(std::move(i)) {}
    Value eval(const Env& env) const override {
        Value b = base->eval(env);
        Value idx = index->eval(env);
        if (b.kind() == Value::Kind::List) {
            if (idx.kind() == Value::Kind::Int) {
                std::int64_t i = idx.asInt();
                std::int64_t n = static_cast<std::int64_t>(b.asList().size());
                if (i < 0) i += n;
                if (i >= 0 && i < n) return b.asList()[static_cast<size_t>(i)];
            }
            return Value::makeUndefined();
        }
        if (b.kind() == Value::Kind::Map) {
            if (idx.kind() == Value::Kind::String) return b.get(idx.asString());
            if (idx.kind() == Value::Kind::Int) return b.get(std::to_string(idx.asInt()));
        }
        return Value::makeUndefined();
    }
};

struct UnaryExpr : Expr {
    enum class Op { Not, Neg };
    Op op;
    ExprPtr operand;
    UnaryExpr(Op o, ExprPtr e) : op(o), operand(std::move(e)) {}
    Value eval(const Env& env) const override {
        Value v = operand->eval(env);
        if (op == Op::Not) return Value::makeBool(!v.truthy());
        if (v.kind() == Value::Kind::Int) return Value::makeInt(-v.asInt());
        return Value::makeInt(0);
    }
};

static std::int64_t asIntVal(const Value& v) {
    return v.kind() == Value::Kind::Int ? v.asInt() : 0;
}

static bool valuesEqual(const Value& a, const Value& b) {
    if (a.kind() == b.kind()) {
        switch (a.kind()) {
            case Value::Kind::Bool: return a.asBool() == b.asBool();
            case Value::Kind::Int: return a.asInt() == b.asInt();
            case Value::Kind::String: return a.asString() == b.asString();
            case Value::Kind::Undefined:
            case Value::Kind::Null: return true;
            default: break;
        }
    }
    if (a.kind() == Value::Kind::Int && b.kind() == Value::Kind::Bool)
        return a.asInt() == (b.asBool() ? 1 : 0);
    if (b.kind() == Value::Kind::Int && a.kind() == Value::Kind::Bool)
        return b.asInt() == (a.asBool() ? 1 : 0);
    return false;
}

struct BinaryExpr : Expr {
    enum class Op { Add, Sub, Eq, Ne, And, Or };
    Op op;
    ExprPtr lhs, rhs;
    BinaryExpr(Op o, ExprPtr l, ExprPtr r) : op(o), lhs(std::move(l)), rhs(std::move(r)) {}
    Value eval(const Env& env) const override {
        if (op == Op::And) {
            Value l = lhs->eval(env);
            if (!l.truthy()) return l;
            return rhs->eval(env);
        }
        if (op == Op::Or) {
            Value l = lhs->eval(env);
            if (l.truthy()) return l;
            return rhs->eval(env);
        }
        Value l = lhs->eval(env);
        Value r = rhs->eval(env);
        switch (op) {
            case Op::Add:
                if (l.kind() == Value::Kind::Int && r.kind() == Value::Kind::Int)
                    return Value::makeInt(l.asInt() + r.asInt());
                return Value::makeString(toStr(l) + toStr(r));
            case Op::Sub:
                return Value::makeInt(asIntVal(l) - asIntVal(r));
            case Op::Eq: return Value::makeBool(valuesEqual(l, r));
            case Op::Ne: return Value::makeBool(!valuesEqual(l, r));
            default: break;
        }
        return Value::makeUndefined();
    }
};

struct IsDefinedExpr : Expr {
    ExprPtr operand;
    bool negate;
    IsDefinedExpr(ExprPtr e, bool neg) : operand(std::move(e)), negate(neg) {}
    Value eval(const Env& env) const override {
        bool d = operand->eval(env).defined();
        return Value::makeBool(negate ? !d : d);
    }
};

struct FilterExpr : Expr {
    ExprPtr base;
    std::string name;
    FilterExpr(ExprPtr b, std::string n) : base(std::move(b)), name(std::move(n)) {}
    Value eval(const Env& env) const override {
        Value v = base->eval(env);
        if (name == "tojson") {
            std::string s;
            toJson(v, s);
            return Value::makeString(std::move(s));
        }
        return v;
    }
};

// ---------------------------------------------------------------------------
// Expression lexer + parser (recursive descent)
// ---------------------------------------------------------------------------

enum class TokKind {
    End, Ident, Int, String,
    Dot, LBracket, RBracket, LParen, RParen,
    Plus, Minus, EqEq, NotEq, Pipe,
    And, Or, Not, Is, Defined, True, False,
};

struct ExprToken {
    TokKind kind = TokKind::End;
    std::string text;
    std::int64_t ival = 0;
};

class ExprLexer {
public:
    explicit ExprLexer(const std::string& s) : s_(s), p_(0) {}

    ExprToken next() {
        skipWs();
        if (p_ >= s_.size()) return {TokKind::End, ""};
        char c = s_[p_];

        // String literals. Both single- and double-quoted strings process the
        // same escape sequences (\n, \t, \r, \\, \", \'), matching Python/
        // Jinja semantics. (The SmolLM2 template relies on \n inside
        // single-quoted strings producing a real newline.)
        if (c == '"' || c == '\'') {
            char quote = c;
            ++p_;
            std::string out;
            while (p_ < s_.size() && s_[p_] != quote) {
                if (s_[p_] == '\\' && p_ + 1 < s_.size()) {
                    char e = s_[p_ + 1];
                    switch (e) {
                        case 'n': out += '\n'; break;
                        case 't': out += '\t'; break;
                        case 'r': out += '\r'; break;
                        case '\\': out += '\\'; break;
                        case '"': out += '"'; break;
                        case '\'': out += '\''; break;
                        default: out += e; break;
                    }
                    p_ += 2;
                } else {
                    out += s_[p_++];
                }
            }
            if (p_ < s_.size()) ++p_;
            return {TokKind::String, std::move(out)};
        }

        if (std::isdigit(static_cast<unsigned char>(c))) {
            std::int64_t v = 0;
            while (p_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[p_]))) {
                v = v * 10 + (s_[p_] - '0');
                ++p_;
            }
            ExprToken t{TokKind::Int, ""};
            t.ival = v;
            return t;
        }

        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            std::string id;
            while (p_ < s_.size() &&
                   (std::isalnum(static_cast<unsigned char>(s_[p_])) || s_[p_] == '_')) {
                id += s_[p_++];
            }
            if (id == "and") return {TokKind::And, id};
            if (id == "or")  return {TokKind::Or, id};
            if (id == "not") return {TokKind::Not, id};
            if (id == "is")  return {TokKind::Is, id};
            if (id == "defined") return {TokKind::Defined, id};
            if (id == "true" || id == "True") return {TokKind::True, id};
            if (id == "false" || id == "False") return {TokKind::False, id};
            return {TokKind::Ident, id};
        }

        ++p_;
        switch (c) {
            case '.': return {TokKind::Dot, "."};
            case '[': return {TokKind::LBracket, "["};
            case ']': return {TokKind::RBracket, "]"};
            case '(': return {TokKind::LParen, "("};
            case ')': return {TokKind::RParen, ")"};
            case '+': return {TokKind::Plus, "+"};
            case '-': return {TokKind::Minus, "-"};
            case '|': return {TokKind::Pipe, "|"};
            case '=':
                if (p_ < s_.size() && s_[p_] == '=') { ++p_; return {TokKind::EqEq, "=="}; }
                break;
            case '!':
                if (p_ < s_.size() && s_[p_] == '=') { ++p_; return {TokKind::NotEq, "!="}; }
                break;
        }
        return next();
    }

private:
    void skipWs() {
        while (p_ < s_.size() &&
               (s_[p_] == ' ' || s_[p_] == '\t' || s_[p_] == '\n' || s_[p_] == '\r')) {
            ++p_;
        }
    }
    const std::string& s_;
    size_t p_;
};

// Precedence (low -> high): or, and, not, comparison/`is`, +/-, filter(|),
// postfix(.attr/[idx]), primary. Parentheses group.
class ExprParser {
public:
    explicit ExprParser(const std::string& src) : lex_(src) { advance(); }
    ExprPtr parse() { return parseOr(); }

private:
    void advance() { cur_ = lex_.next(); }

    ExprPtr parseOr() {
        ExprPtr lhs = parseAnd();
        while (cur_.kind == TokKind::Or) {
            advance();
            lhs = std::make_unique<BinaryExpr>(BinaryExpr::Op::Or, std::move(lhs), parseAnd());
        }
        return lhs;
    }
    ExprPtr parseAnd() {
        ExprPtr lhs = parseNot();
        while (cur_.kind == TokKind::And) {
            advance();
            lhs = std::make_unique<BinaryExpr>(BinaryExpr::Op::And, std::move(lhs), parseNot());
        }
        return lhs;
    }
    ExprPtr parseNot() {
        if (cur_.kind == TokKind::Not) {
            advance();
            return std::make_unique<UnaryExpr>(UnaryExpr::Op::Not, parseNot());
        }
        return parseComparison();
    }
    ExprPtr parseComparison() {
        ExprPtr lhs = parseAddSub();
        while (cur_.kind == TokKind::EqEq || cur_.kind == TokKind::NotEq) {
            bool eq = (cur_.kind == TokKind::EqEq);
            advance();
            lhs = std::make_unique<BinaryExpr>(
                eq ? BinaryExpr::Op::Eq : BinaryExpr::Op::Ne, std::move(lhs), parseAddSub());
        }
        if (cur_.kind == TokKind::Is) {
            advance();
            bool negate = false;
            if (cur_.kind == TokKind::Not) { advance(); negate = true; }
            bool definedTest = (cur_.kind == TokKind::Defined) ||
                               (cur_.kind == TokKind::Ident && cur_.text == "defined");
            // Consume the test name token (and any following args we don't support).
            if (cur_.kind == TokKind::Ident || cur_.kind == TokKind::Defined) advance();
            if (definedTest) {
                return std::make_unique<IsDefinedExpr>(std::move(lhs), negate);
            }
            // Unknown `is` test: fall back to a plain truthiness check.
            if (negate) {
                return std::make_unique<UnaryExpr>(UnaryExpr::Op::Not, std::move(lhs));
            }
            return lhs;
        }
        return lhs;
    }
    ExprPtr parseAddSub() {
        ExprPtr lhs = parseFilter();
        while (cur_.kind == TokKind::Plus || cur_.kind == TokKind::Minus) {
            bool add = (cur_.kind == TokKind::Plus);
            advance();
            lhs = std::make_unique<BinaryExpr>(
                add ? BinaryExpr::Op::Add : BinaryExpr::Op::Sub, std::move(lhs), parseFilter());
        }
        return lhs;
    }
    ExprPtr parseFilter() {
        ExprPtr lhs = parsePostfix();
        while (cur_.kind == TokKind::Pipe) {
            advance();
            std::string name;
            if (cur_.kind == TokKind::Ident) { name = cur_.text; advance(); }
            lhs = std::make_unique<FilterExpr>(std::move(lhs), std::move(name));
        }
        return lhs;
    }
    ExprPtr parsePostfix() {
        ExprPtr base = parsePrimary();
        while (true) {
            if (cur_.kind == TokKind::Dot) {
                advance();
                std::string attr;
                if (cur_.kind == TokKind::Ident) { attr = cur_.text; advance(); }
                base = std::make_unique<AttrExpr>(std::move(base), std::move(attr));
            } else if (cur_.kind == TokKind::LBracket) {
                advance();
                ExprPtr idx = parseOr();
                if (cur_.kind == TokKind::RBracket) advance();
                base = std::make_unique<SubscriptExpr>(std::move(base), std::move(idx));
            } else break;
        }
        return base;
    }
    ExprPtr parsePrimary() {
        if (cur_.kind == TokKind::LParen) {
            advance();
            ExprPtr e = parseOr();
            if (cur_.kind == TokKind::RParen) advance();
            return e;
        }
        if (cur_.kind == TokKind::Minus) {
            advance();
            return std::make_unique<UnaryExpr>(UnaryExpr::Op::Neg, parsePostfix());
        }
        if (cur_.kind == TokKind::Int) {
            ExprPtr e = std::make_unique<LitExpr>(Value::makeInt(cur_.ival));
            advance();
            return e;
        }
        if (cur_.kind == TokKind::String) {
            ExprPtr e = std::make_unique<LitExpr>(Value::makeString(cur_.text));
            advance();
            return e;
        }
        if (cur_.kind == TokKind::True) { advance(); return std::make_unique<LitExpr>(Value::makeBool(true)); }
        if (cur_.kind == TokKind::False) { advance(); return std::make_unique<LitExpr>(Value::makeBool(false)); }
        if (cur_.kind == TokKind::Ident) {
            std::string name = cur_.text;
            advance();
            return std::make_unique<VarExpr>(std::move(name));
        }
        return std::make_unique<LitExpr>(Value::makeUndefined());
    }

    ExprLexer lex_;
    ExprToken cur_;
};

ExprPtr parseExpression(const std::string& src) {
    ExprParser p(src);
    return p.parse();
}

// ---------------------------------------------------------------------------
// Template AST
// ---------------------------------------------------------------------------

struct TmplNode {
    virtual ~TmplNode() = default;
    virtual void render(std::string& out, const Env& env) const = 0;
};
using NodePtr = std::unique_ptr<TmplNode>;

struct TextNode : TmplNode {
    std::string text;
    explicit TextNode(std::string t) : text(std::move(t)) {}
    void render(std::string& out, const Env&) const override { out += text; }
};

struct OutputNode : TmplNode {
    ExprPtr expr;
    explicit OutputNode(ExprPtr e) : expr(std::move(e)) {}
    void render(std::string& out, const Env& env) const override {
        out += toStr(expr->eval(env));
    }
};

struct IfNode : TmplNode {
    struct Branch { ExprPtr cond; std::vector<NodePtr> body; }; // null cond => else
    std::vector<Branch> branches;
    void render(std::string& out, const Env& env) const override {
        for (const auto& br : branches) {
            if (!br.cond || br.cond->eval(env).truthy()) {
                for (const auto& n : br.body) n->render(out, env);
                return;
            }
        }
    }
};

// `set` writes into a mutable local scope carried alongside the env. The
// innermost scope pointer is stashed in a static that the top-level driver
// initializes and ForNode swaps per iteration.
struct SetNode : TmplNode {
    std::string var;
    ExprPtr expr;
    SetNode(std::string v, ExprPtr e) : var(std::move(v)), expr(std::move(e)) {}
    void render(std::string&, const Env& env) const override {
        if (g_currentLocals) {
            Value v = expr->eval(env);
            (*g_currentLocals)[var] = std::move(v);
        }
    }
    static Map* g_currentLocals;
};
Map* SetNode::g_currentLocals = nullptr;

struct ForNode : TmplNode {
    std::string var;
    ExprPtr iterable;
    std::vector<NodePtr> body;
    ForNode(std::string v, ExprPtr it) : var(std::move(v)), iterable(std::move(it)) {}
    void render(std::string& out, const Env& env) const override {
        Value seq = iterable->eval(env);
        if (seq.kind() != Value::Kind::List) return;
        const List& items = seq.asList();
        std::int64_t n = static_cast<std::int64_t>(items.size());
        for (std::int64_t i = 0; i < n; ++i) {
            Map local;
            local[var] = items[static_cast<size_t>(i)];

            Map loopMap;
            loopMap["index0"] = Value::makeInt(i);
            loopMap["index"] = Value::makeInt(i + 1);
            loopMap["first"] = Value::makeBool(i == 0);
            loopMap["last"] = Value::makeBool(i == n - 1);
            loopMap["length"] = Value::makeInt(n);
            local["loop"] = Value::makeMap(loopMap);

            Env child{&env, &local};
            // `set` inside the loop body writes into this iteration's local map.
            Map* saved = SetNode::g_currentLocals;
            SetNode::g_currentLocals = &local;
            for (const auto& node : body) node->render(out, child);
            SetNode::g_currentLocals = saved;
        }
    }
};

// ---------------------------------------------------------------------------
// Template tokenizer (literal text + tags, with whitespace control)
// ---------------------------------------------------------------------------

enum class TagType { For, EndFor, If, Elif, Else, EndIf, Set, Output };

struct RawTag {
    TagType type;
    std::string content; // inner expression source (trimmed)
};

struct RawToken {
    bool isText = false;
    std::string text;     // when isText
    RawTag tag;           // when !isText
};

// Strip leading/trailing whitespace.
std::string trim(const std::string& s) {
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

std::string firstWord(const std::string& s) {
    std::string w;
    for (char c : s) {
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') w += c;
        else break;
    }
    return w;
}

// Split the template into raw text / tag tokens, applying whitespace control.
// trim semantics:
//   `{%-` / `{{-` : strip trailing whitespace from the preceding text back to
//                  and including the preceding newline.
//   `-%}` / `-}}` : strip leading whitespace from the following text through
//                  and including the next newline.
std::vector<RawToken> tokenize(const std::string& tmpl) {
    std::vector<RawToken> out;
    size_t i = 0;
    std::string textBuf;

    auto flushText = [&]() {
        if (!textBuf.empty()) {
            RawToken t;
            t.isText = true;
            t.text = textBuf;
            out.push_back(std::move(t));
            textBuf.clear();
        }
    };

    while (i < tmpl.size()) {
        if (i + 1 < tmpl.size() && tmpl[i] == '{' &&
            (tmpl[i + 1] == '%' || tmpl[i + 1] == '{')) {
            bool isStmt = (tmpl[i + 1] == '%');
            size_t p = i + 2;
            bool trimBefore = false;
            if (p < tmpl.size() && tmpl[p] == '-') { trimBefore = true; ++p; }

            char closeFirst = isStmt ? '%' : '}';
            size_t end = std::string::npos;
            for (size_t q = p; q + 1 < tmpl.size(); ++q) {
                if (tmpl[q] == closeFirst && tmpl[q + 1] == '}') { end = q; break; }
            }
            if (end == std::string::npos) {
                // No closing found; treat '{' as literal text.
                textBuf += tmpl[i];
                ++i;
                continue;
            }

            bool trimAfter = false;
            size_t contentEnd = end;
            if (end > p && tmpl[end - 1] == '-') { trimAfter = true; contentEnd = end - 1; }

            // Apply left-trim to the preceding text buffer.
            if (trimBefore && !textBuf.empty()) {
                size_t k = textBuf.size();
                while (k > 0 && (textBuf[k - 1] == ' ' || textBuf[k - 1] == '\t' ||
                                 textBuf[k - 1] == '\r')) --k;
                if (k > 0 && (textBuf[k - 1] == '\n')) --k;
                textBuf.resize(k);
            }

            flushText();

            RawToken t;
            t.isText = false;
            std::string inner = tmpl.substr(p, contentEnd - p);
            if (isStmt) {
                std::string trimmed = trim(inner);
                std::string fw = firstWord(trimmed);
                if (fw == "for") t.tag.type = TagType::For;
                else if (fw == "endfor") t.tag.type = TagType::EndFor;
                else if (fw == "if") t.tag.type = TagType::If;
                else if (fw == "elif") t.tag.type = TagType::Elif;
                else if (fw == "else") t.tag.type = TagType::Else;
                else if (fw == "endif") t.tag.type = TagType::EndIf;
                else if (fw == "set") t.tag.type = TagType::Set;
                else { t.tag.type = TagType::Set; t.tag.content = ""; }
                t.tag.content = trimmed;
            } else {
                t.tag.type = TagType::Output;
                t.tag.content = trim(inner);
            }
            out.push_back(std::move(t));

            i = end + 2; // past the closing
            if (trimAfter) {
                while (i < tmpl.size() &&
                       (tmpl[i] == ' ' || tmpl[i] == '\t' || tmpl[i] == '\r')) ++i;
                if (i < tmpl.size() && tmpl[i] == '\n') ++i;
            }
            continue;
        }
        textBuf += tmpl[i];
        ++i;
    }
    flushText();
    return out;
}

// ---------------------------------------------------------------------------
// Template parser (builds node tree, balancing for/if blocks)
// ---------------------------------------------------------------------------

class TmplParser {
public:
    explicit TmplParser(std::vector<RawToken> toks) : toks_(std::move(toks)), pos_(0) {}
    std::vector<NodePtr> parse() { return parseNodes(); }

private:
    // parseNodes parses a run of nodes. It stops (without consuming) when it
    // sees a block terminator (endfor/elif/else/endif) at the top level; the
    // caller is responsible for dispatching on that terminator.
    std::vector<NodePtr> parseNodes() {
        std::vector<NodePtr> nodes;
        while (pos_ < toks_.size()) {
            const RawToken& t = toks_[pos_];
            if (t.isText) {
                nodes.push_back(std::make_unique<TextNode>(t.text));
                ++pos_;
                continue;
            }
            switch (t.tag.type) {
                case TagType::Output:
                    nodes.push_back(std::make_unique<OutputNode>(parseExpression(t.tag.content)));
                    ++pos_;
                    break;
                case TagType::Set:
                    nodes.push_back(makeSetNode(t.tag.content));
                    ++pos_;
                    break;
                case TagType::If: {
                    IfNode* ifn = new IfNode();
                    nodes.emplace_back(ifn);
                    IfNode::Branch br;
                    br.cond = parseExpression(stripKw(t.tag.content, "if"));
                    ++pos_;
                    br.body = parseNodes();
                    collectBranches(*ifn, std::move(br));
                    break;
                }
                case TagType::For: {
                    ForNode* forn = new ForNode(parseForVar(t.tag.content),
                                                parseForExpr(t.tag.content));
                    nodes.emplace_back(forn);
                    ++pos_;
                    forn->body = parseNodes();
                    if (pos_ < toks_.size() && !toks_[pos_].isText &&
                        toks_[pos_].tag.type == TagType::EndFor) ++pos_;
                    break;
                }
                // Block terminators: stop here; caller dispatches.
                case TagType::Elif:
                case TagType::Else:
                case TagType::EndIf:
                case TagType::EndFor:
                    return nodes;
            }
        }
        return nodes;
    }

    // After parsing the first branch body of an if, consume any number of
    // elif/else branches until endif.
    void collectBranches(IfNode& ifn, IfNode::Branch first) {
        ifn.branches.push_back(std::move(first));
        while (pos_ < toks_.size() && !toks_[pos_].isText) {
            TagType tt = toks_[pos_].tag.type;
            if (tt == TagType::Elif) {
                ExprPtr cond = parseExpression(stripKw(toks_[pos_].tag.content, "elif"));
                ++pos_;
                IfNode::Branch br;
                br.cond = std::move(cond);
                br.body = parseNodes();
                ifn.branches.push_back(std::move(br));
            } else if (tt == TagType::Else) {
                ++pos_;
                IfNode::Branch br; // null cond
                br.body = parseNodes();
                ifn.branches.push_back(std::move(br));
            } else if (tt == TagType::EndIf) {
                ++pos_;
                return;
            } else {
                // Unexpected (e.g. stray endfor); stop to avoid infinite loop.
                return;
            }
        }
    }

    static std::string stripKw(const std::string& s, const std::string& kw) {
        std::string t = trim(s);
        if (t.compare(0, kw.size(), kw) == 0) {
            std::string rest = t.substr(kw.size());
            size_t b = 0;
            while (b < rest.size() && std::isspace(static_cast<unsigned char>(rest[b]))) ++b;
            return rest.substr(b);
        }
        return t;
    }

    static std::string parseForVar(const std::string& content) {
        // "for X in EXPR"
        std::string body = stripKw(content, "for");
        auto pos = body.find(" in ");
        std::string var = (pos != std::string::npos) ? body.substr(0, pos) : body;
        return trim(var);
    }
    static ExprPtr parseForExpr(const std::string& content) {
        std::string body = stripKw(content, "for");
        auto pos = body.find(" in ");
        std::string exprSrc = (pos != std::string::npos) ? body.substr(pos + 4) : "";
        return parseExpression(exprSrc);
    }

    static NodePtr makeSetNode(const std::string& content) {
        // "set var = expr"
        std::string body = stripKw(content, "set");
        size_t eq = body.find('=');
        if (eq == std::string::npos) return std::make_unique<TextNode>("");
        std::string var = trim(body.substr(0, eq));
        std::string exprSrc = body.substr(eq + 1);
        return std::make_unique<SetNode>(var, parseExpression(exprSrc));
    }

    std::vector<RawToken> toks_;
    size_t pos_;
};

// ---------------------------------------------------------------------------
// Top-level interpreter
// ---------------------------------------------------------------------------

Value buildRootScope(const std::vector<ChatMessage>& messages, bool add_generation_prompt) {
    List msgList;
    msgList.reserve(messages.size());
    for (const auto& m : messages) {
        Map obj;
        obj["role"] = Value::makeString(m.role);
        obj["content"] = Value::makeString(m.content);
        // tool_calls intentionally absent -> undefined (falsy) for our templates.
        msgList.push_back(Value::makeMap(obj));
    }
    Map root;
    root["messages"] = Value::makeList(std::move(msgList));
    root["add_generation_prompt"] = Value::makeBool(add_generation_prompt);
    root["tools"] = Value::makeList(List{}); // empty list (no tool-calling)
    return Value::makeMap(root);
}

} // namespace

std::string apply_chat_template(const std::string& tmpl,
                                const std::vector<ChatMessage>& messages,
                                bool add_generation_prompt) {
    Value root = buildRootScope(messages, add_generation_prompt);

    // The root scope doubles as the mutable locals map so that top-level `set`
    // assignments are visible to subsequent nodes.
    Map scope = root.asMap();
    SetNode::g_currentLocals = &scope;

    Env env{nullptr, &scope};

    std::vector<RawToken> toks = tokenize(tmpl);
    TmplParser parser(std::move(toks));
    std::vector<NodePtr> nodes = parser.parse();

    std::string out;
    for (const auto& n : nodes) n->render(out, env);

    SetNode::g_currentLocals = nullptr;
    return out;
}

} // namespace core
