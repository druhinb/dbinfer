#include "grammar/grammar.hpp"

#include <algorithm>
#include <cstddef>
#include <deque>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "try.hpp"

namespace dbinfer::grammar {

namespace {

bool is_end(const Element* pos) { return pos->type == ElemType::End || pos->type == ElemType::Alt; }

// GPT/JSON grammars mix single-byte and multi-byte codepoints. decode the byte
// span into codepoints, resuming a sequence carried in partial and returning
// any trailing unfinished codepoint. mirrors llama.cpp decode_utf8.
std::pair<std::vector<std::uint32_t>, PartialUtf8> decode_utf8(std::string_view src,
                                                               PartialUtf8 partial) {
  static const int lookup[16] = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4};
  std::vector<std::uint32_t> cpts;
  cpts.reserve(src.size());
  std::size_t i = 0;
  std::uint32_t value = partial.value;
  int n_remain = partial.n_remain;

  while (i < src.size() && n_remain > 0) {
    const std::uint8_t b = static_cast<std::uint8_t>(src[i]);
    if ((b >> 6) != 2) {
      cpts.push_back(0);
      return {cpts, PartialUtf8{0, -1}};
    }
    value = (value << 6) + (b & 0x3F);
    ++i;
    --n_remain;
  }
  if (partial.n_remain > 0 && n_remain == 0) cpts.push_back(value);

  while (i < src.size()) {
    const std::uint8_t first = static_cast<std::uint8_t>(src[i]);
    n_remain = lookup[first >> 4] - 1;
    if (n_remain < 0) {
      cpts.clear();
      cpts.push_back(0);
      return {cpts, PartialUtf8{0, n_remain}};
    }
    const std::uint8_t mask = static_cast<std::uint8_t>((1 << (7 - n_remain)) - 1);
    value = first & mask;
    ++i;
    while (i < src.size() && n_remain > 0) {
      value = (value << 6) + (static_cast<std::uint8_t>(src[i]) & 0x3F);
      ++i;
      --n_remain;
    }
    if (n_remain == 0) cpts.push_back(value);
  }
  return {cpts, PartialUtf8{value, n_remain}};
}

// returns whether chr matches the set opening at pos and the element following
// the set. the set is Char/CharNot then any RngUpper and CharAlt continuations.
std::pair<bool, const Element*> match_char(const Element* pos, std::uint32_t chr) {
  bool found = false;
  const bool positive = pos->type == ElemType::Char;
  const Element* p = pos;
  do {
    if (p[1].type == ElemType::RngUpper) {
      found = found || (p->value <= chr && chr <= p[1].value);
      p += 2;
    } else {
      found = found || p->value == chr;
      p += 1;
    }
  } while (p->type == ElemType::CharAlt);
  return {found == positive, p};
}

// whether the codepoint range a partial UTF-8 sequence could complete to
// overlaps the set opening at pos. keeps tokens whose trailing bytes could
// still satisfy the grammar once the next token completes them.
bool match_partial_char(const Element* pos, PartialUtf8 partial) {
  const bool positive = pos->type == ElemType::Char;
  const std::uint32_t value = partial.value;
  const int n_remain = partial.n_remain;
  if (n_remain < 0 || (n_remain == 1 && value < 2)) return false;

  std::uint32_t low = value << (n_remain * 6);
  const std::uint32_t high = low | ((1u << (n_remain * 6)) - 1);
  if (low == 0) {
    if (n_remain == 2)
      low = 1u << 11;
    else if (n_remain == 3)
      low = 1u << 16;
  }

  const Element* p = pos;
  do {
    if (p[1].type == ElemType::RngUpper) {
      if (p->value <= high && low <= p[1].value) return positive;
      p += 2;
    } else {
      if (low <= p->value && p->value <= high) return positive;
      p += 1;
    }
  } while (p->type == ElemType::CharAlt);
  return !positive;
}

void append_utf8(std::uint32_t cpt, std::string& out) {
  if (cpt < 0x80) {
    out.push_back(static_cast<char>(cpt));
  } else if (cpt < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cpt >> 6)));
    out.push_back(static_cast<char>(0x80 | (cpt & 0x3F)));
  } else if (cpt < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cpt >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cpt >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cpt & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cpt >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cpt >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cpt >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cpt & 0x3F)));
  }
}

// identifies a state by the element positions on its stacks, so a breadth-first
// completion search can skip states it has already visited.
std::string state_key(const State& s) {
  std::string key;
  for (const Stack& stk : s.stacks) {
    for (const Element* e : stk) {
      const std::uintptr_t p = reinterpret_cast<std::uintptr_t>(e);
      key.append(reinterpret_cast<const char*>(&p), sizeof p);
    }
    key.push_back('|');
  }
  return key;
}

bool is_word_char(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' ||
         c == '_';
}

const char* parse_space(const char* pos, bool newline_ok) {
  while (*pos == ' ' || *pos == '\t' || *pos == '#' ||
         (newline_ok && (*pos == '\r' || *pos == '\n'))) {
    if (*pos == '#') {
      while (*pos != 0 && *pos != '\r' && *pos != '\n') ++pos;
    } else {
      ++pos;
    }
  }
  return pos;
}

const char* parse_name(const char* pos) {
  while (is_word_char(*pos)) ++pos;
  return pos;
}

std::expected<std::uint32_t, Error> parse_hex(const char* pos, int n, const char*& out) {
  std::uint32_t value = 0;
  for (int i = 0; i < n; ++i) {
    const char c = pos[i];
    value <<= 4;
    if (c >= '0' && c <= '9')
      value += static_cast<std::uint32_t>(c - '0');
    else if (c >= 'a' && c <= 'f')
      value += static_cast<std::uint32_t>(c - 'a' + 10);
    else if (c >= 'A' && c <= 'F')
      value += static_cast<std::uint32_t>(c - 'A' + 10);
    else
      return std::unexpected(Error{"expecting " + std::to_string(n) + " hex digits in escape"});
  }
  out = pos + n;
  return value;
}

std::expected<std::uint32_t, Error> decode_one(const char* pos, const char*& out) {
  const std::uint8_t first = static_cast<std::uint8_t>(*pos);
  int n_remain = 0;
  std::uint32_t value = first;
  if (first < 0x80) {
    n_remain = 0;
  } else if ((first >> 5) == 0x6) {
    value = first & 0x1F;
    n_remain = 1;
  } else if ((first >> 4) == 0xE) {
    value = first & 0x0F;
    n_remain = 2;
  } else if ((first >> 3) == 0x1E) {
    value = first & 0x07;
    n_remain = 3;
  } else {
    return std::unexpected(Error{"invalid utf-8 lead byte in grammar"});
  }

  ++pos;
  for (int i = 0; i < n_remain; ++i) {
    if ((static_cast<std::uint8_t>(*pos) >> 6) != 2)
      return std::unexpected(Error{"truncated utf-8 sequence in grammar"});
    value = (value << 6) + (static_cast<std::uint8_t>(*pos) & 0x3F);
    ++pos;
  }
  out = pos;
  return value;
}

// reads one character terminal, decoding escapes and multi-byte UTF-8.
std::expected<std::uint32_t, Error> parse_char(const char* pos, const char*& out) {
  if (*pos == '\\') {
    switch (pos[1]) {
      case 'x':
        return parse_hex(pos + 2, 2, out);
      case 'u':
        return parse_hex(pos + 2, 4, out);
      case 'U':
        return parse_hex(pos + 2, 8, out);
      case 't':
        out = pos + 2;
        return static_cast<std::uint32_t>('\t');
      case 'r':
        out = pos + 2;
        return static_cast<std::uint32_t>('\r');
      case 'n':
        out = pos + 2;
        return static_cast<std::uint32_t>('\n');
      case '\\':
      case '"':
      case '[':
      case ']':
        out = pos + 2;
        return static_cast<std::uint32_t>(pos[1]);
      default:
        return std::unexpected(Error{std::string("unknown escape '\\") + pos[1] + "'"});
    }
  }

  if (*pos != 0) return decode_one(pos, out);
  return std::unexpected(Error{"unexpected end of input in character terminal"});
}

struct ParseState {
  std::unordered_map<std::string, std::uint32_t> symbol_ids;
  std::vector<std::string> id_to_name;
  std::vector<std::vector<Element>> rules;
  std::size_t depth = 0;

  std::uint32_t get_symbol_id(std::string_view name) {
    const std::string key(name);
    auto it = symbol_ids.find(key);
    if (it != symbol_ids.end()) return it->second;
    const std::uint32_t id = static_cast<std::uint32_t>(id_to_name.size());
    symbol_ids.emplace(key, id);
    id_to_name.push_back(key);
    return id;
  }

  std::uint32_t generate_symbol_id(std::string_view base) {
    const std::uint32_t id = static_cast<std::uint32_t>(id_to_name.size());
    id_to_name.push_back(std::string(base) + "_" + std::to_string(id));
    return id;
  }

  void add_rule(std::uint32_t id, std::vector<Element> rule) {
    if (rules.size() <= id) rules.resize(id + 1);
    rules[id] = std::move(rule);
  }
};

// nested groups recurse parse_alternates through parse_sequence, so a deeply
// parenthesized grammar would overflow the native stack. bound it at parse time.
constexpr std::size_t kMaxParseDepth = 512;

struct DepthGuard {
  std::size_t& d;
  ~DepthGuard() { --d; }
};

std::expected<const char*, Error> parse_alternates(ParseState& st, const char* src,
                                                   std::string_view rule_name,
                                                   std::uint32_t rule_id, bool is_nested);

// consumes a `"..."` string literal, appending one Char element per decoded
// codepoint starting at last_sym_start.
std::expected<const char*, Error> parse_string_literal(const char* pos, std::vector<Element>& out,
                                                       std::size_t& last_sym_start,
                                                       bool is_nested) {
  ++pos;
  last_sym_start = out.size();
  while (*pos != '"') {
    if (*pos == 0) return std::unexpected(Error{"unterminated string literal"});
    const std::uint32_t cpt = TRY(parse_char(pos, pos));
    out.push_back({ElemType::Char, cpt});
  }
  return parse_space(pos + 1, is_nested);
}

// consumes a `[...]` character class, appending Char/CharNot then its
// RngUpper and CharAlt continuations starting at last_sym_start.
std::expected<const char*, Error> parse_char_class(const char* pos, std::vector<Element>& out,
                                                   std::size_t& last_sym_start, bool is_nested) {
  ++pos;
  ElemType start_type = ElemType::Char;
  if (*pos == '^') {
    ++pos;
    start_type = ElemType::CharNot;
  }

  last_sym_start = out.size();
  while (*pos != ']') {
    if (*pos == 0) return std::unexpected(Error{"unterminated character class"});
    const std::uint32_t cpt = TRY(parse_char(pos, pos));
    const ElemType type = last_sym_start < out.size() ? ElemType::CharAlt : start_type;
    out.push_back({type, cpt});
    if (pos[0] == '-' && pos[1] != ']' && pos[1] != 0) {
      const std::uint32_t upper = TRY(parse_char(pos + 1, pos));
      out.push_back({ElemType::RngUpper, upper});
    }
  }
  return parse_space(pos + 1, is_nested);
}

// consumes a bare rule-name reference, appending one RuleRef element.
const char* parse_rule_ref(ParseState& st, const char* pos, std::vector<Element>& out,
                           std::size_t& last_sym_start, bool is_nested) {
  const char* name_end = parse_name(pos);
  const std::uint32_t ref = st.get_symbol_id(std::string_view(pos, name_end - pos));
  pos = parse_space(name_end, is_nested);
  last_sym_start = out.size();
  out.push_back({ElemType::RuleRef, ref});
  return pos;
}

// consumes a `(...)` group, recursing into parse_alternates for its body and
// appending a RuleRef to the generated sub-rule.
std::expected<const char*, Error> parse_group(ParseState& st, const char* pos,
                                              std::string_view rule_name, std::vector<Element>& out,
                                              std::size_t& last_sym_start, bool is_nested) {
  pos = parse_space(pos + 1, true);
  const std::uint32_t sub_id = st.generate_symbol_id(rule_name);
  last_sym_start = out.size();
  out.push_back({ElemType::RuleRef, sub_id});
  if (*pos != ')') pos = TRY(parse_alternates(st, pos, rule_name, sub_id, true));
  if (*pos != ')') return std::unexpected(Error{"expecting ')' to close group"});
  return parse_space(pos + 1, is_nested);
}

// collapses the symbol at last_sym_start under a `*`/`+`/`?` operator into a
// generated sub-rule, replacing it in out with one RuleRef.
std::expected<const char*, Error> parse_repetition(ParseState& st, const char* pos,
                                                   std::string_view rule_name,
                                                   std::vector<Element>& out,
                                                   std::size_t last_sym_start, bool is_nested) {
  if (last_sym_start == out.size())
    return std::unexpected(Error{"repetition operator with no preceding symbol"});

  const std::uint32_t sub_id = st.generate_symbol_id(rule_name);
  std::vector<Element> sub(out.begin() + static_cast<std::ptrdiff_t>(last_sym_start), out.end());
  if (*pos == '*' || *pos == '+') sub.push_back({ElemType::RuleRef, sub_id});
  sub.push_back({ElemType::Alt, 0});
  if (*pos == '+')
    sub.insert(sub.end(), out.begin() + static_cast<std::ptrdiff_t>(last_sym_start), out.end());
  sub.push_back({ElemType::End, 0});
  st.add_rule(sub_id, std::move(sub));
  out.resize(last_sym_start);
  out.push_back({ElemType::RuleRef, sub_id});
  return parse_space(pos + 1, is_nested);
}

std::expected<const char*, Error> parse_sequence(ParseState& st, const char* src,
                                                 std::string_view rule_name,
                                                 std::vector<Element>& out, bool is_nested) {
  std::size_t last_sym_start = out.size();
  const char* pos = src;

  while (*pos != 0) {
    if (*pos == '"') {
      pos = TRY(parse_string_literal(pos, out, last_sym_start, is_nested));
    } else if (*pos == '[') {
      pos = TRY(parse_char_class(pos, out, last_sym_start, is_nested));
    } else if (is_word_char(*pos)) {
      pos = parse_rule_ref(st, pos, out, last_sym_start, is_nested);
    } else if (*pos == '(') {
      pos = TRY(parse_group(st, pos, rule_name, out, last_sym_start, is_nested));
    } else if (*pos == '*' || *pos == '+' || *pos == '?') {
      pos = TRY(parse_repetition(st, pos, rule_name, out, last_sym_start, is_nested));
    } else {
      break;
    }
  }
  return pos;
}

std::expected<const char*, Error> parse_alternates(ParseState& st, const char* src,
                                                   std::string_view rule_name,
                                                   std::uint32_t rule_id, bool is_nested) {
  ++st.depth;
  DepthGuard guard{st.depth};
  if (st.depth > kMaxParseDepth)
    return std::unexpected(Error{"grammar nesting exceeds maximum depth"});

  std::vector<Element> rule;
  const char* pos = TRY(parse_sequence(st, src, rule_name, rule, is_nested));
  while (*pos == '|') {
    rule.push_back({ElemType::Alt, 0});
    pos = parse_space(pos + 1, true);
    pos = TRY(parse_sequence(st, pos, rule_name, rule, is_nested));
  }
  rule.push_back({ElemType::End, 0});
  st.add_rule(rule_id, std::move(rule));
  return pos;
}

// advances past one grammar symbol: a rule reference or a character set with
// its RngUpper and CharAlt continuations.
const Element* skip_symbol(const Element* p) {
  if (p->type != ElemType::Char && p->type != ElemType::CharNot) return p + 1;
  do {
    p += p[1].type == ElemType::RngUpper ? 2 : 1;
  } while (p->type == ElemType::CharAlt);
  return p;
}

// a rule is nullable when some alternate derives the empty string. characters
// consume input, so only rule references to nullable rules keep an alternate
// nullable.
std::vector<bool> nullable_rules(const std::vector<std::vector<Element>>& rules) {
  std::vector<bool> nullable(rules.size(), false);

  for (bool changed = true; changed;) {
    changed = false;
    for (std::size_t id = 0; id < rules.size(); ++id) {
      if (nullable[id]) continue;
      const Element* p = rules[id].data();
      bool alt_nullable = true;
      for (;;) {
        if (p->type == ElemType::End || p->type == ElemType::Alt) {
          if (alt_nullable) {
            nullable[id] = true;
            changed = true;
            break;
          }
          if (p->type == ElemType::End) break;
          ++p;
          alt_nullable = true;
        } else if (p->type == ElemType::RuleRef) {
          alt_nullable = alt_nullable && nullable[p->value];
          ++p;
        } else {
          alt_nullable = false;
          p = skip_symbol(p);
        }
      }
    }
  }
  return nullable;
}

// a rule left-refers another when an alternate reaches it through only nullable
// prefixes, so the automaton expands it before consuming input. a cycle in that
// relation is left recursion advance_stack recurses on without bound.
std::optional<std::uint32_t> left_recursive_rule(const std::vector<std::vector<Element>>& rules,
                                                 const std::vector<bool>& nullable) {
  const std::size_t n = rules.size();
  std::vector<std::vector<std::uint32_t>> edges(n);

  for (std::size_t id = 0; id < n; ++id) {
    const Element* p = rules[id].data();
    bool in_left = true;
    for (;;) {
      if (p->type == ElemType::End || p->type == ElemType::Alt) {
        if (p->type == ElemType::End) break;
        ++p;
        in_left = true;
      } else if (p->type == ElemType::RuleRef) {
        if (in_left) {
          edges[id].push_back(p->value);
          in_left = nullable[p->value];
        }
        ++p;
      } else {
        in_left = false;
        p = skip_symbol(p);
      }
    }
  }

  // iterative dfs so a deeply nested grammar never overflows the native stack.
  // a back edge to a gray node names a rule sitting on the cycle.
  enum class Color : std::uint8_t { White, Gray, Black };
  std::vector<Color> color(n, Color::White);
  std::vector<std::size_t> next(n, 0);
  std::vector<std::uint32_t> stack;

  for (std::uint32_t start = 0; start < n; ++start) {
    if (color[start] != Color::White) continue;
    color[start] = Color::Gray;
    next[start] = 0;
    stack.push_back(start);
    while (!stack.empty()) {
      const std::uint32_t u = stack.back();
      if (next[u] < edges[u].size()) {
        const std::uint32_t v = edges[u][next[u]++];
        if (color[v] == Color::White) {
          color[v] = Color::Gray;
          next[v] = 0;
          stack.push_back(v);
        } else if (color[v] == Color::Gray) {
          return v;
        }
      } else {
        color[u] = Color::Black;
        stack.pop_back();
      }
    }
  }
  return std::nullopt;
}

std::expected<const char*, Error> parse_rule(ParseState& st, const char* src) {
  const char* name_end = parse_name(src);
  if (name_end == src) return std::unexpected(Error{"expecting rule name"});
  const char* pos = parse_space(name_end, false);
  const std::string_view name(src, name_end - src);
  const std::uint32_t rule_id = st.get_symbol_id(name);

  if (!(pos[0] == ':' && pos[1] == ':' && pos[2] == '='))
    return std::unexpected(Error{"expecting '::=' after rule '" + std::string(name) + "'"});
  pos = parse_space(pos + 3, true);

  pos = TRY(parse_alternates(st, pos, name, rule_id, false));

  if (*pos == '\r')
    pos += pos[1] == '\n' ? 2 : 1;
  else if (*pos == '\n')
    ++pos;
  else if (*pos != 0)
    return std::unexpected(Error{"expecting newline after rule '" + std::string(name) + "'"});
  return parse_space(pos, true);
}

}  // namespace

std::expected<Grammar, Error> Grammar::parse(std::string_view text) {
  const std::string owned(text);
  ParseState st;
  const char* pos = parse_space(owned.c_str(), true);
  while (*pos != 0) pos = TRY(parse_rule(st, pos));

  for (std::size_t id = 0; id < st.rules.size(); ++id) {
    if (st.rules[id].empty())
      return std::unexpected(Error{"undefined rule '" + st.id_to_name[id] + "'"});
    for (const Element& e : st.rules[id]) {
      if (e.type == ElemType::RuleRef && e.value >= st.rules.size())
        return std::unexpected(Error{"rule '" + st.id_to_name[id] + "' references unknown rule"});
    }
  }
  auto root_it = st.symbol_ids.find("root");
  if (root_it == st.symbol_ids.end()) return std::unexpected(Error{"grammar has no 'root' rule"});

  const std::vector<bool> nullable = nullable_rules(st.rules);
  if (auto lr = left_recursive_rule(st.rules, nullable))
    return std::unexpected(Error{"grammar rule '" + st.id_to_name[*lr] + "' is left-recursive"});

  Grammar g;
  g.rules_ = std::move(st.rules);
  g.root_ = root_it->second;
  return g;
}

void Grammar::advance_stack(const Stack& stack, std::vector<Stack>& out) const {
  if (stack.empty()) {
    if (std::find(out.begin(), out.end(), stack) == out.end()) out.push_back(stack);
    return;
  }

  const Element* pos = stack.back();
  switch (pos->type) {
    case ElemType::RuleRef: {
      const Element* sub = rules_[pos->value].data();
      for (;;) {
        Stack ns(stack.begin(), stack.end() - 1);
        if (!is_end(pos + 1)) ns.push_back(pos + 1);
        if (!is_end(sub)) ns.push_back(sub);
        advance_stack(ns, out);
        while (!is_end(sub)) ++sub;
        if (sub->type == ElemType::Alt)
          ++sub;
        else
          break;
      }
      break;
    }
    case ElemType::Char:
    case ElemType::CharNot:
      if (std::find(out.begin(), out.end(), stack) == out.end()) out.push_back(stack);
      break;
    default:
      break;
  }
}

std::vector<Stack> Grammar::accept(const std::vector<Stack>& stacks, std::uint32_t chr) const {
  std::vector<Stack> out;
  for (const Stack& stack : stacks) {
    if (stack.empty()) continue;
    auto [matched, next] = match_char(stack.back(), chr);
    if (!matched) continue;
    Stack ns(stack.begin(), stack.end() - 1);
    if (!is_end(next)) ns.push_back(next);
    advance_stack(ns, out);
  }
  return out;
}

State Grammar::start() const {
  State s;
  const Element* pos = rules_[root_].data();
  for (;;) {
    Stack stack;
    if (!is_end(pos)) stack.push_back(pos);
    advance_stack(stack, s.stacks);
    while (!is_end(pos)) ++pos;
    if (pos->type == ElemType::Alt)
      ++pos;
    else
      break;
  }
  return s;
}

bool Grammar::complete(const State& s) const {
  return std::any_of(s.stacks.begin(), s.stacks.end(),
                     [](const Stack& stack) { return stack.empty(); });
}

std::optional<State> Grammar::feed(const State& s, std::string_view bytes) const {
  if (bytes.empty()) return std::nullopt;

  auto [cpts, partial] = decode_utf8(bytes, s.partial);
  State cur;
  cur.stacks = s.stacks;
  for (const std::uint32_t cpt : cpts) {
    cur.stacks = accept(cur.stacks, cpt);
    if (cur.stacks.empty()) return std::nullopt;
  }
  cur.partial = partial;

  if (partial.n_remain < 0) return std::nullopt;
  if (partial.n_remain > 0) {
    const bool possible = std::any_of(cur.stacks.begin(), cur.stacks.end(), [&](const Stack& stk) {
      return !stk.empty() && match_partial_char(stk.back(), partial);
    });
    if (!possible) return std::nullopt;
  }
  return cur;
}

std::optional<std::string> Grammar::complete_suffix(const State& s) const {
  if (complete(s)) return std::string{};

  struct Node {
    State state;
    std::string path;
  };
  // closing terminals are single codepoints, so a bounded breadth-first search
  // over positive char alternatives finds the shortest completion.
  constexpr std::size_t max_len = 64;
  std::deque<Node> queue;
  std::unordered_set<std::string> seen;
  queue.push_back({s, std::string{}});
  seen.insert(state_key(s));

  while (!queue.empty()) {
    Node node = std::move(queue.front());
    queue.pop_front();
    if (node.path.size() >= max_len) continue;

    std::vector<std::uint32_t> cands;
    for (const Stack& stk : node.state.stacks) {
      if (stk.empty() || stk.back()->type != ElemType::Char) continue;
      const Element* p = stk.back();
      do {
        cands.push_back(p->value);
        p += (p[1].type == ElemType::RngUpper) ? 2 : 1;
      } while (p->type == ElemType::CharAlt);
    }

    for (const std::uint32_t c : cands) {
      std::string bytes;
      append_utf8(c, bytes);
      auto next = feed(node.state, bytes);
      if (!next) continue;
      std::string path = node.path + bytes;
      if (complete(*next)) return path;
      if (seen.insert(state_key(*next)).second)
        queue.push_back({std::move(*next), std::move(path)});
    }
  }
  return std::nullopt;
}

// clang-format off
Matcher::Matcher(Grammar grammar, std::vector<std::string> token_bytes, std::int32_t eos_id)
: grammar_(std::move(grammar)),
token_bytes_(std::move(token_bytes)),
eos_id_(eos_id),
state_(grammar_.start()) {}
// clang-format on

void Matcher::mask(std::span<float> logits) const {
  const float ninf = -std::numeric_limits<float>::infinity();
  const bool comp = complete();
  const std::size_t n = std::min(logits.size(), token_bytes_.size());
  for (std::size_t id = 0; id < n; ++id) {
    if (static_cast<std::int32_t>(id) == eos_id_) {
      if (!comp) logits[id] = ninf;
      continue;
    }
    if (!grammar_.feed(state_, token_bytes_[id])) logits[id] = ninf;
  }
}

void Matcher::accept(std::int32_t token) {
  if (token == eos_id_ || token < 0 || static_cast<std::size_t>(token) >= token_bytes_.size())
    return;
  if (auto next = grammar_.feed(state_, token_bytes_[static_cast<std::size_t>(token)]))
    state_ = std::move(*next);
}

bool Matcher::complete() const { return grammar_.complete(state_); }

}  // namespace dbinfer::grammar
