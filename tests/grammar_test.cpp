// ctest for the GBNF parser, stack automaton, and per-step logit mask.

#include "grammar/grammar.hpp"

#include <cstdio>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "test_util.hpp"

namespace {

using dbinfer::grammar::Grammar;
using dbinfer::grammar::Matcher;
using dbinfer::grammar::State;
using dbinfer::test::check;
using dbinfer::test::g_failures;

Grammar must_parse(std::string_view text) {
  auto g = Grammar::parse(text);
  if (g) return std::move(*g);
  std::printf("FAIL parse: %s\n", g.error().message.c_str());
  ++g_failures;
  auto fallback = Grammar::parse("root ::= \"x\"");
  return std::move(*fallback);
}

// feeds the whole string as one blob and reports whether the grammar reaches a
// complete (accepting) state with no rejection.
bool matches(const Grammar& g, std::string_view s) {
  State st = g.start();
  auto r = g.feed(st, s);
  if (!r) return false;
  return g.complete(*r);
}

bool rejected(const Grammar& g, std::string_view s) {
  State st = g.start();
  return !g.feed(st, s).has_value();
}

}  // namespace

const char kJson[] =
    "root   ::= object\n"
    "value  ::= object | array | string | number | (\"true\" | \"false\" | \"null\") ws\n"
    "object ::= \"{\" ws ( string \":\" ws value (\",\" ws string \":\" ws value)* )? \"}\" ws\n"
    "array  ::= \"[\" ws ( value (\",\" ws value)* )? \"]\" ws\n"
    "string ::= \"\\\"\" ( [^\"\\\\\\x00-\\x1F] | \"\\\\\" ([\"\\\\/bfnrt] | \"u\" [0-9a-fA-F] "
    "[0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) )* \"\\\"\" ws\n"
    "number ::= (\"-\"? ([0-9] | [1-9] [0-9]*)) (\".\" [0-9]+)? ([eE] [-+]? [0-9]+)? ws\n"
    "ws ::= ([ \\t\\n] ws)?\n";

namespace {

void test_literal_matching() {
  Grammar g = must_parse("root ::= \"ab\"");
  check(matches(g, "ab"), "literal accepts ab");
  check(rejected(g, "ac"), "literal rejects ac");
  check(!matches(g, "a"), "literal a is incomplete");
  check(rejected(g, "abc"), "literal rejects trailing c");
}

void test_alternation() {
  Grammar g = must_parse("root ::= \"a\" | \"bb\"");
  check(matches(g, "a"), "alt accepts a");
  check(matches(g, "bb"), "alt accepts bb");
  check(!matches(g, "b"), "alt b is an incomplete prefix");
  check(rejected(g, "c"), "alt rejects c");
}

void test_char_range_plus() {
  Grammar g = must_parse("root ::= [0-9]+");
  check(matches(g, "0"), "range accepts single digit");
  check(matches(g, "12345"), "range accepts many digits");
  check(rejected(g, "1a"), "range rejects letter");
  check(!matches(g, ""), "empty not complete for +");
}

void test_optional() {
  Grammar g = must_parse("root ::= \"x\"? \"y\"");
  check(matches(g, "y"), "optional omitted");
  check(matches(g, "xy"), "optional present");
  check(rejected(g, "xxy"), "optional at most once");
}

void test_negated_char_class() {
  Grammar g = must_parse("root ::= [^\"\\\\]");
  check(matches(g, "a"), "negated set accepts a");
  check(rejected(g, "\""), "negated set rejects quote");
  check(matches(g, "\xC3\xA9"), "negated set accepts multibyte codepoint");
}

void test_partial_utf8_across_feeds() {
  // a multibyte codepoint split across two feeds must carry the partial state.
  Grammar g = must_parse("root ::= [^\"\\\\]");
  State st = g.start();
  auto r1 = g.feed(st, "\xC3");
  check(r1.has_value(), "partial lead byte accepted");
  if (r1) {
    auto r2 = g.feed(*r1, "\xA9");
    check(r2.has_value() && g.complete(*r2), "partial completes to full codepoint");
  }
}

void test_json_grammar() {
  Grammar g = must_parse(kJson);
  check(matches(g, "{}"), "json empty object");
  check(matches(g, "{\"a\":1}"), "json single pair");
  check(matches(g, "{\"a\":[1,2,3],\"b\":{\"c\":true}}"), "json nested");
  check(matches(g, "{\"s\":\"hi\\n\"}"), "json string escape");
  check(matches(g, "{\n  \"x\" : null\n}\n"), "json inner and trailing whitespace");
  check(rejected(g, "{\"a\":\"x\ny\"}"), "json rejects raw newline in string");
  check(rejected(g, "{a:1}"), "json rejects unquoted key");
  check(rejected(g, "[1,2]"), "json root is object only");
  check(!matches(g, "{\"a\":1"), "json missing brace incomplete");
  check(rejected(g, "{\"a\":1}x"), "json rejects trailing garbage");
}

void test_json_completion_suffix() {
  Grammar g = must_parse(kJson);
  State st = g.start();
  auto after = g.feed(st, "{\"a\":1");
  check(after.has_value(), "json partial feeds");
  if (after) {
    check(!g.complete(*after), "json partial not complete");
    auto suffix = g.complete_suffix(*after);
    check(suffix.has_value(), "json partial has a completion");
    if (suffix) {
      auto whole = g.feed(*after, *suffix);
      check(whole.has_value() && g.complete(*whole), "completion drives to complete");
      std::string full = std::string("{\"a\":1") + *suffix;
      check(matches(g, full), "forced-closed json is accepted end to end");
    }
  }
  auto empty = g.complete_suffix(*g.feed(g.start(), "{}"));
  check(empty.has_value() && empty->empty(), "already complete gives empty suffix");
}

void test_matcher_logit_mask() {
  Grammar g = must_parse("root ::= \"a\" \"b\"");
  std::vector<std::string> toks = {"a", "b", "c"};
  Matcher m(std::move(g), toks, /*eos=*/-1);
  std::vector<float> logits = {1.0f, 1.0f, 1.0f};
  m.mask(logits);
  const float ninf = -std::numeric_limits<float>::infinity();
  check(logits[0] == 1.0f, "mask keeps a at start");
  check(logits[1] == ninf, "mask rejects b at start");
  check(logits[2] == ninf, "mask rejects c at start");

  m.accept(0);
  std::vector<float> logits2 = {1.0f, 1.0f, 1.0f};
  m.mask(logits2);
  check(logits2[0] == ninf, "mask rejects a after a");
  check(logits2[1] == 1.0f, "mask keeps b after a");

  m.accept(1);
  check(m.complete(), "matcher complete after ab");
}

void test_parse_errors() {
  check(!Grammar::parse("value ::= \"x\"").has_value(), "missing root is an error");
  check(!Grammar::parse("root ::= undefined_rule").has_value(), "undefined rule is an error");
  check(!Grammar::parse("root := \"x\"").has_value(), "missing ::= is an error");
  check(!Grammar::parse("root ::= [a").has_value(), "unterminated class is an error");
  check(!Grammar::parse("root ::= *").has_value(), "leading repetition is an error");
  check(!Grammar::parse("root ::= root \"a\" | \"a\"").has_value(),
        "direct left recursion is an error");
  check(!Grammar::parse("root ::= a \"z\"\na ::= root | \"a\"").has_value(),
        "indirect left recursion is an error");
  check(!Grammar::parse("root ::= n root | \"a\"\nn ::= \"b\"?").has_value(),
        "left recursion through a nullable prefix is an error");

  std::string deep = "root ::= ";
  deep.append(600, '(');
  deep += "\"a\"";
  deep.append(600, ')');
  check(!Grammar::parse(deep).has_value(), "excessive group nesting is an error");

  check(Grammar::parse("root ::= \"a\" root | \"a\"").has_value(), "right recursion is accepted");
  check(matches(must_parse("root ::= \"a\" root | \"a\""), "aaa"),
        "right recursion still generates");
}

}  // namespace

int main() {
  test_literal_matching();
  test_alternation();
  test_char_range_plus();
  test_optional();
  test_negated_char_class();
  test_partial_utf8_across_feeds();
  test_json_grammar();
  test_json_completion_suffix();
  test_matcher_logit_mask();
  test_parse_errors();

  std::printf("%s (%d failures)\n", g_failures == 0 ? "ALL PASS" : "FAILURES", g_failures);
  return g_failures == 0 ? 0 : 1;
}
