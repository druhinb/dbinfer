#ifndef DBINFER_GRAMMAR_GRAMMAR_HPP
#define DBINFER_GRAMMAR_GRAMMAR_HPP

#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace dbinfer::grammar {

// GBNF element, matching llama.cpp's rule encoding. a rule is a flat sequence
// of elements terminated by End, with Alt separating its alternates. Char and
// CharNot open a character set that RngUpper and CharAlt extend.
enum class ElemType : std::uint8_t {
  End,       // terminates an alternate
  Alt,       // separates alternates within a rule
  RuleRef,   // value is a rule index
  Char,      // value is a codepoint, opens a positive set
  CharNot,   // value is a codepoint, opens a negated set
  RngUpper,  // value is the inclusive upper bound of the preceding char's range
  CharAlt,   // value is another codepoint in the open set
};

struct Element {
  ElemType type;
  std::uint32_t value;
};

struct Error {
  std::string message;
};

// carries a UTF-8 sequence split across token boundaries. n_remain > 0 means
// value holds the high bits of an unfinished codepoint; -1 marks invalid bytes.
struct PartialUtf8 {
  std::uint32_t value = 0;
  int n_remain = 0;
};

using Stack = std::vector<const Element*>;

// automaton position: a set of stacks whose tops are the char elements that may
// match next. an empty stack means the root derivation may terminate here.
struct State {
  std::vector<Stack> stacks;
  PartialUtf8 partial;
};

class Grammar {
 public:
  static std::expected<Grammar, Error> parse(std::string_view text);

  State start() const;
  // true when a stack is empty, so the grammar may accept end-of-input here.
  bool complete(const State& s) const;
  // advances s by every codepoint in bytes; nullopt when bytes are not an
  // accepted continuation. a trailing partial codepoint is carried in the result.
  std::optional<State> feed(const State& s, std::string_view bytes) const;
  // shortest byte string that drives s to a complete state, empty when s is
  // already complete. nullopt when no completion exists within the search bound.
  std::optional<std::string> complete_suffix(const State& s) const;

 private:
  std::vector<std::vector<Element>> rules_;
  std::uint32_t root_ = 0;

  void advance_stack(const Stack& stack, std::vector<Stack>& out) const;
  std::vector<Stack> accept(const std::vector<Stack>& stacks, std::uint32_t chr) const;
};

// per-step logit mask over a token vocabulary. rejected tokens are driven to
// -inf; the chosen token then advances the automaton. token bytes are supplied
// by the caller so this stays independent of any tokenizer.
class Matcher {
 public:
  Matcher(Grammar grammar, std::vector<std::string> token_bytes, std::int32_t eos_id);

  void mask(std::span<float> logits) const;
  void accept(std::int32_t token);
  bool complete() const;

 private:
  Grammar grammar_;
  std::vector<std::string> token_bytes_;
  std::int32_t eos_id_;
  State state_;
};

}  // namespace dbinfer::grammar

#endif  // DBINFER_GRAMMAR_GRAMMAR_HPP
