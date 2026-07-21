#include <algorithm>
#include <cstdint>
#include <queue>
#include <vector>

#include "tokenizer/tokenizer.hpp"
#include "tokenizer/unicode.hpp"

namespace dbinfer::tokenizer {

namespace {

constexpr std::uint32_t kOOR = 0xFFFFFFFFu;

std::uint32_t tolower_ascii(std::uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

// hand-rolled equivalent of the Qwen2/GPT2 pretokenizer regex
// ('s|'t|'re|'ve|'m|'ll|'d|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|<space>?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+),
// tried in that priority order at each position. Returns (start, len)
// codepoint ranges; each becomes one word passed to bpe_word.
std::vector<std::pair<std::size_t, std::size_t>> qwen2_split(
    const std::vector<std::uint32_t>& cpts) {
  std::vector<std::pair<std::size_t, std::size_t>> out;
  const std::size_t n = cpts.size();

  auto cpt = [&](std::size_t p) -> std::uint32_t { return p < n ? cpts[p] : kOOR; };
  auto is_l = [&](std::size_t p) { return p < n && is_letter(cpts[p]); };
  auto is_n = [&](std::size_t p) { return p < n && is_number(cpts[p]); };
  auto is_w = [&](std::size_t p) { return p < n && is_whitespace(cpts[p]); };
  auto is_other = [&](std::size_t p) { return p < n && !is_w(p) && !is_l(p) && !is_n(p); };

  std::size_t prev = 0;
  auto emit = [&](std::size_t end) {
    if (end > prev) out.emplace_back(prev, end - prev);
    prev = end;
  };

  for (std::size_t pos = 0; pos < n;) {
    const std::uint32_t c = cpt(pos);

    // (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if (c == '\'' && pos + 1 < n) {
      std::uint32_t c1 = tolower_ascii(cpt(pos + 1));
      if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
        pos += 2;
        emit(pos);
        continue;
      }
      if (pos + 2 < n) {
        std::uint32_t c2 = tolower_ascii(cpt(pos + 2));
        if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
          pos += 3;
          emit(pos);
          continue;
        }
      }
    }

    // [^\r\n\p{L}\p{N}]?\p{L}+
    if (!(c == '\r' || c == '\n' || is_n(pos))) {
      if (is_l(pos) || is_l(pos + 1)) {
        ++pos;
        while (is_l(pos)) ++pos;
        emit(pos);
        continue;
      }
    }

    // \p{N}
    if (is_n(pos)) {
      ++pos;
      emit(pos);
      continue;
    }

    // <space>?[^\s\p{L}\p{N}]+[\r\n]*
    {
      const bool lead_space = (c == ' ');
      const std::size_t probe = lead_space ? pos + 1 : pos;
      if (is_other(probe)) {
        pos += lead_space ? 1 : 0;
        while (is_other(pos)) ++pos;
        while (cpt(pos) == '\r' || cpt(pos) == '\n') ++pos;
        emit(pos);
        continue;
      }
    }

    // \s*[\r\n]+ | \s+(?!\S) | \s+
    std::size_t ws = 0;
    std::size_t last_nl = 0;
    while (is_w(pos + ws)) {
      std::uint32_t cw = cpt(pos + ws);
      if (cw == '\r' || cw == '\n') last_nl = pos + ws + 1;
      ++ws;
    }
    if (last_nl > 0) {
      pos = last_nl;
      emit(pos);
      continue;
    }
    if (ws > 1 && cpt(pos + ws) != kOOR) {
      pos += ws - 1;
      emit(pos);
      continue;
    }
    if (ws > 0) {
      pos += ws;
      emit(pos);
      continue;
    }

    ++pos;
    emit(pos);
  }

  return out;
}

// GPT2 byte-level encoding: re-encodes each codepoint's UTF-8 bytes through
// byte_to_unicode so every raw byte becomes a printable, BPE-mergeable
// codepoint (this is what lets BPE vocab entries be plain strings).
std::string byte_encode(const std::vector<std::uint32_t>& cpts, std::size_t start,
                        std::size_t len) {
  std::string s;
  for (std::size_t i = start; i < start + len; ++i) {
    std::string u = cpt_to_utf8(cpts[i]);
    for (char ch : u) s += byte_to_unicode(static_cast<std::uint8_t>(ch));
  }
  return s;
}

// one piece of a word being merged, as a doubly linked list over fixed
// storage: prev/next are indices into the symbols vector (-1 = no neighbor),
// and n == 0 marks a symbol that has been absorbed into its left neighbor.
struct Symbol {
  int prev;
  int next;
  const char* text;
  std::size_t n;
};

// a candidate merge between two still-adjacent symbols; text is the
// concatenation of both sides at the time this Bigram was queued, used to
// detect staleness if either symbol got merged into something else first.
struct Bigram {
  int left;
  int right;
  std::int32_t rank;
  std::string text;
};

// orders the priority_queue so top() is the lowest-rank (highest-priority,
// earliest-listed) merge, ties broken by leftmost position — the standard
// greedy BPE merge order.
struct BigramLess {
  bool operator()(const Bigram& l, const Bigram& r) const {
    return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
  }
};

}  // namespace

std::vector<std::string> pretokenize(std::string_view text) {
  auto cpts = utf8_to_cpts(text);
  auto ranges = qwen2_split(cpts);
  std::vector<std::string> words;
  words.reserve(ranges.size());
  for (auto [start, len] : ranges) words.push_back(byte_encode(cpts, start, len));
  return words;
}

// standard priority-queue BPE: repeatedly pop the best still-valid bigram,
// merge its two symbols in place (extending the left one, zeroing the
// right), and requeue the merge's new left/right neighbor pairs. A popped
// bigram whose symbols were already consumed, or whose concatenated text no
// longer matches (because a neighbor merged first), is skipped rather than
// applied.
void bpe_word(const std::string& word, const BpeRanks& ranks, const Vocab& vocab,
              std::vector<std::int32_t>& out) {
  if (word.empty()) return;

  std::vector<Symbol> symbols;
  int index = 0;
  std::size_t offset = 0;
  while (offset < word.size()) {
    std::size_t len =
        std::min(word.size() - offset, utf8_len(static_cast<unsigned char>(word[offset])));
    Symbol s;
    s.text = word.c_str() + offset;
    s.n = len;
    offset += len;
    s.prev = index - 1;
    s.next = offset == word.size() ? -1 : index + 1;
    ++index;
    symbols.push_back(s);
  }

  std::priority_queue<Bigram, std::vector<Bigram>, BigramLess> queue;
  auto add_bigram = [&](int left, int right) {
    if (left == -1 || right == -1) return;
    std::string l(symbols[static_cast<std::size_t>(left)].text,
                  symbols[static_cast<std::size_t>(left)].n);
    std::string r(symbols[static_cast<std::size_t>(right)].text,
                  symbols[static_cast<std::size_t>(right)].n);
    auto it = ranks.find({l, r});
    if (it == ranks.end()) return;
    queue.push(Bigram{left, right, it->second, l + r});
  };

  for (std::size_t i = 1; i < symbols.size(); ++i)
    add_bigram(static_cast<int>(i - 1), static_cast<int>(i));

  while (!queue.empty()) {
    Bigram b = queue.top();
    queue.pop();
    Symbol& ls = symbols[static_cast<std::size_t>(b.left)];
    Symbol& rs = symbols[static_cast<std::size_t>(b.right)];
    if (ls.n == 0 || rs.n == 0) continue;
    if (std::string(ls.text, ls.n) + std::string(rs.text, rs.n) != b.text) continue;

    ls.n += rs.n;
    rs.n = 0;
    ls.next = rs.next;
    if (rs.next >= 0) symbols[static_cast<std::size_t>(rs.next)].prev = b.left;

    add_bigram(ls.prev, b.left);
    add_bigram(b.left, ls.next);
  }

  for (const Symbol& s : symbols) {
    if (s.n == 0) continue;
    std::string str(s.text, s.n);
    auto it = vocab.find(str);
    if (it != vocab.end()) {
      out.push_back(it->second);
    } else {
      for (char ch : str) {
        auto bit = vocab.find(std::string(1, ch));
        if (bit != vocab.end()) out.push_back(bit->second);
      }
    }
  }
}

}  // namespace dbinfer::tokenizer
