#ifndef ARCAE_FINALLY_H
#define ARCAE_FINALLY_H

#include <utility>

namespace arcae {
namespace detail {

template <typename Fn>
struct Finally {
  Fn fn;
  bool enabled;
  ~Finally() {
    if (enabled) fn();
  }
};

template <typename Fn>
auto finally(Fn&& fn) {
  return Finally<Fn>{std::forward<Fn>(fn), true};
}

}  // namespace detail
}  // namespace arcae

#endif  // #define ARCAE_FINALLY_H
