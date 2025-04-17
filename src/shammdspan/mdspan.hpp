
#if defined(__cpp_lib_span)
    #include <span>
#else
    #include <cstddef>
    #include <limits>
#endif

namespace sham::mdspan {
#if defined(__cpp_lib_span)
    using std::dynamic_extent;
#else
    constexpr auto dynamic_extent = std::numeric_limits<size_t>::max();
#endif
} // namespace sham::mdspan
