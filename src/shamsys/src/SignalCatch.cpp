// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SignalCatch.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/exception.hpp"
#include "shambase/popen.hpp"
#include "shambase/stacktrace.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include <csignal>
#include <stdexcept>

/*
feature test for strsignal()

strsignal():
    Since glibc 2.10:
    _XOPEN_SOURCE >= 700 || _POSIX_C_SOURCE >= 200809L
    Before glibc 2.10:
    _GNU_SOURCE

*/
#if defined(_XOPEN_SOURCE) && defined(_POSIX_C_SOURCE) && _XOPEN_SOURCE >= 700                     \
    && _POSIX_C_SOURCE >= 200809L
    #define HAVE_STRSIGNAL
#elif defined(_GNU_SOURCE)
    #define HAVE_STRSIGNAL
#endif

#if __has_include(<dlfcn.h>)
    #include <dlfcn.h> // dladdr
    #define HAVE_DLFCN_H 1
#endif

#ifndef _WIN32
    #if defined(__GLIBC__) || defined(__APPLE__)
        #include <execinfo.h> // backtrace and backtrace_symbols
    #endif
    #include <unistd.h> // getpid
#endif

namespace {

    struct resolve_addr_result_t {
        const char *file_name;
        uintptr_t offset;
    };

    // resolve address to file name and offset, return nullptr if not found or if HAVE_DLFCN_H is
    // not defined
    inline resolve_addr_result_t resolve_addr(void *addr_global) {
#if HAVE_DLFCN_H
        Dl_info info;
        if (dladdr(addr_global, &info) && info.dli_fname) {
            uintptr_t addr   = reinterpret_cast<uintptr_t>(addr_global);
            uintptr_t base   = reinterpret_cast<uintptr_t>(info.dli_fbase);
            uintptr_t offset = addr - base;
            return {info.dli_fname, offset};
        }
#endif
        return {nullptr, 0};
    }
} // namespace

namespace {

    struct stackframe_t {
        static constexpr int nbuf = 64;
        void *bt_buffer[nbuf];
        int nentries;
        char **symbols;

        bool is_valid() const { return nentries > 0 && symbols != nullptr; }

        ~stackframe_t() {
            if (symbols != nullptr) {
                std::free(symbols);
            }
        }
    };

    // call backtrace and backtrace_symbols
    inline stackframe_t get_stacktrace() {
        stackframe_t st{};
        st.nentries = backtrace(st.bt_buffer, stackframe_t::nbuf);
        st.symbols  = backtrace_symbols(st.bt_buffer, st.nentries);
        return st;
    }

    inline u64 count_lines(const std::string &s) {
        u64 count = 0;
        for (auto c : s) {
            if (c == '\n') {
                count++;
            }
        }
        return count;
    }

    struct addr2line_result_t {
        struct entry_t {
            std::string function_name;
            std::string path;
            bool inlined = false;
        };
        std::vector<entry_t> entries = {};
        bool success                 = false;
    };

    inline std::vector<addr2line_result_t::entry_t> parse_addr2line_output(
        const std::string &output) {
        std::vector<addr2line_result_t::entry_t> entries = {};
        std::stringstream ss(output);

        addr2line_result_t::entry_t tmp;
        u64 line_count = 0;
        for (std::string line; std::getline(ss, line);) {

            if (line_count % 2 == 0) {
                auto inlined_pos = line.find("inlined at ");
                tmp.inlined      = inlined_pos != std::string::npos;
                if (tmp.inlined) {
                    tmp.function_name = line.substr(0, inlined_pos);
                } else {
                    tmp.function_name = line;
                }
            } else {
                tmp.path = line;
                entries.push_back(tmp);
            }

            line_count++;
        }
        return entries;
    }

    inline addr2line_result_t call_addr2line(u64 pid, void *addr) {
        addr2line_result_t result{};
        std::string command = fmt::format("eu-addr2line -C -f -i -p {} {}", pid, addr);
        std::string output = shambase::popen_fetch_output_noexcept(command.c_str(), result.success);
        if (result.success) {
            result.entries = parse_addr2line_output(output);
            result.success = true;
        }
        return result;
    }

    inline void sanitize_addr2line_output(std::string &output) {
        shambase::replace_all(output, "std::__cxx11", "std::");
        shambase::replace_all(
            output,
            "hipsycl::sycl::vec<double, 3, hipsycl::sycl::detail::vec_storage<double, 3> >",
            "f64_3");
        shambase::replace_all(
            output,
            "hipsycl::sycl::vec<long, 3, hipsycl::sycl::detail::vec_storage<long, 3> >",
            "i64_3");
        shambase::replace_all(
            output,
            "std::::basic_string<char, std::char_traits<char>, std::allocator<char> >",
            "std::string");
    }

    inline void sanitize_addr2line_output(addr2line_result_t &output) {
        for (auto &entry : output.entries) {
            sanitize_addr2line_output(entry.function_name);
        }
    }

    inline void print_stacktrace(std::stringstream &ss) {
        stackframe_t stacktrace = get_stacktrace();

        if (!stacktrace.is_valid()) {
            ss << "  Failed to get stacktrace\n";
            return;
        }

        // get pid
        pid_t pid = getpid();

        for (int i = 0; i < stacktrace.nentries; i++) {
            resolve_addr_result_t result = resolve_addr(stacktrace.bt_buffer[i]);

            auto fallback_print = [&]() {
                ss << "  " << i << " : " << stacktrace.symbols[i] << "\n";
            };

            if (result.file_name == nullptr) {

                fallback_print();

            } else {

                std::string reset  = shambase::term_colors::reset();
                std::string cyan   = shambase::term_colors::col8b_cyan();
                std::string orange = shambase::term_colors::col8b_yellow();

                addr2line_result_t addr2line_result = call_addr2line(pid, stacktrace.bt_buffer[i]);
                if (addr2line_result.success) {
                    sanitize_addr2line_output(addr2line_result);
                    for (auto &entry : addr2line_result.entries) {
                        if (entry.inlined) {
                            ss << "  " << i << " : " << "     (inlined)" << " in " << cyan
                               << entry.function_name << reset << " at " << orange << entry.path
                               << reset << "\n";
                        } else {
                            ss << "  " << i << " : " << stacktrace.bt_buffer[i] << " in " << cyan
                               << entry.function_name << reset << " at " << orange << entry.path
                               << reset << "\n";
                        }
                    }
                } else {
                    fallback_print();
                }
            }
        }
    }
} // namespace

namespace shamsys::details {

    /**
     * @brief Name the received signal
     * @note List made from <bits/signum-arch.h> and <bits/signum-generic.h>.
     * @param signum The signal number
     * @return const char* The name of the signal
     */
    const char *get_signal_name(int signum) {
#ifdef HAVE_STRSIGNAL
        const char *signame = strsignal(signum);
#else
        const char *signame = "UNKNOWN";
#endif
        return signame;
    }

    void signal_callback_handler(int signum) {

        // ensure that we print in one block to avoid interleaving
        std::stringstream ss;
        ss << fmt::format(
            "!!! Received signal : {} (code {}) from world rank {}\n"
            "Estimated stacktrace : \n"
            "{}\n",
            get_signal_name(signum),
            signum,
            shamcomm::world_rank(),
            shambase::fmt_callstack());

        ss << "Actual stacktrace : \n";
        print_stacktrace(ss);

        ss << "exiting ..." << std::endl;

        std::cout << ss.str();

        // raise signal again since the handler was reset to the default (see SA_RESETHAND)
        raise(signum);
    }
} // namespace shamsys::details

namespace shamsys {
    void register_signals() {
        struct sigaction sa = {};

        sa.sa_handler = details::signal_callback_handler;
        sigemptyset(&sa.sa_mask);
        // SA_RESETHAND resets the signal action to the default before calling the handler.
        sa.sa_flags = SA_RESETHAND;

        std::array catched_signals = {SIGTERM, SIGINT, SIGSEGV, SIGIOT};
        for (auto signum : catched_signals) {
            if (sigaction(signum, &sa, NULL) != 0) {
                shambase::throw_with_loc<std::runtime_error>(fmt::format(
                    "Failed to register {} signal handler", details::get_signal_name(signum)));
            }
        }
    }
} // namespace shamsys
