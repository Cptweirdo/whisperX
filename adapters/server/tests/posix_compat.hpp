// POSIX shims for the Windows test build: the MSVC CRT has no setenv/unsetenv
// (only _putenv_s, where an empty value removes the variable), and getpid lives
// in <process.h> as _getpid. Elsewhere this is just <unistd.h>.
#pragma once

#if defined(_WIN32)
#include <process.h>

#include <cstdlib>

inline int setenv(const char* key, const char* value, int overwrite) {
    if (!overwrite && std::getenv(key) != nullptr) return 0;
    return _putenv_s(key, value);
}

inline int unsetenv(const char* key) { return _putenv_s(key, ""); }

#ifndef getpid
#define getpid _getpid
#endif
#else
#include <unistd.h>
#endif
