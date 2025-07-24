// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdlib>
#include <fstream>
#include <string>
#include <cstdio>
#include <unistd.h>
#include <limits.h>
#include <libgen.h>
#include <stdexcept>
#include <sys/wait.h>
#include <filesystem>
#include <iostream>
#include "debug_tools_fixture.hpp"
#include <fmt/base.h>
#include <tt-metalium/host_api.hpp>
#include <functional>
#include <variant>
#include <vector>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "debug_tools_test_utils.hpp"
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/utils.hpp>

using namespace tt;
using namespace tt::tt_metal;

// Helper to get the directory of the currently running executable
std::string get_executable_dir() {
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    if (count == -1) throw std::runtime_error("Failed to read /proc/self/exe");
    result[count] = '\0';
    char* dirc = strdup(result);
    std::string dir = dirname(dirc);
    free(dirc);
    return dir;
}

// Helper to get TT_METAL_HOME from the environment
std::string get_tt_metal_home() {
    const char* env = std::getenv("TT_METAL_HOME");
    if (!env) throw std::runtime_error("TT_METAL_HOME is not set");
    return std::string(env);
}

// Helper to find watcher_dump executable robustly
std::string find_watcher_dump(const std::string& tools_dir) {
    namespace fs = std::filesystem;
    fs::path tools_path(tools_dir);

    // Check directly in tools/
    fs::path candidate = tools_path / "watcher_dump";
    if (fs::exists(candidate) && fs::is_regular_file(candidate)) {
        return candidate.string();
    }

    // Check in immediate subdirectories (e.g., RelWithDebInfo, Debug, Release)
    for (const auto& entry : fs::directory_iterator(tools_path)) {
        if (entry.is_directory()) {
            fs::path sub_candidate = entry.path() / "watcher_dump";
            if (fs::exists(sub_candidate) && fs::is_regular_file(sub_candidate)) {
                return sub_candidate.string();
            }
        }
    }

    throw std::runtime_error("Could not find watcher_dump in " + tools_dir + " or its immediate subdirectories.");
}
