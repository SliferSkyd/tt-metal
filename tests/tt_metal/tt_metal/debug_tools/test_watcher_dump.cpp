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
#include <filesystem>

// Helper to get the directory of the currently running executable
// This test file assumes that the unit_tests_debug_tools executable is in the same directory as this file
// and that the watcher_dump executable is in the tools/ directory
// if any of these assumption are no longer true, this test will need to be updated
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

// Helper to find watcher_dump in tools/ or its immediate subdirectories (e.g., RelWithDebInfo)
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

// Integration test replicating tests/scripts/run_tools_tests.sh (up to watcher dump tool tests)
TEST(ToolsIntegration, WatcherDumpToolWorkflow) {
    // Compute paths
    std::string exe_dir = get_executable_dir();
    std::string unit_tests_path = exe_dir + "/unit_tests_debug_tools";
    // Use a robust search for watcher_dump to handle possible RelWithDebInfo or other subdirs
    std::string watcher_dump_path = find_watcher_dump(std::string(BUILD_ROOT_DIR) + "/tools");
    std::string watcher_log_path = get_tt_metal_home() + "/generated/watcher/watcher.log";

    // 1. Run a test that populates basic fields but not watcher fields
    setenv("TT_METAL_WATCHER_KEEP_ERRORS", "1", 1);
    ASSERT_EQ(std::system((unit_tests_path + " --gtest_filter=*PrintHanging").c_str()), 0) << "PrintHanging test failed";

    // 2. Run dump tool w/ minimum data - no error expected.
    ASSERT_EQ(std::system((watcher_dump_path + " -d=0 -w -c").c_str()), 0) << "watcher_dump minimal failed";

    // 3. Verify the kernel we ran shows up in the log.
    {
        std::ifstream log(watcher_log_path);
        ASSERT_TRUE(log.is_open()) << "Could not open watcher.log";
        std::string line;
        bool found = false;
        while (std::getline(log, line)) {
            if (line.find("tests/tt_metal/tt_metal/test_kernels/misc/print_hang.cpp") != std::string::npos) {
                found = true;
                break;
            }
        }
        ASSERT_TRUE(found) << "Expected kernel string not found in watcher log";
    }

    // Interlude: Change to TT_METAL_HOME directory
    // This is necessary because for some reason some test inside the unit_tests_debug_tools executable
    // That uses the WatcherAssertBrisc flag will fail if the current working directory is not TT_METAL_HOME (WatcherFixture.TestWatcherRingBufferBrisc I believe)
    // This is a workaround to ensure that the test can run successfully
    // TODO: Remove this once we have a better solution
    std::string tt_metal_home = get_tt_metal_home();
    if (chdir(tt_metal_home.c_str()) != 0) {
        throw std::runtime_error("Failed to change to TT_METAL_HOME directory: " + tt_metal_home);
    }

    // 4. Now run with all watcher features, expect it to throw.
    setenv("TT_METAL_WATCHER_KEEP_ERRORS", "1", 1);
    ASSERT_EQ(std::system((unit_tests_path + " --gtest_filter=*WatcherAssertBrisc").c_str()), 0) << "WatcherAssertBrisc test failed";
    int ret = std::system((watcher_dump_path + " -d=0 -w > tmp.log 2>&1").c_str());
    // watcher_dump is expected to fail (nonzero exit), so don't assert on ret

    // 5. Verify the error we expect showed up in the program output.
    {
        std::ifstream log("tmp.log");
        ASSERT_TRUE(log.is_open()) << "Could not open tmp.log";
        std::string line;
        bool found = false;
        while (std::getline(log, line)) {
            if (line.find("brisc tripped an assert") != std::string::npos) {
                found = true;
                break;
            }
        }
        ASSERT_TRUE(found) << "Expected error string not found in tmp.log";
    }

    // 6. Check that stack dumping is working
    ASSERT_EQ(std::system((unit_tests_path + " --gtest_filter=*TestWatcherRingBufferBrisc").c_str()), 0) << "TestWatcherRingBufferBrisc test failed";
    ASSERT_EQ(std::system((watcher_dump_path + " -d=0 -w").c_str()), 0) << "watcher_dump for stack usage failed";
    {
        std::ifstream log(watcher_log_path);
        ASSERT_TRUE(log.is_open()) << "Could not open watcher.log (stack usage)";
        std::string line;
        bool found = false;
        while (std::getline(log, line)) {
            if (line.find("brisc highest stack usage:") != std::string::npos) {
                found = true;
                break;
            }
        }
        ASSERT_TRUE(found) << "Expected stack usage string not found in watcher log";
    }

    // 7. Remove created files (cleanup)
    std::remove("tmp.log");
    std::remove(watcher_log_path.c_str());
    std::string watcher_cq_dump_dir = get_tt_metal_home() + "/generated/watcher/command_queue_dump/*";
    std::system(("rm -f " + watcher_cq_dump_dir).c_str());
}
