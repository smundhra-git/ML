# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.30.1/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.30.1/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/shlokmundhra/Documents/proj/ML_Library/Shlok_ML

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/shlokmundhra/Documents/proj/ML_Library/Shlok_ML/build

# Include any dependencies generated for this target.
include CMakeFiles/test_ElasticNetRegression.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/test_ElasticNetRegression.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/test_ElasticNetRegression.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_ElasticNetRegression.dir/flags.make

CMakeFiles/test_ElasticNetRegression.dir/tests/regression/test_ElasticNetRegression.cpp.o: CMakeFiles/test_ElasticNetRegression.dir/flags.make
CMakeFiles/test_ElasticNetRegression.dir/tests/regression/test_ElasticNetRegression.cpp.o: /Users/shlokmundhra/Documents/proj/ML_Library/Shlok_ML/tests/regression/test_ElasticNetRegression.cpp
CMakeFiles/test_ElasticNetRegression.dir/tests/regression/test_ElasticNetRegression.cpp.o: CMakeFiles/test_ElasticNetRegression.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/shlokmundhra/Documents/proj/ML_Library/Shlok_ML/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_ElasticNetRegression.dir/tests/regression/test_ElasticNetRegression.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/test_ElasticNetRegression.dir/tests/regression/test_ElasticNetRegression.cpp.o -MF CMakeFiles/test_ElasticNetRegression.dir/tests/regression/test_ElasticNetRegression.cpp.o.d -o CMakeFiles/test_ElasticNetRegression.dir/tests/regression/test_ElasticNetRegression.cpp.o -c /Users/shlokmundhra/Documents/proj/ML_Library/Shlok_ML/tests/regression/test_ElasticNetRegression.cpp

CMakeFiles/test_ElasticNetRegression.dir/tests/regression/test_ElasticNetRegression.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/test_ElasticNetRegression.dir/tests/regression/test_ElasticNetRegression.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/shlokmundhra/Documents/proj/ML_Library/Shlok_ML/tests/regression/test_ElasticNetRegression.cpp > CMakeFiles/test_ElasticNetRegression.dir/tests/regression/test_ElasticNetRegression.cpp.i

CMakeFiles/test_ElasticNetRegression.dir/tests/regression/test_ElasticNetRegression.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/test_ElasticNetRegression.dir/tests/regression/test_ElasticNetRegression.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/shlokmundhra/Documents/proj/ML_Library/Shlok_ML/tests/regression/test_ElasticNetRegression.cpp -o CMakeFiles/test_ElasticNetRegression.dir/tests/regression/test_ElasticNetRegression.cpp.s

# Object files for target test_ElasticNetRegression
test_ElasticNetRegression_OBJECTS = \
"CMakeFiles/test_ElasticNetRegression.dir/tests/regression/test_ElasticNetRegression.cpp.o"

# External object files for target test_ElasticNetRegression
test_ElasticNetRegression_EXTERNAL_OBJECTS =

test_ElasticNetRegression: CMakeFiles/test_ElasticNetRegression.dir/tests/regression/test_ElasticNetRegression.cpp.o
test_ElasticNetRegression: CMakeFiles/test_ElasticNetRegression.dir/build.make
test_ElasticNetRegression: libregression_ElasticNetRegression.a
test_ElasticNetRegression: CMakeFiles/test_ElasticNetRegression.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/shlokmundhra/Documents/proj/ML_Library/Shlok_ML/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_ElasticNetRegression"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_ElasticNetRegression.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_ElasticNetRegression.dir/build: test_ElasticNetRegression
.PHONY : CMakeFiles/test_ElasticNetRegression.dir/build

CMakeFiles/test_ElasticNetRegression.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_ElasticNetRegression.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_ElasticNetRegression.dir/clean

CMakeFiles/test_ElasticNetRegression.dir/depend:
	cd /Users/shlokmundhra/Documents/proj/ML_Library/Shlok_ML/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/shlokmundhra/Documents/proj/ML_Library/Shlok_ML /Users/shlokmundhra/Documents/proj/ML_Library/Shlok_ML /Users/shlokmundhra/Documents/proj/ML_Library/Shlok_ML/build /Users/shlokmundhra/Documents/proj/ML_Library/Shlok_ML/build /Users/shlokmundhra/Documents/proj/ML_Library/Shlok_ML/build/CMakeFiles/test_ElasticNetRegression.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/test_ElasticNetRegression.dir/depend

