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
include CMakeFiles/Shlok_ML.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/Shlok_ML.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/Shlok_ML.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Shlok_ML.dir/flags.make

CMakeFiles/Shlok_ML.dir/python/Shlok_ML.cpp.o: CMakeFiles/Shlok_ML.dir/flags.make
CMakeFiles/Shlok_ML.dir/python/Shlok_ML.cpp.o: /Users/shlokmundhra/Documents/proj/ML_Library/Shlok_ML/python/Shlok_ML.cpp
CMakeFiles/Shlok_ML.dir/python/Shlok_ML.cpp.o: CMakeFiles/Shlok_ML.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/shlokmundhra/Documents/proj/ML_Library/Shlok_ML/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Shlok_ML.dir/python/Shlok_ML.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Shlok_ML.dir/python/Shlok_ML.cpp.o -MF CMakeFiles/Shlok_ML.dir/python/Shlok_ML.cpp.o.d -o CMakeFiles/Shlok_ML.dir/python/Shlok_ML.cpp.o -c /Users/shlokmundhra/Documents/proj/ML_Library/Shlok_ML/python/Shlok_ML.cpp

CMakeFiles/Shlok_ML.dir/python/Shlok_ML.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/Shlok_ML.dir/python/Shlok_ML.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/shlokmundhra/Documents/proj/ML_Library/Shlok_ML/python/Shlok_ML.cpp > CMakeFiles/Shlok_ML.dir/python/Shlok_ML.cpp.i

CMakeFiles/Shlok_ML.dir/python/Shlok_ML.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/Shlok_ML.dir/python/Shlok_ML.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/shlokmundhra/Documents/proj/ML_Library/Shlok_ML/python/Shlok_ML.cpp -o CMakeFiles/Shlok_ML.dir/python/Shlok_ML.cpp.s

# Object files for target Shlok_ML
Shlok_ML_OBJECTS = \
"CMakeFiles/Shlok_ML.dir/python/Shlok_ML.cpp.o"

# External object files for target Shlok_ML
Shlok_ML_EXTERNAL_OBJECTS =

Shlok_ML.cpython-313-darwin.so: CMakeFiles/Shlok_ML.dir/python/Shlok_ML.cpp.o
Shlok_ML.cpython-313-darwin.so: CMakeFiles/Shlok_ML.dir/build.make
Shlok_ML.cpython-313-darwin.so: /opt/homebrew/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/lib/libpython3.13.dylib
Shlok_ML.cpython-313-darwin.so: CMakeFiles/Shlok_ML.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/shlokmundhra/Documents/proj/ML_Library/Shlok_ML/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module Shlok_ML.cpython-313-darwin.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Shlok_ML.dir/link.txt --verbose=$(VERBOSE)
	/Library/Developer/CommandLineTools/usr/bin/strip -x /Users/shlokmundhra/Documents/proj/ML_Library/Shlok_ML/build/Shlok_ML.cpython-313-darwin.so

# Rule to build all files generated by this target.
CMakeFiles/Shlok_ML.dir/build: Shlok_ML.cpython-313-darwin.so
.PHONY : CMakeFiles/Shlok_ML.dir/build

CMakeFiles/Shlok_ML.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Shlok_ML.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Shlok_ML.dir/clean

CMakeFiles/Shlok_ML.dir/depend:
	cd /Users/shlokmundhra/Documents/proj/ML_Library/Shlok_ML/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/shlokmundhra/Documents/proj/ML_Library/Shlok_ML /Users/shlokmundhra/Documents/proj/ML_Library/Shlok_ML /Users/shlokmundhra/Documents/proj/ML_Library/Shlok_ML/build /Users/shlokmundhra/Documents/proj/ML_Library/Shlok_ML/build /Users/shlokmundhra/Documents/proj/ML_Library/Shlok_ML/build/CMakeFiles/Shlok_ML.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/Shlok_ML.dir/depend
