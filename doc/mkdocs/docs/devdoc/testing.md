# Testing

Shamrock uses a custom testing framework to handle unit tests. This guide explains how to write and run them.

## Running Unit Tests

Shamrock's unit tests are not run using `ctest`. Instead, they are run directly through the `shamrock_test` executable.

Follow these steps to run the tests:

1.  **Build the project:** From the project root, run the `shammake` command to compile the code, including your new test.
    ```bash
    shammake
    ```
2.  **Run the tests:** Navigate to your build directory and execute the test runner:
    ```bash
    cd build
    ./shamrock_test --sycl-cfg 0:0 --unittest
    mpirun -n 2 ./shamrock_test --sycl-cfg 0:0 --unittest
    mpirun -n 3 ./shamrock_test --sycl-cfg 0:0 --unittest
    mpirun -n 4 ./shamrock_test --sycl-cfg 0:0 --unittest
    ```

The `--unittest` flag tells the executable to run all registered unit tests.

## Writing a Unit Test

To add a new unit test, you first need to create a new `.cpp` file inside the `src/tests` directory. It is good practice to mirror the source directory structure. For example, a test for a feature in `src/shammath` should be placed in `src/tests/shammath`.

The build system will automatically discover any `.cpp` file you add to this directory or its subdirectories.

Here is a basic template for a test file:

```cpp
// Include the header for the code you want to test
#include "shammath/my_component.hpp" // Example include

// Include the shamtest header
#include "shamtest/shamtest.hpp"

// Use the TestStart macro to define your test
TestStart(Unittest, "testname", testfuncname, 1) {
    shamtest::asserts().assert_bool("what a reliable test", true);
}
```

The `TestStart` macro registers the test. 
You can find available assertion macros like `REQUIRE_EQUAL` in `src/shamtest/shamtest.hpp`.
