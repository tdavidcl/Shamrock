# Testing

Shamrock uses a custom testing framework to handle unit tests. This guide explains how to write and run them.

## Running Unit Tests

After writing your test, you can run it using the following steps.

1.  **Build the project:** From the project root, run the `shammake` command to compile the code, including your new test.
    ```bash
    # Activate the workspace
    source activate
    # Build the project
    shammake
    ```

2.  **Run the tests:** Navigate to your build directory and execute the test runner.
    ```bash
    # Navigate to the build directory
    cd build
    # Run the tests
    ./shamrock_test --sycl-cfg 0:0 --unittest
    ```
    The `--unittest` flag tells the executable to run all registered unit tests.

    You can also run tests with MPI to check tests enabled only when using multiple processes. Here are some examples:
    ```bash
    # Run with 2 MPI processes
    mpirun -n 2 ./shamrock_test --sycl-cfg 0:0 --unittest

    # Run with 4 MPI processes
    mpirun -n 4 ./shamrock_test --sycl-cfg 0:0 --unittest
    ```

## Test Categories

Shamrock supports different types of tests:

- **Unittest**: Fast, focused tests for individual components
- **ValidationTest**: Tests that validate correctness against known results
- **LongValidationTest**: Extended validation tests that take longer to run
- **Benchmark**: Performance measurement tests
- **LongBenchmark**: Extended performance tests

To run specific test categories:
```bash
# Run only unit tests
./shamrock_test --sycl-cfg 0:0 --unittest

# Run only validation tests
./shamrock_test --sycl-cfg 0:0 --validation

# Run only benchmarks
./shamrock_test --sycl-cfg 0:0 --benchmark

# Run long tests (includes LongValidationTest)
./shamrock_test --sycl-cfg 0:0 --long-test --validation

# Run long tests (includes LongBenchmark)
./shamrock_test --sycl-cfg 0:0 --long-test --benchmark
```

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
TestStart(
    Unittest, // Test suite (usually Unittest)
    "shammath/my_component/my_first_test", // Unique name for the test
    test_my_first_component, // Unique identifier for this test block
    1) // Run only with this number of MPI ranks (-1 means always)
{
    // Set up your test data
    int expected_value = 42;
    int actual_value = my_component_function(); // Call the function you're testing

    // Use an assertion to check the result
    REQUIRE_EQUAL(expected_value, actual_value);
}
```

The `TestStart` macro registers the test. You can find other assertion macros (like `REQUIRE_EQUAL`, `REQUIRE_FLOAT_EQUAL`, etc.) in `src/shamtest/shamtest.hpp`.

## Available Assertions

### Basic Assertions
```cpp
REQUIRE(condition);                    // Basic boolean assertion
REQUIRE_NAMED("name", condition);      // Named boolean assertion
```

### Equality Assertions
```cpp
REQUIRE_EQUAL(a, b);                   // Check if a == b
REQUIRE_EQUAL_NAMED("name", a, b);     // Named equality check
REQUIRE_EQUAL_CUSTOM_COMP(a, b, comp); // Custom comparison function
```

### Floating Point Assertions
```cpp
REQUIRE_FLOAT_EQUAL(a, b, tolerance);  // Float equality with tolerance
REQUIRE_FLOAT_EQUAL_NAMED("name", a, b, tolerance);
REQUIRE_FLOAT_EQUAL_CUSTOM_DIST(a, b, tolerance, distance_func);
```

### Exception Assertions
```cpp
REQUIRE_EXCEPTION_THROW(function_call(), exception_type);
```

### LaTeX Report Generation
Tests can generate LaTeX output:
```cpp
TEX_REPORT(R"==(
  \section{Test Results}
  The test completed successfully with $N = 1000$ particles.
)==");
```

### Python Integration
Tests can execute Python scripts:
```cpp
#include "shamtest/PyScriptHandle.hpp"

PyScriptHandle handle;
handle.exec(R"(
    import matplotlib.pyplot as plt
    plt.plot(x, y)
    plt.savefig("test_result.png")
)");
```

## Debugging Tests

### Common Issues

**Test is skipped due to wrong number of nodes:**
- Tests are automatically skipped when the MPI rank count doesn't match the test's node count parameter (-1 will always be run)
- Check the test output to see which tests were skipped

**SYCL device issues:**
- Check available devices: `./shamrock_test --smi`
- Try different device configurations: `--sycl-cfg 0:0` or `--sycl-cfg 1:1`

**Test discovery issues:**
- Ensure your test file is in the correct `src/tests/` subdirectory

### Debugging Commands

```bash
# List all available tests
./shamrock_test --sycl-cfg 0:0 --test-list

# Run with verbose output
./shamrock_test --sycl-cfg 0:0 --unittest --full-output

# Run only specific tests
./shamrock_test --sycl-cfg 0:0 --unittest --run-only "test_name_pattern"
```

## Continuous Integration

Shamrock uses comprehensive CI/CD pipelines that run tests on multiple:
- Compilers (Clang, GCC, Intel)
- Platforms (Linux, macOS, Windows)
- Backends (OpenMP, CUDA, HIP)
- Configurations (Debug, Release, Coverage)

### Local CI testing
To run tests similar to CI locally:
```bash
source ./activate && shammake
./shamrock_test --sycl-cfg 0:0
mpirun -n 2 ./shamrock_test --sycl-cfg 0:0
mpirun -n 3 ./shamrock_test --sycl-cfg 0:0
mpirun -n 4 ./shamrock_test --sycl-cfg 0:0
```

### Coverage Reports
Coverage reports are generated automatically and available as artifacts in CI runs.

## Testing Best Practices

### Test Organization
- Mirror the source directory structure in `src/tests/`
- Use descriptive test names that indicate what is being tested
- Group related tests in the same file

### Test Design
- Keep tests focused and independent
- Use meaningful assertion messages
- Test both success and failure cases
- Include edge cases and boundary conditions

### Performance Testing
- Use `Benchmark` tests to test algorithms performance
- Include multiple data sizes to test scalability
- Document expected performance characteristics

### MPI Testing
- Test with different numbers of MPI processes
- Ensure tests work with both single and multi-process configurations
- Use `-1` for node count when the test is process-count independent

## IDE Integration

### Visual Studio Code
- Install clangd vscode extension for better debugging (see TODO: insert link to the vscode setup guide)

### Debugging Tests
```bash
# Run specific test with debugger
gdb --args ./shamrock_test --sycl-cfg 0:0 --run-only "test_name"
```
