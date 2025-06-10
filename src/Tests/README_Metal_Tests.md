# Metal Backend Tests

This directory contains comprehensive tests for the ARBD Metal backend implementation. The tests are designed to validate all aspects of the Metal backend functionality, from basic device operations to complex compute shader execution.

## Test Files

- `test_metal.mm` - Main Metal test suite (Objective-C++)
- `test_metal_kernels.metal` - Metal compute shaders for testing
- `README_Metal_Tests.md` - This documentation file

## Test Categories

### 1. Metal Availability Tests (`[metal][availability]`)
- Verifies Metal is available on the system
- Checks basic device properties
- Ensures Metal devices are discoverable

### 2. DeviceMemory Tests (`[metal][memory]`)
- **Basic Operations**: Construction, destruction, size validation
- **Move Semantics**: Move constructor and assignment operators
- **Data Transfer**: Host-to-device and device-to-host memory transfers
- **Error Handling**: Invalid sizes, null devices, buffer validation

### 3. Queue Operations Tests (`[metal][queue]`)
- Queue creation and management
- Command buffer operations
- Synchronization mechanisms
- Move semantics for queue objects
- Error handling for invalid operations

### 4. Event Operations Tests (`[metal][event]`)
- Event creation and lifecycle management
- Command buffer timing and profiling
- Event completion status tracking
- Move semantics for event objects

### 5. METALManager Tests (`[metal][manager]`)
- **Initialization**: Device discovery and initialization
- **Device Operations**: Device selection, switching, and management
- **Synchronization**: Global and per-device synchronization
- **Properties**: Device categorization and property queries
- **Power Management**: Low-power device preferences
- **Error Handling**: Invalid device IDs and edge cases

### 6. Compute Shader Integration Tests (`[metal][compute][integration]`)
- **Vector Addition**: Real Metal kernel execution with vector operations
- **Fill Operations**: Simple buffer fill kernels
- **Pipeline State Management**: Compute pipeline creation and execution
- **Memory Binding**: Buffer binding and kernel parameter passing

### 7. Performance Tests (`[metal][performance]`)
- Memory transfer timing and benchmarks
- Queue operation performance metrics
- Synchronization overhead measurements

### 8. Integration Tests (`[metal][integration]`)
- End-to-end workflows combining multiple components
- Complex memory operations with compute
- Real-world usage scenarios

### 9. Stress Tests (`[metal][stress]`)
- Multiple simultaneous memory allocations
- Rapid queue operations
- Resource exhaustion testing

## Building and Running Tests

### Prerequisites

1. macOS system with Metal support
2. Xcode command line tools installed
3. CMake with Metal backend enabled (`USE_METAL=ON`)

### Building

```bash
# From the project root directory
mkdir build && cd build
cmake .. -DUSE_METAL=ON
make arbd_tests
```

### Running Tests

```bash
# Run all Metal tests
./arbd_tests "[metal]"

# Run specific test categories
./arbd_tests "[metal][memory]"           # Memory tests only
./arbd_tests "[metal][manager]"          # Manager tests only
./arbd_tests "[metal][compute]"          # Compute shader tests only
./arbd_tests "[metal][performance]"      # Performance tests only

# Run with verbose output
./arbd_tests "[metal]" -s

# List all available Metal tests
./arbd_tests "[metal]" --list-tests
```

### Test Tags

Tests are organized with hierarchical tags for easy filtering:

- `[metal]` - All Metal tests
- `[metal][availability]` - Metal availability checks
- `[metal][memory]` - DeviceMemory class tests
- `[metal][memory][move]` - Move semantics tests
- `[metal][memory][transfer]` - Data transfer tests
- `[metal][queue]` - Queue class tests
- `[metal][event]` - Event class tests
- `[metal][manager]` - METALManager class tests
- `[metal][manager][init]` - Initialization tests
- `[metal][manager][device]` - Device management tests
- `[metal][manager][sync]` - Synchronization tests
- `[metal][manager][properties]` - Device property tests
- `[metal][manager][power]` - Power management tests
- `[metal][manager][error]` - Error handling tests
- `[metal][compute]` - Compute shader tests
- `[metal][integration]` - Integration tests
- `[metal][performance]` - Performance benchmarks
- `[metal][stress]` - Stress tests
- `[metal][manager][finalize]` - Cleanup tests

## Test Features

### Automatic Error Handling
- All tests use ARBD's exception handling system
- Tests validate both success and failure cases
- Memory leaks are prevented through RAII

### Platform Independence
- Tests automatically skip on non-Metal systems
- Graceful degradation when Metal features are unavailable
- Conditional compilation with `#ifdef USE_METAL`

### Comprehensive Coverage
- Unit tests for individual classes and methods
- Integration tests for component interactions
- Performance tests for benchmarking
- Stress tests for stability validation

### Real Hardware Testing
- Tests run on actual Metal devices
- Compute shaders are executed on GPU hardware
- Memory operations use real device memory

## Expected Test Results

### On Metal-capable Systems
- All tests should pass
- Performance tests provide timing information
- Device discovery should find at least one Metal device
- Compute shaders should execute successfully

### On Non-Metal Systems
- Tests are automatically skipped
- No failures should occur due to missing Metal support
- Build system handles conditional compilation

## Debugging and Troubleshooting

### Common Issues

1. **Metal Not Available**: Ensure you're running on macOS with Metal support
2. **Compilation Errors**: Verify Xcode command line tools are installed
3. **Link Errors**: Ensure Metal and Foundation frameworks are linked
4. **Test Failures**: Check device availability and system resources

### Debug Options

```bash
# Enable verbose test output
./arbd_tests "[metal]" -s

# Run specific failing test
./arbd_tests "specific test name" -s

# Enable Metal validation layers (if available)
export MTL_DEBUG_LAYER=1
./arbd_tests "[metal]"
```

### Performance Considerations

- Performance tests provide relative measurements
- Results vary based on hardware and system load
- Some tests may be skipped on low-performance systems
- Large memory allocations may fail on memory-constrained systems

## Contributing

When adding new Metal tests:

1. Use appropriate test tags for organization
2. Include both positive and negative test cases
3. Test error conditions and edge cases
4. Use descriptive test names and sections
5. Add performance tests for new functionality
6. Ensure tests work across different Metal device types
7. Document any special requirements or limitations

## Integration with CI/CD

These tests are designed to integrate with continuous integration systems:

- Conditional execution based on Metal availability
- Proper exit codes for pass/fail status
- Detailed output for debugging failures
- Performance metrics for regression detection 