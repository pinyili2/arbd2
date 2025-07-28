#include "../catch_boiler.h"
#include "Random/Random.h"
#include "Backend/Profiler.h"
#include "Backend/Resource.h"
#include "Backend/Buffer.h"
#include "Backend/Kernels.h"
#include "Math/Types.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>

using namespace ARBD;
using namespace ARBD::Profiling;
using Catch::Approx;
using Catch::Matchers::WithinAbs;

// ============================================================================
// Kernel Functors for Profiling Tests
// ============================================================================

struct InitializeWalkersKernel {
    template<typename... Args>
    void operator()(size_t i, Args... args) const {
        auto tuple_args = std::make_tuple(args...);
        auto* positions = std::get<0>(tuple_args);
        positions[i] = Vector3{0.0f, 0.0f, 0.0f};
    }
};

struct RandomWalkKernel {
    size_t NUM_STEPS;
    size_t NUM_WALKERS;

    template<typename... Args>
    void operator()(size_t walker_id, Args... args) const {
        auto tuple_args = std::make_tuple(args...);
        auto* steps = std::get<0>(tuple_args);
        auto* positions = std::get<1>(tuple_args);

        Vector3 pos = positions[walker_id];

        // Take NUM_STEPS/NUM_WALKERS steps per walker
        size_t steps_per_walker = NUM_STEPS / NUM_WALKERS;
        size_t start_step = walker_id * steps_per_walker;

        for (size_t step = 0; step < steps_per_walker && (start_step + step) < NUM_STEPS; ++step) {
            size_t step_idx = start_step + step;
            Vector3 step_vec = steps[step_idx];

            // Normalize step to unit length
            float length = std::sqrt(step_vec.x * step_vec.x +
                                   step_vec.y * step_vec.y +
                                   step_vec.z * step_vec.z);
            if (length > 0.0f) {
                step_vec.x /= length;
                step_vec.y /= length;
                step_vec.z /= length;
            }

            // Take the step
            pos.x += step_vec.x;
            pos.y += step_vec.y;
            pos.z += step_vec.z;
        }

        positions[walker_id] = pos;
    }
};

struct CalculateDistancesKernel {
    template<typename... Args>
    void operator()(size_t i, Args... args) const {
        auto tuple_args = std::make_tuple(args...);
        auto* positions = std::get<0>(tuple_args);
        auto* distances = std::get<1>(tuple_args);

        Vector3 pos = positions[i];
        distances[i] = std::sqrt(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z);
    }
};

struct SmoothingFilterKernel {
    size_t GRID_SIZE;

    template<typename... Args>
    void operator()(size_t i, Args... args) const {
        auto tuple_args = std::make_tuple(args...);
        auto* input = std::get<0>(tuple_args);
        auto* output = std::get<1>(tuple_args);

        size_t x = i % GRID_SIZE;
        size_t y = i / GRID_SIZE;

        // Simple 3x3 averaging filter
        float sum = 0.0f;
        int count = 0;

        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = static_cast<int>(x) + dx;
                int ny = static_cast<int>(y) + dy;

                if (nx >= 0 && nx < static_cast<int>(GRID_SIZE) &&
                    ny >= 0 && ny < static_cast<int>(GRID_SIZE)) {

                    size_t idx = ny * GRID_SIZE + nx;
                    sum += input[idx];
                    count++;
                }
            }
        }

        output[i] = (count > 0) ? sum / count : input[i];
    }
};

struct GradientCalculationKernel {
    size_t GRID_SIZE;

    template<typename... Args>
    void operator()(size_t i, Args... args) const {
        auto tuple_args = std::make_tuple(args...);
        auto* input = std::get<0>(tuple_args);
        auto* output = std::get<1>(tuple_args);

        size_t x = i % GRID_SIZE;
        size_t y = i / GRID_SIZE;

        float grad_x = 0.0f, grad_y = 0.0f;

        // Calculate finite difference gradients
        if (x > 0 && x < GRID_SIZE - 1) {
            size_t left_idx = y * GRID_SIZE + (x - 1);
            size_t right_idx = y * GRID_SIZE + (x + 1);
            grad_x = (input[right_idx] - input[left_idx]) * 0.5f;
        }

        if (y > 0 && y < GRID_SIZE - 1) {
            size_t top_idx = (y - 1) * GRID_SIZE + x;
            size_t bottom_idx = (y + 1) * GRID_SIZE + x;
            grad_y = (input[bottom_idx] - input[top_idx]) * 0.5f;
        }

        output[i] = std::sqrt(grad_x * grad_x + grad_y * grad_y);
    }
};

// ============================================================================
// Profiled Random Test Fixture
// ============================================================================

struct ProfiledRandomTestFixture {
    Resource cuda_resource, sycl_resource, metal_resource;
    bool cuda_available = false;
    bool sycl_available = false;
    bool metal_available = false;

    ProfiledRandomTestFixture() {
        // Initialize profiling
        ProfilingConfig config;
        config.enable_timing = true;
        config.enable_memory_tracking = true;
        config.enable_kernel_profiling = true;
        config.enable_backend_markers = true;
        config.output_file = "random_profile_test.json";
        ProfileManager::init(config);

        // Initialize only one backend at a time as per project rules
        try {
#ifdef USE_CUDA
            CUDA::CUDAManager::init();
            CUDA::CUDAManager::load_info();
            if (!CUDA::CUDAManager::devices().empty()) {
                CUDA::CUDAManager::use(0);
                cuda_resource = Resource(ResourceType::CUDA, 0);
                cuda_available = true;
                LOGINFO("CUDA backend available for profiled Random tests");
            }
#elif defined(USE_SYCL)
            SYCL::SYCLManager::init();
            SYCL::SYCLManager::load_info();
            if (!SYCL::SYCLManager::devices().empty()) {
                SYCL::SYCLManager::use(0);
                sycl_resource = Resource(ResourceType::SYCL, 0);
                sycl_available = true;
                LOGINFO("SYCL backend available for profiled Random tests");
            }
#elif defined(USE_METAL)
            METAL::METALManager::init();
            METAL::METALManager::load_info();
            if (!METAL::METALManager::devices().empty()) {
                METAL::METALManager::use(0);
                metal_resource = Resource(ResourceType::METAL, 0);
                metal_available = true;
                LOGINFO("Metal backend available for profiled Random tests");
            }
#endif
        } catch (const std::exception& e) {
            LOGWARN("Backend initialization failed in ProfiledRandomTestFixture: {}", e.what());
        }
    }

    ~ProfiledRandomTestFixture() {
        ProfileManager::print_summary();
        ProfileManager::finalize();

        try {
#ifdef USE_CUDA
            if (cuda_available) CUDA::CUDAManager::finalize();
#elif defined(USE_SYCL)
            if (sycl_available) SYCL::SYCLManager::finalize();
#elif defined(USE_METAL)
            if (metal_available) METAL::METALManager::finalize();
#endif
        } catch (const std::exception& e) {
            std::cerr << "Error during ProfiledRandomTestFixture cleanup: " << e.what() << std::endl;
        }
    }
};

// ============================================================================
// Profiled Random Generation Tests
// ============================================================================

TEST_CASE_METHOD(ProfiledRandomTestFixture, "Profiled Random Generation Performance", "[random][profiling][performance]") {

    auto test_profiled_generation = [this](const Resource& resource, const std::string& backend_name, ResourceType backend_type) {
        if (!resource.is_device()) {
            SKIP("Backend " + backend_name + " not available");
        }

        SECTION("Profiled uniform generation on " + backend_name) {
            PROFILE_RANGE("RandomGenerator::Creation", backend_type);

            Random<Resource> rng(resource, 128);
            rng.init(42, 0);

            // Profile memory allocation
            {
                PROFILE_RANGE("DeviceBuffer::Allocation", backend_type);
                DeviceBuffer<float> device_buffer(100000); // 100K elements

                PROFILE_MEMORY(backend_type, nullptr);

                // Profile random number generation
                {
                    PROFILE_RANGE("Random::GenerateUniform", backend_type);
                    Event generation_event = rng.generate_uniform(device_buffer, 0.0f, 1.0f);
                    generation_event.wait();
                }

                // Profile memory copy
                {
                    PROFILE_RANGE("Buffer::CopyToHost", backend_type);
                    std::vector<float> host_results(100000);
                    device_buffer.copy_to_host(host_results);
                }

                PROFILE_MARK("Uniform generation completed", backend_type);
            }

            LOGINFO("{} profiled uniform generation completed", backend_name);
        }

        SECTION("Profiled gaussian generation on " + backend_name) {
            PROFILE_RANGE("RandomGenerator::GaussianTest", backend_type);

            Random<Resource> rng(resource, 128);
            rng.init(12345, 0);

            DeviceBuffer<float> device_buffer(50000);

            {
                PROFILE_RANGE("Random::GenerateGaussian", backend_type);
                Event generation_event = rng.generate_gaussian(device_buffer, 0.0f, 1.0f);
                generation_event.wait();
            }

            {
                PROFILE_RANGE("Statistical::Validation", backend_type);
                std::vector<float> host_results(50000);
                device_buffer.copy_to_host(host_results);

                // Quick statistical validation
                double mean = std::accumulate(host_results.begin(), host_results.end(), 0.0) / host_results.size();
                double sq_sum = std::inner_product(host_results.begin(), host_results.end(), host_results.begin(), 0.0);
                double stdev = std::sqrt(sq_sum / host_results.size() - mean * mean);

                REQUIRE_THAT(mean, WithinAbs(0.0, 0.1));
                REQUIRE_THAT(stdev, WithinAbs(1.0, 0.2));
            }

            PROFILE_MARK("Gaussian generation validated", backend_type);
            LOGINFO("{} profiled gaussian generation completed", backend_name);
        }

        SECTION("Profiled Vector3 generation on " + backend_name) {
            PROFILE_RANGE("RandomGenerator::Vector3Test", backend_type);

            Random<Resource> rng(resource, 128);
            rng.init(9876, 0);

            DeviceBuffer<Vector3> device_buffer(25000);
            Vector3 mean(0.0f, 0.0f, 0.0f);
            Vector3 dev(1.0f, 1.0f, 1.0f);

            {
                PROFILE_RANGE("Random::GenerateVector3", backend_type);
                Event generation_event = rng.generate_gaussian(device_buffer, mean, dev);
                generation_event.wait();
            }

            {
                PROFILE_RANGE("Vector3::Validation", backend_type);
                std::vector<Vector3> host_results(25000);
                device_buffer.copy_to_host(host_results);

                // Validate that we have proper Vector3 data
                bool all_finite = std::all_of(host_results.begin(), host_results.end(),
                    [](const Vector3& v) {
                        return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
                    });

                REQUIRE(all_finite);
            }

            PROFILE_MARK("Vector3 generation validated", backend_type);
            LOGINFO("{} profiled Vector3 generation completed", backend_name);
        }
    };

    // Only test available backends - single backend usage as per project rules
#ifdef USE_CUDA
    test_profiled_generation(cuda_resource, "CUDA", ResourceType::CUDA);
#endif
#ifdef USE_SYCL
    test_profiled_generation(sycl_resource, "SYCL", ResourceType::SYCL);
#endif
#ifdef USE_METAL
    test_profiled_generation(metal_resource, "Metal", ResourceType::METAL);
#endif
}

// ============================================================================
// Profiled Random + Kernels Integration Tests
// ============================================================================

TEST_CASE_METHOD(ProfiledRandomTestFixture, "Profiled Random-Kernel Integration", "[random][kernels][profiling][integration]") {

    auto test_profiled_integration = [this](const Resource& resource, const std::string& backend_name, ResourceType backend_type) {
        if (!resource.is_device()) {
            SKIP("Backend " + backend_name + " not available");
        }

        SECTION("Profiled Monte Carlo simulation on " + backend_name) {
            PROFILE_RANGE("MonteCarlo::Simulation", backend_type);

            constexpr size_t NUM_SAMPLES = 100000; // 100K samples for good statistics

            Random<Resource> rng(resource, 128);
            rng.init(314159, 0);

            // First, test if random generation is working as expected
            DeviceBuffer<float> test_buffer(100);
            Event test_event = rng.generate_uniform(test_buffer, -1.0f, 1.0f);
            test_event.wait();

            std::vector<float> test_values(100);
            test_buffer.copy_to_host(test_values);

            auto [test_min, test_max] = std::minmax_element(test_values.begin(), test_values.end());
            double test_mean = std::accumulate(test_values.begin(), test_values.end(), 0.0) / test_values.size();

            LOGINFO("Test random generation: min={:.3f}, max={:.3f}, mean={:.3f}", *test_min, *test_max, test_mean);

            // If this is showing values outside [-1, 1] or mean not near 0, we have found the issue

            DeviceBuffer<float> x_coords(NUM_SAMPLES);
            DeviceBuffer<float> y_coords(NUM_SAMPLES);
            DeviceBuffer<int> inside_circle(NUM_SAMPLES);

            // Generate random coordinates
            {
                PROFILE_RANGE("Random::GenerateCoordinates", backend_type);
                Event x_event = rng.generate_uniform(x_coords, -1.0f, 1.0f);
                x_event.wait(); // Wait for X generation to complete

                // Reinitialize with different seed/offset to ensure different sequence for Y
                rng.init(314159, NUM_SAMPLES); // Use different offset
                Event y_event = rng.generate_uniform(y_coords, -1.0f, 1.0f);
                y_event.wait();

                // Quick validation that random generation worked
                std::vector<float> quick_x_check(10), quick_y_check(10);
                x_coords.copy_to_host(quick_x_check.data(), 10);
                y_coords.copy_to_host(quick_y_check.data(), 10);

                bool x_in_range = std::all_of(quick_x_check.begin(), quick_x_check.end(),
                    [](float x) { return x >= -1.0f && x <= 1.0f; });
                bool y_in_range = std::all_of(quick_y_check.begin(), quick_y_check.end(),
                    [](float y) { return y >= -1.0f && y <= 1.0f; });

                if (!x_in_range || !y_in_range) {
                    LOGWARN("Random numbers out of expected range! X in range: {}, Y in range: {}", x_in_range, y_in_range);
                }
            }

            // Check which points are inside unit circle using custom kernel
            // (Random.h already handles its own kernels, but we need a custom one for this logic)
            {
                PROFILE_RANGE("Kernel::CircleTest", backend_type);

                // Use a simple CPU-side computation for this test
                // or implement as a separate kernel in your Kernels.h if needed
                std::vector<float> x_host(NUM_SAMPLES), y_host(NUM_SAMPLES);
                std::vector<int> inside_host(NUM_SAMPLES);

                x_coords.copy_to_host(x_host);
                y_coords.copy_to_host(y_host);

                // Debug: Check first few random numbers
                LOGINFO("First 10 x coordinates: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}",
                        x_host[0], x_host[1], x_host[2], x_host[3], x_host[4], x_host[5], x_host[6], x_host[7], x_host[8], x_host[9]);
                LOGINFO("First 10 y coordinates: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}",
                        y_host[0], y_host[1], y_host[2], y_host[3], y_host[4], y_host[5], y_host[6], y_host[7], y_host[8], y_host[9]);

                // Check range of all values
                auto [x_min, x_max] = std::minmax_element(x_host.begin(), x_host.end());
                auto [y_min, y_max] = std::minmax_element(y_host.begin(), y_host.end());
                LOGINFO("X range: [{:.3f}, {:.3f}], Y range: [{:.3f}, {:.3f}]", *x_min, *x_max, *y_min, *y_max);

                // Statistical analysis of the random numbers
                double x_mean = std::accumulate(x_host.begin(), x_host.end(), 0.0) / x_host.size();
                double y_mean = std::accumulate(y_host.begin(), y_host.end(), 0.0) / y_host.size();
                LOGINFO("X mean: {:.6f} (should be ~0.0), Y mean: {:.6f} (should be ~0.0)", x_mean, y_mean);

                int debug_count = 0;
                int total_positive_x = 0, total_positive_y = 0;
                double sum_dist_sq = 0.0;

                for (size_t i = 0; i < NUM_SAMPLES; ++i) {
                    float dist_sq = x_host[i] * x_host[i] + y_host[i] * y_host[i];
                    inside_host[i] = (dist_sq <= 1.0f) ? 1 : 0;

                    // Additional statistics
                    if (x_host[i] > 0) total_positive_x++;
                    if (y_host[i] > 0) total_positive_y++;
                    sum_dist_sq += dist_sq;

                    // Debug first few calculations
                    if (i < 10) {
                        LOGINFO("Point {}: ({:.3f}, {:.3f}) -> dist_sq={:.3f}, inside={}",
                                i, x_host[i], y_host[i], dist_sq, inside_host[i]);
                    }
                    if (inside_host[i] == 1) debug_count++;
                }

                double mean_dist_sq = sum_dist_sq / NUM_SAMPLES;
                LOGINFO("Debug: Found {} points inside circle out of {} samples", debug_count, NUM_SAMPLES);
                LOGINFO("Positive X: {} (should be ~{}), Positive Y: {} (should be ~{})",
                        total_positive_x, NUM_SAMPLES/2, total_positive_y, NUM_SAMPLES/2);
                LOGINFO("Mean distance squared: {:.6f} (should be ~0.667 for uniform in [-1,1]²)", mean_dist_sq);

                inside_circle.copy_from_host(inside_host);
            }

            // Calculate π estimate
            {
                PROFILE_RANGE("MonteCarlo::PiEstimation", backend_type);
                std::vector<int> inside_results(NUM_SAMPLES);
                inside_circle.copy_to_host(inside_results);

                int points_inside = std::accumulate(inside_results.begin(), inside_results.end(), 0);
                double pi_estimate = 4.0 * static_cast<double>(points_inside) / NUM_SAMPLES;

                // Debug: Print statistics
                LOGINFO("Points inside circle: {} out of {} (ratio: {:.6f})",
                        points_inside, NUM_SAMPLES, static_cast<double>(points_inside) / NUM_SAMPLES);

                // π should be approximately 3.14159
                // For Monte Carlo with 10K samples, expect reasonable accuracy
                // Fixed the bug where X and Y coordinates were identical
                REQUIRE_THAT(pi_estimate, WithinAbs(3.14159, 0.1));

                LOGINFO("{} Monte Carlo π estimate: {:.5f} (error: {:.5f})",
                        backend_name, pi_estimate, std::abs(pi_estimate - 3.14159));
            }

            PROFILE_MARK("Monte Carlo simulation completed", backend_type);
        }

        SECTION("Profiled random walk simulation on " + backend_name) {
            PROFILE_RANGE("RandomWalk::Simulation", backend_type);

            constexpr size_t NUM_STEPS = 100000;
            constexpr size_t NUM_WALKERS = 1000;

            Random<Resource> rng(resource, 128);
            rng.init(271828, 0);

            DeviceBuffer<Vector3> random_steps(NUM_STEPS);
            DeviceBuffer<Vector3> walker_positions(NUM_WALKERS);
            DeviceBuffer<float> final_distances(NUM_WALKERS);
            Vector3 mean(0.0f, 0.0f, 0.0f);
            Vector3 dev(1.0f, 1.0f, 1.0f);
            // Generate random step directions
            {
                PROFILE_RANGE("Random::GenerateSteps", backend_type);
                Event steps_event = rng.generate_gaussian(random_steps, mean, dev);
                steps_event.wait();
            }

            // Initialize walker positions to origin
            {
                PROFILE_RANGE("Kernel::InitializeWalkers", backend_type);
                KernelConfig config{.block_size = 256, .async = false};

                auto inputs = std::make_tuple();
                auto outputs = std::make_tuple(std::ref(walker_positions));

                Event init_event = launch_kernel(
                    resource,
                    NUM_WALKERS,
                    config,
                    inputs,
                    outputs,
                    InitializeWalkersKernel{}
                );

                init_event.wait();
            }

            // Simulate random walk
            {
                PROFILE_RANGE("Kernel::RandomWalk", backend_type);
                KernelConfig config{.block_size = 256, .async = false};

                auto inputs = std::make_tuple(std::ref(random_steps));
                auto outputs = std::make_tuple(std::ref(walker_positions));

                Event walk_event = launch_kernel(
                    resource,
                    NUM_WALKERS,
                    config,
                    inputs,
                    outputs,
                    RandomWalkKernel{NUM_STEPS, NUM_WALKERS}
                );

                walk_event.wait();
            }

            // Calculate final distances from origin
            {
                PROFILE_RANGE("Kernel::CalculateDistances", backend_type);
                KernelConfig config{.block_size = 256, .async = false};

                auto inputs = std::make_tuple(std::ref(walker_positions));
                auto outputs = std::make_tuple(std::ref(final_distances));

                Event distance_event = launch_kernel(
                    resource,
                    NUM_WALKERS,
                    config,
                    inputs,
                    outputs,
                    CalculateDistancesKernel{}
                );

                distance_event.wait();
            }

            // Analyze results
            {
                PROFILE_RANGE("RandomWalk::Analysis", backend_type);
                std::vector<float> distances(NUM_WALKERS);
                final_distances.copy_to_host(distances);

                double mean_distance = std::accumulate(distances.begin(), distances.end(), 0.0) / NUM_WALKERS;
                double max_distance = *std::max_element(distances.begin(), distances.end());

                // For a 3D random walk, mean distance should scale roughly as sqrt(N)
                double expected_distance = std::sqrt(NUM_STEPS / NUM_WALKERS);

                REQUIRE(mean_distance > 0.0);
                REQUIRE(max_distance > mean_distance);

                LOGINFO("{} Random Walk: {} walkers, mean distance: {:.3f}, max distance: {:.3f}, expected: {:.3f}",
                        backend_name, NUM_WALKERS, mean_distance, max_distance, expected_distance);
            }

            PROFILE_MARK("Random walk simulation completed", backend_type);
        }

        SECTION("Profiled noise generation pipeline on " + backend_name) {
            PROFILE_RANGE("NoiseGeneration::Pipeline", backend_type);

            constexpr size_t GRID_SIZE = 256;
            constexpr size_t TOTAL_POINTS = GRID_SIZE * GRID_SIZE;

            Random<Resource> rng(resource, 128);
            rng.init(161803, 0);

            DeviceBuffer<float> noise_values(TOTAL_POINTS);
            DeviceBuffer<float> smoothed_noise(TOTAL_POINTS);
            DeviceBuffer<float> gradient_magnitude(TOTAL_POINTS);

            // Generate base noise
            {
                PROFILE_RANGE("Random::GenerateNoise", backend_type);
                Event noise_event = rng.generate_gaussian(noise_values, 0.0f, 1.0f);
                noise_event.wait();
            }

            // Apply smoothing filter
            {
                PROFILE_RANGE("Kernel::SmoothingFilter", backend_type);
                KernelConfig config{.block_size = 256, .async = false};

                auto inputs = std::make_tuple(std::ref(noise_values));
                auto outputs = std::make_tuple(std::ref(smoothed_noise));

                Event smooth_event = launch_kernel(
                    resource,
                    TOTAL_POINTS,
                    config,
                    inputs,
                    outputs,
                    SmoothingFilterKernel{GRID_SIZE}
                );

                smooth_event.wait();
            }

            // Calculate gradient magnitude
            {
                PROFILE_RANGE("Kernel::GradientCalculation", backend_type);
                KernelConfig config{.block_size = 256, .async = false};

                auto inputs = std::make_tuple(std::ref(smoothed_noise));
                auto outputs = std::make_tuple(std::ref(gradient_magnitude));

                Event gradient_event = launch_kernel(
                    resource,
                    TOTAL_POINTS,
                    config,
                    inputs,
                    outputs,
                    GradientCalculationKernel{GRID_SIZE}
                );

                gradient_event.wait();
            }

            // Validate results
            {
                PROFILE_RANGE("NoiseGeneration::Validation", backend_type);
                std::vector<float> original_noise(TOTAL_POINTS);
                std::vector<float> smoothed(TOTAL_POINTS);
                std::vector<float> gradients(TOTAL_POINTS);

                noise_values.copy_to_host(original_noise);
                smoothed_noise.copy_to_host(smoothed);
                gradient_magnitude.copy_to_host(gradients);

                // Basic validation
                double original_variance = 0.0, smoothed_variance = 0.0;
                for (size_t i = 0; i < TOTAL_POINTS; ++i) {
                    original_variance += original_noise[i] * original_noise[i];
                    smoothed_variance += smoothed[i] * smoothed[i];
                }
                original_variance /= TOTAL_POINTS;
                smoothed_variance /= TOTAL_POINTS;

                // Smoothing should reduce variance
                REQUIRE(smoothed_variance <= original_variance);

                // All gradients should be non-negative
                bool all_gradients_valid = std::all_of(gradients.begin(), gradients.end(),
                    [](float g) { return g >= 0.0f && std::isfinite(g); });
                REQUIRE(all_gradients_valid);

                double max_gradient = *std::max_element(gradients.begin(), gradients.end());

                LOGINFO("{} Noise Pipeline: {}x{} grid, original variance: {:.3f}, smoothed variance: {:.3f}, max gradient: {:.3f}",
                        backend_name, GRID_SIZE, GRID_SIZE, original_variance, smoothed_variance, max_gradient);
            }

            PROFILE_MARK("Noise generation pipeline completed", backend_type);
        }
    };

    // Only test available backends - single backend usage as per project rules
#ifdef USE_CUDA
    test_profiled_integration(cuda_resource, "CUDA", ResourceType::CUDA);
#endif
#ifdef USE_SYCL
    test_profiled_integration(sycl_resource, "SYCL", ResourceType::SYCL);
#endif
#ifdef USE_METAL
    test_profiled_integration(metal_resource, "Metal", ResourceType::METAL);
#endif
}

// ============================================================================
// Comparative Performance Tests
// ============================================================================

TEST_CASE_METHOD(ProfiledRandomTestFixture, "Random Generation Comparative Performance", "[random][profiling][comparison]") {

    std::vector<std::pair<Resource, std::string>> available_backends;

    // Only use one backend at a time as per project rules
#ifdef USE_CUDA
    if (cuda_available) available_backends.emplace_back(cuda_resource, "CUDA");
#endif
#ifdef USE_SYCL
    if (sycl_available) available_backends.emplace_back(sycl_resource, "SYCL");
#endif
#ifdef USE_METAL
    if (metal_available) available_backends.emplace_back(metal_resource, "Metal");
#endif

    if (available_backends.empty()) {
        SKIP("No backends available for comparative testing");
    }

    SECTION("Multi-device uniform generation with cross-device analysis") {
        constexpr size_t PERF_SIZE = 10000000; // 10M elements per device
        constexpr size_t SAMPLE_SIZE = 10000;   // Sample for cross-device comparison

        // Get all available SYCL devices (your 8 A100s)
        const auto& sycl_devices = SYCL::SYCLManager::devices();
        const size_t num_devices = sycl_devices.size();

        REQUIRE(num_devices > 0);
        LOGINFO("Testing parallel generation across {} SYCL devices", num_devices);

        // Per-device data structures
        std::vector<Resource> device_resources;
        std::vector<std::unique_ptr<Random<Resource>>> device_rngs;
        std::vector<DeviceBuffer<float>> device_buffers;
        std::vector<Event> generation_events;
        std::vector<std::chrono::high_resolution_clock::time_point> start_times;
        std::vector<double> device_timings;

        // Initialize all devices
        for (size_t i = 0; i < num_devices; ++i) {
            device_resources.emplace_back(ResourceType::SYCL, i);
            device_rngs.emplace_back(std::make_unique<Random<Resource>>(device_resources[i], 256));
            device_buffers.emplace_back(PERF_SIZE);

            // Use different seeds for each device to ensure independence
            device_rngs[i]->init(123456 + i * 1000, i);
        }

        PROFILE_RANGE("MultiDevice::ParallelGeneration", ResourceType::SYCL);

        // Launch generation on all devices simultaneously
        auto global_start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < num_devices; ++i) {
            start_times.push_back(std::chrono::high_resolution_clock::now());

            Event event = device_rngs[i]->generate_uniform(device_buffers[i], 0.0f, 1.0f);
            generation_events.push_back(std::move(event));
        }

        // Wait for all devices to complete and measure individual timings
        device_timings.resize(num_devices);
        for (size_t i = 0; i < num_devices; ++i) {
            generation_events[i].wait();
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_times[i]);
            device_timings[i] = static_cast<double>(duration.count()) / 1000.0; // Convert to ms
        }

        auto global_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(global_end - global_start);
        double total_time_ms = static_cast<double>(total_duration.count()) / 1000.0;

        // Performance analysis
        double min_time = *std::min_element(device_timings.begin(), device_timings.end());
        double max_time = *std::max_element(device_timings.begin(), device_timings.end());
        double avg_time = std::accumulate(device_timings.begin(), device_timings.end(), 0.0) / num_devices;

        size_t total_numbers = PERF_SIZE * num_devices;
        double total_throughput = (total_numbers / 1000000.0) / (total_time_ms / 1000.0); // M numbers/sec
        double theoretical_throughput = (total_numbers / 1000000.0) / (min_time / 1000.0); // If all ran at fastest speed

        LOGINFO("=== Multi-Device Performance Results ===");
        LOGINFO("Total numbers generated: {} ({} per device)", total_numbers, PERF_SIZE);
        LOGINFO("Total wall-clock time: {:.3f} ms", total_time_ms);
        LOGINFO("Individual device times: min={:.3f}ms, max={:.3f}ms, avg={:.3f}ms", min_time, max_time, avg_time);
        LOGINFO("Total throughput: {:.1f} M numbers/sec", total_throughput);
        LOGINFO("Theoretical max throughput: {:.1f} M numbers/sec (if all devices matched fastest)", theoretical_throughput);
        LOGINFO("Parallel efficiency: {:.1f}% (actual vs theoretical)", (total_throughput / theoretical_throughput) * 100.0);

        for (size_t i = 0; i < num_devices; ++i) {
            double device_throughput = (PERF_SIZE / 1000000.0) / (device_timings[i] / 1000.0);
            LOGINFO("Device {}: {:.3f} ms ({:.1f} M numbers/sec)", i, device_timings[i], device_throughput);
        }

        // Cross-device statistical analysis
        //PROFILE_RANGE("CrossDevice::StatisticalAnalysis", ResourceType::SYCL);

        std::vector<std::vector<float>> device_samples(num_devices);
        std::vector<float> device_means(num_devices);
        std::vector<float> device_mins(num_devices);
        std::vector<float> device_maxes(num_devices);
        std::vector<float> device_variances(num_devices);

        // Sample data from each device for statistical comparison
        for (size_t i = 0; i < num_devices; ++i) {
            device_samples[i].resize(SAMPLE_SIZE);
            device_buffers[i].copy_to_host(device_samples[i].data(), SAMPLE_SIZE);

            // Calculate statistics
            auto [min_it, max_it] = std::minmax_element(device_samples[i].begin(), device_samples[i].end());
            device_mins[i] = *min_it;
            device_maxes[i] = *max_it;

            double sum = std::accumulate(device_samples[i].begin(), device_samples[i].end(), 0.0);
            device_means[i] = sum / SAMPLE_SIZE;

            double variance_sum = 0.0;
            for (float val : device_samples[i]) {
                variance_sum += (val - device_means[i]) * (val - device_means[i]);
            }
            device_variances[i] = variance_sum / SAMPLE_SIZE;
        }

        // Cross-device analysis
        float global_min = *std::min_element(device_mins.begin(), device_mins.end());
        float global_max = *std::max_element(device_maxes.begin(), device_maxes.end());
        float mean_of_means = std::accumulate(device_means.begin(), device_means.end(), 0.0f) / num_devices;
        float mean_of_variances = std::accumulate(device_variances.begin(), device_variances.end(), 0.0f) / num_devices;

        LOGINFO("=== Cross-Device Statistical Analysis ===");
        LOGINFO("Global range: [{:.6f}, {:.6f}]", global_min, global_max);
        LOGINFO("Mean of device means: {:.6f} (should be ~0.5)", mean_of_means);
        LOGINFO("Mean of device variances: {:.6f} (should be ~0.083 for uniform[0,1])", mean_of_variances);

        // Device-by-device breakdown
        for (size_t i = 0; i < num_devices; ++i) {
            LOGINFO("Device {}: mean={:.6f}, var={:.6f}, range=[{:.6f}, {:.6f}]",
                   i, device_means[i], device_variances[i], device_mins[i], device_maxes[i]);
        }

        // Statistical tests for uniformity across devices
        // Test 1: Check if all device means are close to 0.5
        float max_mean_deviation = 0.0f;
        for (float mean : device_means) {
            max_mean_deviation = std::max(max_mean_deviation, std::abs(mean - 0.5f));
        }

        // Test 2: Check variance consistency across devices
        float min_variance = *std::min_element(device_variances.begin(), device_variances.end());
        float max_variance = *std::max_element(device_variances.begin(), device_variances.end());
        float variance_ratio = max_variance / min_variance;

        LOGINFO("=== Cross-Device Quality Metrics ===");
        LOGINFO("Max mean deviation from 0.5: {:.6f} (should be < 0.01)", max_mean_deviation);
        LOGINFO("Variance ratio (max/min): {:.3f} (should be < 1.5)", variance_ratio);

        // Quality assertions
        REQUIRE(global_min >= 0.0f);
        REQUIRE(global_max <= 1.0f);
        REQUIRE(max_mean_deviation < 0.01f); // All devices should have mean ~0.5
        REQUIRE(variance_ratio < 1.5f);      // Variances shouldn't differ too much
        REQUIRE(mean_of_means > 0.45f);      // Overall mean should be reasonable
        REQUIRE(mean_of_means < 0.55f);

        // Cross-device correlation test (optional advanced test)
        if (num_devices >= 2) {
            PROFILE_RANGE("CrossDevice::CorrelationTest", ResourceType::SYCL);

            // Test that different devices with different seeds produce uncorrelated sequences
            std::vector<float> correlation_coeffs;

            for (size_t i = 0; i < num_devices - 1; ++i) {
                for (size_t j = i + 1; j < num_devices; ++j) {
                    double correlation = calculate_correlation(device_samples[i], device_samples[j]);
                    correlation_coeffs.push_back(std::abs(correlation));

                    LOGINFO("Correlation between device {} and {}: {:.6f}", i, j, correlation);
                }
            }

            float max_correlation = *std::max_element(correlation_coeffs.begin(), correlation_coeffs.end());
            LOGINFO("Maximum cross-device correlation: {:.6f} (should be < 0.05)", max_correlation);

            REQUIRE(max_correlation < 0.05f); // Devices should produce uncorrelated sequences
        }

        LOGINFO("=== Multi-Device Test Summary ===");
        LOGINFO("✓ {} devices generated {:.1f}M numbers in {:.3f}ms",
               num_devices, total_numbers / 1000000.0, total_time_ms);
        LOGINFO("✓ Aggregate throughput: {:.1f} M numbers/sec", total_throughput);
        LOGINFO("✓ All quality metrics passed");
    }

    private:
        // Helper function to calculate Pearson correlation coefficient
        double calculate_correlation(const std::vector<float>& x, const std::vector<float>& y) {
            if (x.size() != y.size() || x.empty()) return 0.0;

            double sum_x = std::accumulate(x.begin(), x.end(), 0.0);
            double sum_y = std::accumulate(y.begin(), y.end(), 0.0);
            double mean_x = sum_x / x.size();
            double mean_y = sum_y / y.size();

            double numerator = 0.0;
            double sum_sq_x = 0.0;
            double sum_sq_y = 0.0;

            for (size_t i = 0; i < x.size(); ++i) {
                double dx = x[i] - mean_x;
                double dy = y[i] - mean_y;
                numerator += dx * dy;
                sum_sq_x += dx * dx;
                sum_sq_y += dy * dy;
            }

            double denominator = std::sqrt(sum_sq_x * sum_sq_y);
            return (denominator > 1e-10) ? (numerator / denominator) : 0.0;
        }
    }
    SECTION("Gaussian generation performance comparison") {
        constexpr size_t PERF_SIZE = 5000000; // 5M elements (Gaussian is typically slower)

        std::vector<std::pair<std::string, double>> performance_results;

        for (const auto& [resource, backend_name] : available_backends) {
            PROFILE_RANGE("Comparative::Gaussian::" + backend_name, resource.type);

            Random<Resource> rng(resource, 256);
            rng.init(654321, 0);

            DeviceBuffer<float> device_buffer(PERF_SIZE);

            auto start = std::chrono::high_resolution_clock::now();

            Event generation_event = rng.generate_gaussian(device_buffer, 0.0f, 1.0f);
            generation_event.wait();

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            double time_ms = static_cast<double>(duration.count()) / 1000.0;

            performance_results.emplace_back(backend_name, time_ms);

            LOGINFO("{} generated {} Gaussian numbers in {:.3f} ms ({:.1f} M numbers/sec)",
                    backend_name, PERF_SIZE, time_ms, (PERF_SIZE / 1000000.0) / (time_ms / 1000.0));
        }

        // Find fastest backend
        auto fastest = std::min_element(performance_results.begin(), performance_results.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

        LOGINFO("Fastest backend for Gaussian generation: {} ({:.3f} ms)", fastest->first, fastest->second);
    }

    SECTION("Memory bandwidth test") {
        constexpr size_t BANDWIDTH_SIZE = 100000000; // 100M floats = ~400MB

        for (const auto& [resource, backend_name] : available_backends) {
            PROFILE_RANGE("Bandwidth::" + backend_name, resource.type);

            DeviceBuffer<float> device_buffer(BANDWIDTH_SIZE);
            std::vector<float> host_buffer(BANDWIDTH_SIZE, 1.0f);

            // Measure host-to-device bandwidth
            auto start = std::chrono::high_resolution_clock::now();
            device_buffer.copy_from_host(host_buffer);
            auto end = std::chrono::high_resolution_clock::now();

            auto h2d_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            double h2d_time_ms = static_cast<double>(h2d_duration.count()) / 1000.0;
            double h2d_bandwidth_gbps = (BANDWIDTH_SIZE * sizeof(float)) / (h2d_time_ms * 1000000.0);

            // Measure device-to-host bandwidth
            start = std::chrono::high_resolution_clock::now();
            device_buffer.copy_to_host(host_buffer);
            end = std::chrono::high_resolution_clock::now();

            auto d2h_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            double d2h_time_ms = static_cast<double>(d2h_duration.count()) / 1000.0;
            double d2h_bandwidth_gbps = (BANDWIDTH_SIZE * sizeof(float)) / (d2h_time_ms * 1000000.0);

            LOGINFO("{} Memory Bandwidth - H2D: {:.2f} GB/s, D2H: {:.2f} GB/s",
                    backend_name, h2d_bandwidth_gbps, d2h_bandwidth_gbps);
        }
    }
}
