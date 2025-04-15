#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_approx.hpp>
#include "../Proxy.h"
#include <memory>

// Simple test class
struct TestObject {
    int value;
    TestObject() : value(0) {}
    TestObject(int v) : value(v) {}
    void increment() { value++; }
    int add(int x) { value += x; return value; }
    void clear() {}
};

// Test class with metadata and dynamic memory
struct DynamicObject {
    struct Metadata {
        int id;
        std::string description;
        Metadata() : id(0), description("") {}
        explicit Metadata(int i) : id(i), description("test") {}
    };

    Metadata metadata;
    std::vector<int> data;
    
    DynamicObject() = default;
    explicit DynamicObject(size_t size) : data(size) {
        for(size_t i = 0; i < size; i++) data[i] = i;
    }

    DynamicObject send_children(const Resource& r) const {
        DynamicObject result;
        result.metadata = metadata;
        result.data = data;
        return result;
    }

    void clear() {
        data.clear();
    }
};

TEST_CASE("Basic Proxy Operations", "[proxy]") {
    Resource cpu_resource{Resource::CPU, 0};

    SECTION("Default construction") {
        Proxy<TestObject> proxy;
        REQUIRE(proxy.location.type == Resource::CPU);
        REQUIRE(proxy.addr == nullptr);
    }

    SECTION("Resource-specific construction") {
        Proxy<TestObject> proxy(cpu_resource);
        REQUIRE(proxy.location.type == Resource::CPU);
        REQUIRE(proxy.addr == nullptr);
    }
}

TEST_CASE("Memory Management", "[proxy]") {
    Resource cpu_resource{Resource::CPU, 0};

    SECTION("Send and access") {
        TestObject obj(42);
        auto proxy = send(cpu_resource, obj);
        
        REQUIRE(proxy.location.type == Resource::CPU);
        REQUIRE(proxy.addr != nullptr);
        REQUIRE(proxy->value == 42);
    }

    SECTION("Move construction") {
        TestObject obj(42);
        auto proxy1 = send(cpu_resource, obj);
        void* original_addr = proxy1.addr;
        
        Proxy<TestObject> proxy2(std::move(proxy1));
        REQUIRE(proxy2.addr == original_addr);
        REQUIRE(proxy1.addr == nullptr);
        REQUIRE(proxy2->value == 42);
    }
}

TEST_CASE("Dynamic Object Operations", "[proxy]") {
    Resource cpu_resource{Resource::CPU, 0};

    SECTION("Object with dynamic memory") {
        DynamicObject obj(10);
        obj.metadata.id = 42;
        
        auto proxy = send(cpu_resource, obj);
        REQUIRE(proxy.location.type == Resource::CPU);
        REQUIRE(proxy.addr != nullptr);
        REQUIRE(proxy->metadata.id == 42);
        REQUIRE(proxy->data.size() == 10);
        
        // Check data contents
        for(size_t i = 0; i < 10; i++) {
            REQUIRE(proxy->data[i] == i);
        }
    }
}

TEST_CASE("Type Safety", "[proxy]") {
    Resource cpu_resource{Resource::CPU, 0};

    SECTION("Integer proxy") {
        int value = 42;
        auto proxy = send(cpu_resource, value);
        REQUIRE(proxy.location.type == Resource::CPU);
        REQUIRE(*proxy.addr == 42);
    }

    SECTION("Float proxy") {
        float value = 3.14f;
        auto proxy = send(cpu_resource, value);
        REQUIRE(proxy.location.type == Resource::CPU);
        REQUIRE(*proxy.addr == Catch::Approx(3.14f));
    }
}

TEST_CASE("Proxy Method Calls", "[proxy]") {
    Resource cpu_resource{Resource::CPU, 0};
    
    SECTION("Direct method call") {
        TestObject obj(41);
        auto proxy = send(cpu_resource, obj);
        proxy->increment();
        REQUIRE(proxy->value == 42);
    }

    SECTION("Method with parameters") {
        TestObject obj(40);
        auto proxy = send(cpu_resource, obj);
        int result = proxy->add(2);
        REQUIRE(result == 42);
        REQUIRE(proxy->value == 42);
    }
}

TEST_CASE("Resource Location Tests", "[proxy]") {
    SECTION("CPU resource locality") {
        Resource cpu_resource{Resource::CPU, 0};
        REQUIRE(cpu_resource.is_local());
    }

    SECTION("Resource equality") {
        Resource res1{Resource::CPU, 0};
        Resource res2{Resource::CPU, 0};
        Resource res3{Resource::CPU, 1};
        
        REQUIRE(res1 == res2);
        REQUIRE(res1 != res3);
    }
}
