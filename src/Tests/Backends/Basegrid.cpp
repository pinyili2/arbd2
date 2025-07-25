// ============================================================================
// BaseGrid Buffer Tests
// ============================================================================

TEST_CASE_METHOD(BackendInitFixture, "BaseGrid Creation with Different Backends", "[buffer][basegrid][creation]") {
    
    SECTION("BaseGrid with local resource") {
        Resource resource = Resource::Local();
        
        Matrix3 basis = Matrix3(1.0f);  // Identity matrix
        Vector3 origin(0.0f, 0.0f, 0.0f);
        
        BaseGrid grid(basis, origin, GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z);
        
        REQUIRE(grid.nx() == GRID_SIZE_X);
        REQUIRE(grid.ny() == GRID_SIZE_Y);
        REQUIRE(grid.nz() == GRID_SIZE_Z);
        REQUIRE(grid.size() == GRID_SIZE_X * GRID_SIZE_Y * GRID_SIZE_Z);
        REQUIRE(grid.data() != nullptr);
        
        // Test resource assignment
        REQUIRE(grid.resource().type == resource.type);
        REQUIRE(grid.resource().id == resource.id);
    }
    
    SECTION("BaseGrid with box constructor") {
        Resource resource = Resource::Local();
        
        Vector3 box(10.0f, 10.0f, 10.0f);
        BaseGrid grid(box, 0.5f);
        
        REQUIRE(grid.size() > 0);
        REQUIRE(grid.nx() > 0);
        REQUIRE(grid.ny() > 0);
        REQUIRE(grid.nz() > 0);
        
        // Verify grid spacing is approximately correct
        Matrix3 computed_basis = grid.basis();
        float dx = computed_basis.ex().length();
        REQUIRE(dx <= 0.5f);  // Should be <= requested spacing
    }
    
    SECTION("BaseGrid copy constructor") {
        Resource resource = Resource::Local();
        
        Matrix3 basis = Matrix3(1.0f);
        Vector3 origin(0.0f);
        BaseGrid original_grid(basis, origin, GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z);
        
        // Test copy constructor
        BaseGrid copied_grid(original_grid);
        
        REQUIRE(copied_grid.nx() == original_grid.nx());
        REQUIRE(copied_grid.ny() == original_grid.ny());
        REQUIRE(copied_grid.nz() == original_grid.nz());
        REQUIRE(copied_grid.size() == original_grid.size());
        REQUIRE(copied_grid.resource().type == original_grid.resource().type);
    }
}

TEST_CASE_METHOD(BackendInitFixture, "BaseGrid Operations", "[buffer][basegrid][operations]") {
    
    SECTION("Basic grid operations") {
        Resource resource = Resource::Local();
        
        Matrix3 basis = Matrix3(1.0f);
        Vector3 origin(0.0f);
        BaseGrid grid(basis, origin, GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z);
        
        // Test zero operation
        auto zero_event = grid.zero();
        zero_event.wait();
        
        // Test scalar addition
        auto add_event = grid.add_scalar(5.0f);
        add_event.wait();
        
        // Test scalar multiplication
        auto mult_event = grid.multiply_scalar(2.0f);
        mult_event.wait();
        
        // All operations should complete without throwing
        REQUIRE(true);
    }
    
    SECTION("Grid copy operations") {
        Resource resource = Resource::Local();
        
        Matrix3 basis = Matrix3(1.0f);
        Vector3 origin(0.0f);
        BaseGrid grid1(basis, origin, GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z);
        BaseGrid grid2(basis, origin, GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z);
        
        // Initialize grid1
        auto zero_event1 = grid1.zero();
        zero_event1.wait();
        auto add_event1 = grid1.add_scalar(3.0f);
        add_event1.wait();
        
        // Copy from grid1 to grid2
        auto copy_event = grid2.copy_from(grid1);
        copy_event.wait();
        
        REQUIRE(true);  // Should complete without error
    }
    
    SECTION("Grid arithmetic operations") {
        Resource resource = Resource::Local();
        
        Matrix3 basis = Matrix3(1.0f);
        Vector3 origin(0.0f);
        BaseGrid grid1(basis, origin, GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z);
        BaseGrid grid2(basis, origin, GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z);
        
        // Initialize both grids
        auto zero_event1 = grid1.zero();
        zero_event1.wait();
        auto add_event1 = grid1.add_scalar(2.0f);
        add_event1.wait();
        
        auto zero_event2 = grid2.zero();
        zero_event2.wait();
        auto add_event2 = grid2.add_scalar(3.0f);
        add_event2.wait();
        
        // Test grid addition
        auto add_grid_event = grid1.add_grid(grid2);
        add_grid_event.wait();
        
        // Test grid multiplication
        auto mult_grid_event = grid1.multiply_grid(grid2);
        mult_grid_event.wait();
        
        REQUIRE(true);  // Should complete without error
    }
}
