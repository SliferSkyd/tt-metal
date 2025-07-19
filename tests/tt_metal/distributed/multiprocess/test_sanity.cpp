// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <impl/context/metal_context.hpp>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/fabric_types.hpp>
#include <tt-metalium/host_buffer.hpp>

#include <tt-metalium/tt_metal.hpp>

namespace tt::tt_metal::distributed {

using tt_fabric::HostRankId;
using tt_fabric::MeshId;
using tt_fabric::MeshScope;

TEST(BigMeshDualRankTestT3K, DistributedContext) {
    auto& dctx = MetalContext::instance().get_distributed_context();
    auto world_size = dctx.size();
    EXPECT_EQ(*world_size, 2);
}

TEST(BigMeshDualRankTestT3K, LocalRankBinding) {
    auto& dctx = MetalContext::instance().get_distributed_context();
    auto& control_plane = MetalContext::instance().get_control_plane();

    tt_fabric::HostRankId local_rank_binding = control_plane.get_local_host_rank_id_binding();
    if (*dctx.rank() == 0) {
        EXPECT_EQ(*local_rank_binding, 0);
    } else {
        EXPECT_EQ(*local_rank_binding, 1);
    }
}

TEST(BigMeshDualRankTestT3K, SystemMeshValidation) {
    EXPECT_NO_THROW({
        const auto& system_mesh = SystemMesh::instance();
        EXPECT_EQ(system_mesh.shape(), MeshShape(2,4));
        EXPECT_EQ(system_mesh.local_shape(), MeshShape(2,2));
    });
}

// Parameterized test fixture for mesh device validation
class BigMeshDualRankTestT3KFixture : public ::testing::Test, public ::testing::WithParamInterface<MeshShape> {};

TEST_P(BigMeshDualRankTestT3KFixture, MeshDeviceValidation) {
    MeshShape mesh_shape = GetParam();
    auto mesh_device = MeshDevice::create(MeshDeviceConfig(mesh_shape), DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, tt::tt_metal::DispatchCoreType::WORKER);
    EXPECT_EQ(mesh_device->shape(), mesh_shape);
}

INSTANTIATE_TEST_SUITE_P(
    MeshDeviceTests,
    BigMeshDualRankTestT3KFixture,
    ::testing::Values(
        MeshShape(2, 4),
        /* Issue #25355: Cannot create a MeshDevice with only one rank active.
        MeshShape(1, 1),
        MeshShape(1, 2),
        MeshShape(2, 1),
        MeshShape(2, 2),
        */
        MeshShape(1, 8),
        MeshShape(8, 1)
    )
);

TEST(BigMeshDualRankTestT3K, SystemMeshShape) {
    const auto& system_mesh = SystemMesh::instance();
    EXPECT_EQ(system_mesh.local_shape(), MeshShape(2, 2));

    auto& control_plane = MetalContext::instance().get_control_plane();
    auto rank = control_plane.get_local_host_rank_id_binding();

    if (rank == HostRankId{0}) {
        EXPECT_NO_THROW(system_mesh.get_physical_device_id(MeshCoordinate(0, 0)));
        EXPECT_NO_THROW(system_mesh.get_physical_device_id(MeshCoordinate(0, 1)));
        EXPECT_NO_THROW(system_mesh.get_physical_device_id(MeshCoordinate(1, 0)));
        EXPECT_NO_THROW(system_mesh.get_physical_device_id(MeshCoordinate(1, 1)));
    } else {
        EXPECT_NO_THROW(system_mesh.get_physical_device_id(MeshCoordinate(0, 2)));
        EXPECT_NO_THROW(system_mesh.get_physical_device_id(MeshCoordinate(0, 3)));
        EXPECT_NO_THROW(system_mesh.get_physical_device_id(MeshCoordinate(1, 2)));
        EXPECT_NO_THROW(system_mesh.get_physical_device_id(MeshCoordinate(1, 3)));
    }
}

TEST(BigMeshDualRankTestT3K, DistributedHostBuffer) {
    auto mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape(2, 4)));
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    DistributedHostBuffer host_buffer = DistributedHostBuffer::create(mesh_device->get_view());
    auto rank = control_plane.get_local_host_rank_id_binding();
    const auto EXPECTED_RANK_VALUE = (rank == HostRankId{0}) ? 0 : 1;

    host_buffer.emplace_shard(MeshCoordinate(0, 0), []() { return HostBuffer(std::vector<int>{0, 0, 0}); });
    host_buffer.emplace_shard(MeshCoordinate(0, 1), []() { return HostBuffer(std::vector<int>{0, 0, 0}); });
    host_buffer.emplace_shard(MeshCoordinate(1, 0), []() { return HostBuffer(std::vector<int>{0, 0, 0}); });
    host_buffer.emplace_shard(MeshCoordinate(1, 1), []() { return HostBuffer(std::vector<int>{0, 0, 0}); });

    host_buffer.emplace_shard(MeshCoordinate(0, 2), []() { return HostBuffer(std::vector<int>{1, 1, 1}); });
    host_buffer.emplace_shard(MeshCoordinate(0, 3), []() { return HostBuffer(std::vector<int>{1, 1, 1}); });
    host_buffer.emplace_shard(MeshCoordinate(1, 2), []() { return HostBuffer(std::vector<int>{1, 1, 1}); });
    host_buffer.emplace_shard(MeshCoordinate(1, 3), []() { return HostBuffer(std::vector<int>{1, 1, 1}); });

    auto validate_local_shards = [EXPECTED_RANK_VALUE](const HostBuffer& buffer) {
        fmt::print("Rank {}: {}\n", EXPECTED_RANK_VALUE, std::vector<int>(buffer.view_as<int>().begin(), buffer.view_as<int>().end()));
        auto span = buffer.view_as<int>();
        for (const auto& value : span) {
            EXPECT_EQ(value, EXPECTED_RANK_VALUE);
        }
    };

    host_buffer.apply(validate_local_shards);
}

/*
TEST(BigMeshDualRankTestT3K, TensorReadWriteLoopback) {
    auto mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape(2, 4)));

    const tt::tt_metal::Shape shape{1, 1, 32, 32};
    const tt::tt_metal::TensorSpec tensor_spec =
        tt::tt_metal::TensorSpec(shape, tt::tt_metal::TensorLayout(tt::tt_metal::DataType::FLOAT32, tt::tt_metal::Layout::ROW_MAJOR, tt::tt_metal::MemoryConfig{}));

    std::vector<float> host_data(shape.volume());
    std::iota(host_data.begin(), host_data.end(), 0);

    // Prepare host tensor to offload on device.
    tt::tt_metal::Tensor input_host_tensor = tt::tt_metal::Tensor::from_vector(host_data, tensor_spec);
    EXPECT_TRUE(input_host_tensor.storage_type() == tt::tt_metal::StorageType::HOST);
    EXPECT_EQ(input_host_tensor.tensor_spec().logical_shape(), shape);

    // Write host tensor to device.
    tt::tt_metal::Tensor device_tensor =
        tt::tt_metal::tensor_impl::to_device_mesh_tensor_wrapper(input_host_tensor, mesh_device.get(), tt::tt_metal::MemoryConfig{});
    EXPECT_EQ(device_tensor.tensor_spec().logical_shape(), shape);

    auto* device_storage = std::get_if<tt::tt_metal::DeviceStorage>(&device_tensor.storage());
    ASSERT_NE(device_storage, nullptr);
    EXPECT_NE(device_storage->mesh_buffer, nullptr);
    EXPECT_THAT(device_storage->coords, SizeIs(mesh_device->num_devices()));

    // Read the tensor back, and compare it with input data.
    tt::tt_metal::Tensor output_host_tensor = tt::tt_metal::tensor_impl::to_host_mesh_tensor_wrapper(device_tensor);
    EXPECT_TRUE(output_host_tensor.storage_type() == tt::tt_metal::StorageType::MULTI_DEVICE_HOST);
    EXPECT_EQ(output_host_tensor.tensor_spec().logical_shape(), shape);

    // TODO: Re-evaluate this. ttnn::distributed::get_device_tensors only returns local set
    auto tensors = ttnn::distributed::get_device_tensors(output_host_tensor);
    EXPECT_THAT(tensors, SizeIs(mesh_device->local_shape().mesh_size()));

    for (const auto& tensor : tensors) {
        EXPECT_EQ(tensor.tensor_spec().logical_shape(), shape);
        EXPECT_THAT(tensor.to_vector<float>(), ::testing::Pointwise(::testing::FloatEq(), host_data));
    }
}
*/

}  // namespace tt::tt_metal::distributed
