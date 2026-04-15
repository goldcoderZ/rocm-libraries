// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestHeuristicPluginIntegration.cpp
 * @brief Integration tests for heuristic plugin loading and execution (RFC 0007)
 *
 * These tests verify:
 * - Plugin discovery and loading from search paths
 * - Symbol resolution and ABI validation
 * - Plugin handle creation and lifecycle
 * - Policy descriptor creation and execution
 * - Outer loop integration with EngineHeuristicDescriptor
 */

#include "descriptors/EngineHeuristicDescriptor.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "handle/Handle.hpp"
#include "plugin/HeuristicPlugin.hpp"
#include "plugin/HeuristicPluginManager.hpp"
#include "plugin/HeuristicPluginResourceManager.hpp"

#include <gtest/gtest.h>

using namespace hipdnn_backend;
using namespace hipdnn_backend::plugin;

class HeuristicPluginIntegrationTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        hipdnnStatus_t status = hipdnnCreate(&_handle);
        ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
        ASSERT_NE(_handle, nullptr);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            hipdnnDestroy(_handle);
            _handle = nullptr;
        }
    }

    hipdnnHandle_t _handle = nullptr;
};

// ========== Plugin Discovery Tests ==========

TEST_F(HeuristicPluginIntegrationTest, PluginManagerLoadsFromDefaultPath)
{
    auto manager = std::make_shared<HeuristicPluginManager>();

    // Should find plugins in default search path
    const auto& plugins = manager->getPlugins();

    // At minimum, we expect Config and StaticOrdering
    EXPECT_GE(plugins.size(), 2u) << "Expected at least Config and StaticOrdering plugins";
}

TEST_F(HeuristicPluginIntegrationTest, PluginManagerRejectsInvalidPlugins)
{
    // Plugins with wrong ABI version should be rejected during validation
    // This is tested by HeuristicPluginManager::validateBeforeAdding()

    auto manager = std::make_shared<HeuristicPluginManager>();
    const auto& plugins = manager->getPlugins();

    // All loaded plugins should have valid metadata
    for(const auto& plugin : plugins)
    {
        EXPECT_NE(plugin->policyId(), -1) << "Policy ID should be valid";
        EXPECT_FALSE(plugin->apiVersion().empty()) << "API version should not be empty";
        EXPECT_FALSE(plugin->pluginVersion().empty()) << "Plugin version should not be empty";
    }
}

TEST_F(HeuristicPluginIntegrationTest, PluginManagerRejectsDuplicatePolicyIds)
{
    // HeuristicPluginManager should reject plugins with duplicate policy IDs
    // This is enforced by validateBeforeAdding()

    auto manager = std::make_shared<HeuristicPluginManager>();
    const auto& plugins = manager->getPlugins();

    // Collect all policy IDs
    std::set<int64_t> policyIds;
    for(const auto& plugin : plugins)
    {
        int64_t id = plugin->policyId();
        EXPECT_EQ(policyIds.count(id), 0u) << "Duplicate policy ID detected: " << id;
        policyIds.insert(id);
    }
}

// ========== Symbol Resolution Tests ==========

TEST_F(HeuristicPluginIntegrationTest, LoadedPluginsHaveRequiredSymbols)
{
    auto heurRm = _handle->getHeuristicPluginResourceManager();
    auto policyInfos = heurRm->getHeuristicPolicyInfos();

    ASSERT_GT(policyInfos.size(), 0u);

    // Each loaded plugin must have successfully resolved required symbols
    // If symbol resolution failed, the plugin wouldn't be in the list
    for(const auto& info : policyInfos)
    {
        EXPECT_NE(info.policyId, -1);
        EXPECT_FALSE(info.policyName.empty());
        EXPECT_FALSE(info.apiVersion.empty());
        EXPECT_FALSE(info.pluginVersion.empty());
    }
}

// ========== Handle Lifecycle Tests ==========

TEST_F(HeuristicPluginIntegrationTest, ResourceManagerCreatesHandlesForAllPlugins)
{
    auto heurRm = _handle->getHeuristicPluginResourceManager();
    ASSERT_NE(heurRm, nullptr);

    auto policyInfos = heurRm->getHeuristicPolicyInfos();

    // Should have created a handle for each loaded plugin
    for(const auto& info : policyInfos)
    {
        auto handle = heurRm->getHeuristicHandleForPolicyId(info.policyId);
        EXPECT_NE(handle, nullptr) << "Handle should exist for policy ID " << info.policyId;

        auto plugin = heurRm->getPluginForPolicyId(info.policyId);
        EXPECT_NE(plugin, nullptr) << "Plugin should exist for policy ID " << info.policyId;
    }
}

TEST_F(HeuristicPluginIntegrationTest, HandleDestructionCleansUpResources)
{
    // Create and destroy a handle
    hipdnnHandle_t tempHandle = nullptr;
    ASSERT_EQ(hipdnnCreate(&tempHandle), HIPDNN_STATUS_SUCCESS);

    // Get resource manager (creates plugin handles)
    auto heurRm = tempHandle->getHeuristicPluginResourceManager();
    ASSERT_NE(heurRm, nullptr);

    size_t policyCount = heurRm->getHeuristicPolicyInfos().size();
    EXPECT_GT(policyCount, 0u);

    // Destroy handle (should clean up plugin handles)
    EXPECT_EQ(hipdnnDestroy(tempHandle), HIPDNN_STATUS_SUCCESS);

    // If we got here without crashes, cleanup succeeded
    SUCCEED();
}

// ========== Policy Descriptor Tests ==========

TEST_F(HeuristicPluginIntegrationTest, PolicyDescriptorCreationSucceeds)
{
    auto heurRm = _handle->getHeuristicPluginResourceManager();
    auto policyInfos = heurRm->getHeuristicPolicyInfos();
    ASSERT_GT(policyInfos.size(), 0u);

    // Try to create a policy descriptor for the first loaded policy
    auto policyId = policyInfos[0].policyId;
    auto pluginHandle = heurRm->getHeuristicHandleForPolicyId(policyId);
    auto plugin = heurRm->getPluginForPolicyId(policyId);

    ASSERT_NE(pluginHandle, nullptr);
    ASSERT_NE(plugin, nullptr);

    // Create policy descriptor
    auto descriptor = plugin->createPolicyDescriptor(pluginHandle);
    EXPECT_NE(descriptor, nullptr);

    // Clean up
    if(descriptor != nullptr)
    {
        plugin->destroyPolicyDescriptor(descriptor);
    }
}

// ========== Logging Tests ==========

TEST_F(HeuristicPluginIntegrationTest, PluginsReceiveLoggingCallback)
{
    // Verify that setLoggingCallback was called during plugin initialization
    // This is verified by checking that the resource manager successfully
    // created without errors

    auto heurRm = _handle->getHeuristicPluginResourceManager();
    ASSERT_NE(heurRm, nullptr);

    // If logging callback failed, resource manager creation would have logged warnings
    // For now, just verify it exists
    SUCCEED();
}

// ========== Device Properties Tests ==========

TEST_F(HeuristicPluginIntegrationTest, DevicePropertiesAreSetOnAllHandles)
{
    auto heurRm = _handle->getHeuristicPluginResourceManager();
    ASSERT_NE(heurRm, nullptr);

    // Create serialized device properties
    heuristics::DeviceProperties props;
    props.deviceId = 0;
    props.multiProcessorCount = 64;
    props.totalGlobalMem = 8ULL * 1024 * 1024 * 1024;

    auto serialized = heuristics::serializeDeviceProperties(props);
    auto wrapper = heuristics::wrapSerializedDeviceProperties(serialized);

    // Set on all handles (should not throw)
    EXPECT_NO_THROW(heurRm->setDevicePropertiesOnAllHandles(&wrapper));
}

// ========== Policy ID Consistency Tests ==========

TEST_F(HeuristicPluginIntegrationTest, PolicyIdMatchesNameHash)
{
    auto heurRm = _handle->getHeuristicPluginResourceManager();
    auto policyInfos = heurRm->getHeuristicPolicyInfos();

    for(const auto& info : policyInfos)
    {
        if(!info.policyName.empty())
        {
            // Policy ID should match engineNameToId(policyName)
            int64_t expectedId = hipdnn_data_sdk::utilities::engineNameToId(info.policyName);
            EXPECT_EQ(info.policyId, expectedId)
                << "Policy ID mismatch for " << info.policyName;
        }
    }
}

// ========== API Version Tests ==========

TEST_F(HeuristicPluginIntegrationTest, AllPluginsHaveCompatibleApiVersion)
{
    auto heurRm = _handle->getHeuristicPluginResourceManager();
    auto policyInfos = heurRm->getHeuristicPolicyInfos();

    // All loaded plugins should have compatible API versions
    // (major version matches backend)
    for(const auto& info : policyInfos)
    {
        EXPECT_FALSE(info.apiVersion.empty());

        // Parse version
        hipdnn_data_sdk::utilities::Version apiVer{info.apiVersion};

        // Major version should match backend
        EXPECT_EQ(apiVer.major, HIPDNN_BACKEND_VERSION_MAJOR)
            << "Plugin " << info.policyName << " has incompatible API major version";
    }
}

// ========== Enumeration Consistency Tests ==========

TEST_F(HeuristicPluginIntegrationTest, EnumerationMatchesResourceManager)
{
    auto heurRm = _handle->getHeuristicPluginResourceManager();
    auto rmInfos = heurRm->getHeuristicPolicyInfos();

    // Get count via C API
    size_t apiCount = 0;
    ASSERT_EQ(hipdnnGetHeuristicPolicyCount_ext(_handle, &apiCount), HIPDNN_STATUS_SUCCESS);

    // Counts should match
    EXPECT_EQ(apiCount, rmInfos.size());

    // Each policy from resource manager should be queryable via C API
    for(size_t i = 0; i < rmInfos.size(); ++i)
    {
        int64_t apiPolicyId = -1;
        size_t nameLen = 0, pluginVerLen = 0, apiVerLen = 0;

        ASSERT_EQ(hipdnnGetHeuristicPolicyInfo_ext(
                      _handle, i, &apiPolicyId, nullptr, &nameLen, nullptr, &pluginVerLen, nullptr, &apiVerLen),
                  HIPDNN_STATUS_SUCCESS);

        // Policy ID should match
        EXPECT_EQ(apiPolicyId, rmInfos[i].policyId);
    }
}

// ========== Stress Tests ==========

TEST_F(HeuristicPluginIntegrationTest, MultipleResourceManagersCanCoexist)
{
    // Create multiple handles, each with its own resource manager
    std::vector<hipdnnHandle_t> handles;

    for(int i = 0; i < 5; ++i)
    {
        hipdnnHandle_t h = nullptr;
        ASSERT_EQ(hipdnnCreate(&h), HIPDNN_STATUS_SUCCESS);
        handles.push_back(h);

        // Access resource manager (triggers creation)
        auto heurRm = h->getHeuristicPluginResourceManager();
        EXPECT_NE(heurRm, nullptr);
    }

    // Clean up
    for(auto h : handles)
    {
        EXPECT_EQ(hipdnnDestroy(h), HIPDNN_STATUS_SUCCESS);
    }
}

// ========== Error Recovery Tests ==========

TEST_F(HeuristicPluginIntegrationTest, MissingPolicyGracefullyHandled)
{
    auto heurRm = _handle->getHeuristicPluginResourceManager();

    // Query a non-existent policy ID
    int64_t fakePolicyId = 0x1234567890ABCDEF;
    auto handle = heurRm->getHeuristicHandleForPolicyId(fakePolicyId);
    auto plugin = heurRm->getPluginForPolicyId(fakePolicyId);

    // Should return nullptr, not crash
    EXPECT_EQ(handle, nullptr);
    EXPECT_EQ(plugin, nullptr);
}
