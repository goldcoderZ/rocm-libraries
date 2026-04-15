// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file ConfigPlugin.cpp
 * @brief Config heuristic plugin - honors graph-level preferences
 *
 * This plugin implements configuration-based engine selection by interpreting
 * graph-level preferences such as preferred_engine_id. It allows users to
 * directly specify engine ordering preferences in their operation graphs.
 *
 * RFC 0007 Section 13.3: Graph-level preferences are interpreted by this policy,
 * not by hard-coded backend logic.
 */

#include <hipdnn_plugin_sdk/HeuristicsPluginApi.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>

#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/verifier.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

// Plugin version and identification
static constexpr const char* PLUGIN_VERSION = "1.0.0";
static constexpr const char* API_VERSION = "1.0.0";
static constexpr const char* POLICY_NAME = "SelectionHeuristic::Config";

// Policy ID computed from name using FNV-1a hash
static const int64_t POLICY_ID = hipdnn_data_sdk::utilities::engineNameToId(POLICY_NAME);

// Logging callback and log level
static hipdnnHeuristicLoggingCallback_t g_loggingCallback = nullptr;
static hipdnnSeverity_t g_logLevel = HIPDNN_SEV_INFO;

// Log buffer size
constexpr size_t LOG_BUFFER_SIZE = 1024;

// Helper macro for logging
#define CONFIG_LOG(severity, ...)                                             \
    do                                                                         \
    {                                                                          \
        if(g_loggingCallback != nullptr && (severity) >= g_logLevel)          \
        {                                                                      \
            char buffer[LOG_BUFFER_SIZE];                                      \
            snprintf(buffer, sizeof(buffer), __VA_ARGS__);                     \
            g_loggingCallback(severity, "[Config] ", buffer);                  \
        }                                                                      \
    } while(0)

//=============================================================================
// Plugin Handle Implementation
//=============================================================================

struct ConfigHandle
{
    // Device properties (stored when SetDeviceProperties is called)
    std::vector<uint8_t> devicePropertiesBuffer;
    bool devicePropertiesSet = false;

    ConfigHandle()
    {
        CONFIG_LOG(HIPDNN_SEV_INFO, "Handle created");
    }

    ~ConfigHandle()
    {
        CONFIG_LOG(HIPDNN_SEV_INFO, "Handle destroyed");
    }
};

//=============================================================================
// Policy Descriptor Implementation
//=============================================================================

struct ConfigPolicyDescriptor
{
    ConfigHandle* handle = nullptr;
    std::vector<int64_t> candidateEngineIds;
    std::vector<uint8_t> serializedGraphBuffer;
    std::vector<int64_t> sortedEngineIds;
    bool finalized = false;

    ConfigPolicyDescriptor(ConfigHandle* h) : handle(h)
    {
        CONFIG_LOG(HIPDNN_SEV_INFO, "Policy descriptor created");
    }

    ~ConfigPolicyDescriptor()
    {
        CONFIG_LOG(HIPDNN_SEV_INFO, "Policy descriptor destroyed");
    }
};

//=============================================================================
// C ABI Implementation - Module Metadata
//=============================================================================

extern "C" {

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicGetApiVersion(const char** version)
{
    if(version == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *version = API_VERSION;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnHeuristicGetPolicyId(int64_t* policy_id)
{
    if(policy_id == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *policy_id = POLICY_ID;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnHeuristicGetPolicyName(const char** policy_name)
{
    if(policy_name == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *policy_name = POLICY_NAME;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicGetPluginVersion(const char** version)
{
    if(version == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *version = PLUGIN_VERSION;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

//=============================================================================
// C ABI Implementation - Logging
//=============================================================================

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicSetLoggingCallback(hipdnnHeuristicLoggingCallback_t callback)
{
    g_loggingCallback = callback;
    CONFIG_LOG(HIPDNN_SEV_INFO, "Logging callback set");
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnHeuristicSetLogLevel(hipdnnSeverity_t level)
{
    g_logLevel = level;
    CONFIG_LOG(HIPDNN_SEV_INFO, "Log level set to %d", level);
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

//=============================================================================
// C ABI Implementation - Handle Lifecycle
//=============================================================================

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicHandleCreate(hipdnnHeuristicHandle_t* out_handle)
{
    if(out_handle == nullptr)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "HandleCreate: null output pointer");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

    try
    {
        auto handle = new ConfigHandle();
        *out_handle = reinterpret_cast<hipdnnHeuristicHandle_t>(handle);
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "HandleCreate failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicHandleDestroy(hipdnnHeuristicHandle_t handle)
{
    if(handle == nullptr)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "HandleDestroy: null handle");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

    try
    {
        auto h = reinterpret_cast<ConfigHandle*>(handle);
        delete h;
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "HandleDestroy failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnHeuristicHandleSetDeviceProperties(
    hipdnnHeuristicHandle_t handle, const hipdnnPluginConstData_t* device_props_serialized)
{
    if(handle == nullptr)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "SetDeviceProperties: null handle");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

    if(device_props_serialized == nullptr || device_props_serialized->ptr == nullptr
       || device_props_serialized->size == 0)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "SetDeviceProperties: invalid buffer");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

    try
    {
        auto h = reinterpret_cast<ConfigHandle*>(handle);

        // Store device properties (Config doesn't use them currently, but store for completeness)
        const auto* data = reinterpret_cast<const uint8_t*>(device_props_serialized->ptr);
        h->devicePropertiesBuffer.assign(data, data + device_props_serialized->size);
        h->devicePropertiesSet = true;

        CONFIG_LOG(HIPDNN_SEV_INFO,
                   "Device properties set (%zu bytes)",
                   device_props_serialized->size);
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "SetDeviceProperties failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

//=============================================================================
// C ABI Implementation - Policy Descriptor Lifecycle
//=============================================================================

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnHeuristicPolicyDescriptorCreate(
    hipdnnHeuristicHandle_t plugin_handle, hipdnnHeuristicPolicyDescriptor_t* out_desc)
{
    if(plugin_handle == nullptr)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "PolicyDescriptorCreate: null handle");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

    if(out_desc == nullptr)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "PolicyDescriptorCreate: null output pointer");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

    try
    {
        auto h = reinterpret_cast<ConfigHandle*>(plugin_handle);
        auto desc = new ConfigPolicyDescriptor(h);
        *out_desc = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(desc);
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "PolicyDescriptorCreate failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicPolicyDescriptorDestroy(hipdnnHeuristicPolicyDescriptor_t desc)
{
    if(desc == nullptr)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "PolicyDescriptorDestroy: null descriptor");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

    try
    {
        auto d = reinterpret_cast<ConfigPolicyDescriptor*>(desc);
        delete d;
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "PolicyDescriptorDestroy failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

//=============================================================================
// C ABI Implementation - Policy Inputs
//=============================================================================

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnHeuristicPolicySetEngineIds(
    hipdnnHeuristicPolicyDescriptor_t desc, const int64_t* engine_ids, size_t engine_id_count)
{
    if(desc == nullptr)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "SetEngineIds: null descriptor");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

    if(engine_ids == nullptr && engine_id_count > 0)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "SetEngineIds: null engine_ids with count > 0");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

    try
    {
        auto d = reinterpret_cast<ConfigPolicyDescriptor*>(desc);
        d->candidateEngineIds.assign(engine_ids, engine_ids + engine_id_count);
        d->finalized = false; // Reset finalized state when inputs change

        CONFIG_LOG(HIPDNN_SEV_INFO, "SetEngineIds: %zu candidate engines", engine_id_count);
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "SetEngineIds failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnHeuristicPolicySetSerializedGraph(
    hipdnnHeuristicPolicyDescriptor_t desc, const hipdnnPluginConstData_t* serialized_graph)
{
    if(desc == nullptr)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "SetSerializedGraph: null descriptor");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

    if(serialized_graph == nullptr || serialized_graph->ptr == nullptr)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "SetSerializedGraph: null graph pointer");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

    try
    {
        auto d = reinterpret_cast<ConfigPolicyDescriptor*>(desc);

        // Store serialized graph for parsing during finalize
        const auto* data = reinterpret_cast<const uint8_t*>(serialized_graph->ptr);
        d->serializedGraphBuffer.assign(data, data + serialized_graph->size);

        CONFIG_LOG(HIPDNN_SEV_INFO,
                   "SetSerializedGraph: graph stored (%zu bytes)",
                   serialized_graph->size);
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "SetSerializedGraph failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

//=============================================================================
// C ABI Implementation - Selection
//=============================================================================

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicPolicyFinalize(hipdnnHeuristicPolicyDescriptor_t desc, int32_t* out_applied)
{
    if(desc == nullptr)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "PolicyFinalize: null descriptor");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

    if(out_applied == nullptr)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "PolicyFinalize: null output pointer");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

    try
    {
        auto d = reinterpret_cast<ConfigPolicyDescriptor*>(desc);

        if(d->candidateEngineIds.empty())
        {
            CONFIG_LOG(HIPDNN_SEV_WARN, "PolicyFinalize: no candidate engines - not applicable");
            *out_applied = 0;
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }

        if(d->serializedGraphBuffer.empty())
        {
            CONFIG_LOG(HIPDNN_SEV_WARN, "PolicyFinalize: no serialized graph - not applicable");
            *out_applied = 0;
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }

        // Parse the serialized graph using FlatBuffers
        flatbuffers::Verifier verifier(d->serializedGraphBuffer.data(), d->serializedGraphBuffer.size());
        if(!verifier.VerifyBuffer<hipdnn_data_sdk::data_objects::Graph>())
        {
            CONFIG_LOG(HIPDNN_SEV_ERROR, "PolicyFinalize: invalid graph buffer");
            return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
        }

        auto graph = hipdnn_data_sdk::data_objects::GetGraph(d->serializedGraphBuffer.data());
        if(graph == nullptr)
        {
            CONFIG_LOG(HIPDNN_SEV_ERROR, "PolicyFinalize: failed to get graph root");
            return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
        }

        // Check if graph has a preferred_engine_id
        // In FlatBuffers, optional int64 fields use flatbuffers::Optional
        // If null, the field was not set
        auto preferredEngineIdOpt = graph->preferred_engine_id();

        if(!preferredEngineIdOpt.has_value())
        {
            // No preference set - policy not applicable
            CONFIG_LOG(HIPDNN_SEV_INFO, "PolicyFinalize: no preferred_engine_id set - not applicable");
            *out_applied = 0;
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }

        int64_t preferredEngineId = preferredEngineIdOpt.value();

        // Check if the preferred engine ID is in the candidates
        auto it = std::find(d->candidateEngineIds.begin(),
                           d->candidateEngineIds.end(),
                           preferredEngineId);

        if(it == d->candidateEngineIds.end())
        {
            // Preferred engine not in candidates - policy not applicable
            CONFIG_LOG(HIPDNN_SEV_INFO,
                      "PolicyFinalize: preferred_engine_id 0x%llx not in candidates - not applicable",
                      static_cast<long long>(preferredEngineId));
            *out_applied = 0;
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }

        // Reorder candidates: preferred engine first, others maintain relative order
        d->sortedEngineIds.clear();
        d->sortedEngineIds.reserve(d->candidateEngineIds.size());

        // Add preferred engine first
        d->sortedEngineIds.push_back(preferredEngineId);

        // Add remaining engines in original order
        for(int64_t engineId : d->candidateEngineIds)
        {
            if(engineId != preferredEngineId)
            {
                d->sortedEngineIds.push_back(engineId);
            }
        }

        d->finalized = true;
        *out_applied = 1; // Policy succeeded

        CONFIG_LOG(HIPDNN_SEV_INFO,
                   "PolicyFinalize: reordered %zu engines with preferred engine 0x%llx first",
                   d->sortedEngineIds.size(),
                   static_cast<long long>(preferredEngineId));
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "PolicyFinalize failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnHeuristicPolicyGetSortedEngineIds(
    hipdnnHeuristicPolicyDescriptor_t desc, int64_t* engine_ids, uint32_t* num_engines)
{
    if(desc == nullptr)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "GetSortedEngineIds: null descriptor");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

    if(num_engines == nullptr)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "GetSortedEngineIds: null num_engines pointer");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

    try
    {
        auto d = reinterpret_cast<ConfigPolicyDescriptor*>(desc);

        if(!d->finalized)
        {
            CONFIG_LOG(HIPDNN_SEV_ERROR, "GetSortedEngineIds: descriptor not finalized");
            return HIPDNN_PLUGIN_STATUS_NOT_INITIALIZED;
        }

        // Return count only if engine_ids is null
        if(engine_ids == nullptr)
        {
            *num_engines = static_cast<uint32_t>(d->sortedEngineIds.size());
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }

        // Copy sorted engine IDs
        uint32_t count = std::min(*num_engines, static_cast<uint32_t>(d->sortedEngineIds.size()));
        std::copy_n(d->sortedEngineIds.begin(), count, engine_ids);
        *num_engines = count;

        CONFIG_LOG(HIPDNN_SEV_INFO, "GetSortedEngineIds: returned %u engines", count);
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        CONFIG_LOG(HIPDNN_SEV_ERROR, "GetSortedEngineIds failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

} // extern "C"
