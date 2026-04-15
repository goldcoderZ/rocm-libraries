// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <hipdnn_plugin_sdk/HeuristicsPluginApi.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

namespace hipdnn_backend::plugin
{
class HeuristicPlugin;
}

namespace hipdnn_backend::heuristics
{

/**
 * @brief C++ facade for one policy slot on EngineHeuristicDescriptor.
 *
 * This class wraps a hipdnnHeuristicPolicyDescriptor_t created with a
 * hipdnnHeuristicHandle_t for that policy's module. It provides a clean
 * C++ interface over the heuristic plugin C ABI.
 *
 * Session state (caches, tuning data, etc.) lives in the plugin behind the
 * handle, not in this wrapper.
 *
 * Device properties are NOT set on this facade. The host calls
 * hipdnnHeuristicHandleSetDeviceProperties on the handle BEFORE calling
 * finalize() on any descriptor created with that handle.
 *
 * Lifecycle: Owned by EngineHeuristicDescriptor, one per resolved policy slot.
 * Created when the policy list is established, destroyed with the descriptor.
 *
 * RFC Reference: Section 7
 */
class SelectionHeuristic
{
public:
    /**
     * @brief Constructs a SelectionHeuristic for a given policy.
     *
     * Creates the underlying hipdnnHeuristicPolicyDescriptor_t by calling
     * hipdnnHeuristicPolicyDescriptorCreate with the plugin handle.
     *
     * @param plugin Pointer to the HeuristicPlugin that implements this policy.
     *               Must remain valid for the lifetime of this object.
     * @param pluginHandle The hipdnnHeuristicHandle_t for this policy's module.
     *                     Created by HeuristicPluginResourceManager.
     */
    SelectionHeuristic(const plugin::HeuristicPlugin* plugin,
                       hipdnnHeuristicHandle_t pluginHandle);

    /**
     * @brief Destroys the SelectionHeuristic and releases the policy descriptor.
     *
     * Calls hipdnnHeuristicPolicyDescriptorDestroy on the underlying descriptor.
     */
    ~SelectionHeuristic();

    // Prevent copying
    SelectionHeuristic(const SelectionHeuristic&) = delete;
    SelectionHeuristic& operator=(const SelectionHeuristic&) = delete;

    // Allow moving
    SelectionHeuristic(SelectionHeuristic&& other) noexcept;
    SelectionHeuristic& operator=(SelectionHeuristic&& other) noexcept;

    /**
     * @brief Sets the candidate engine IDs for this policy.
     *
     * Provides the list of candidate engine IDs from
     * EnginePluginResourceManager::getApplicableEngineIds.
     * The plugin must produce a reordered subset or permutation of these IDs.
     *
     * Mirrors hipdnnHeuristicPolicySetEngineIds (§8.8).
     *
     * @param engineIds Vector of candidate engine IDs.
     */
    void setEngineIds(const std::vector<int64_t>& engineIds);

    /**
     * @brief Sets the serialized operation graph for this policy.
     *
     * Provides the FlatBuffer-serialized operation graph from
     * GraphDescriptor::getSerializedGraph().
     *
     * Mirrors hipdnnHeuristicPolicySetSerializedGraph (§8.8).
     *
     * @param serializedGraph Pointer to hipdnnPluginConstData_t containing
     *                        the serialized graph buffer.
     */
    void setSerializedGraph(const hipdnnPluginConstData_t* serializedGraph);

    /**
     * @brief Executes the policy selection logic.
     *
     * Performs applicability checking and engine ordering based on the inputs
     * previously set via setEngineIds and setSerializedGraph.
     *
     * Device properties are queried from the bound hipdnnHeuristicHandle_t
     * (which received SetDeviceProperties earlier in the finalize() flow).
     *
     * Two-phase design: This function performs the selection work;
     * getSortedEngineIds retrieves the result.
     *
     * Mirrors hipdnnHeuristicPolicyFinalize (§8.9).
     *
     * @return true if policy succeeded (policy won the outer loop),
     *         false if not applicable or declined (host continues outer loop).
     */
    bool finalize();

    /**
     * @brief Retrieves the sorted engine IDs after successful finalize.
     *
     * Valid only after finalize() returned true. Returns the reordered
     * engine IDs produced by the policy.
     *
     * The output IDs are a permutation or subset of the input IDs from
     * setEngineIds. The host validates this constraint.
     *
     * Mirrors hipdnnHeuristicPolicyGetSortedEngineIds (§8.9).
     *
     * @return Vector of sorted engine IDs.
     */
    std::vector<int64_t> getSortedEngineIds();

private:
    const plugin::HeuristicPlugin* _plugin;
    hipdnnHeuristicPolicyDescriptor_t _descriptor;
};

} // namespace hipdnn_backend::heuristics
