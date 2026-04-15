// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "SelectionHeuristic.hpp"

#include "HipdnnException.hpp"
#include "plugin/HeuristicPlugin.hpp"

namespace hipdnn_backend::heuristics
{

SelectionHeuristic::SelectionHeuristic(const plugin::HeuristicPlugin* plugin,
                                       hipdnnHeuristicHandle_t pluginHandle)
    : _plugin(plugin), _descriptor(nullptr)
{
    THROW_IF_FALSE(_plugin != nullptr,
                   HIPDNN_STATUS_BAD_PARAM,
                   "HeuristicPlugin pointer cannot be null");
    THROW_IF_FALSE(pluginHandle != nullptr,
                   HIPDNN_STATUS_BAD_PARAM,
                   "hipdnnHeuristicHandle_t cannot be null");

    // Create the policy descriptor bound to this plugin handle
    _descriptor = _plugin->createPolicyDescriptor(pluginHandle);
}

SelectionHeuristic::~SelectionHeuristic()
{
    if(_descriptor != nullptr && _plugin != nullptr)
    {
        try
        {
            _plugin->destroyPolicyDescriptor(_descriptor);
        }
        catch(const HipdnnException& e)
        {
            // Log but don't throw from destructor
            // The plugin's logging callback should have already reported this
        }
        _descriptor = nullptr;
    }
}

SelectionHeuristic::SelectionHeuristic(SelectionHeuristic&& other) noexcept
    : _plugin(other._plugin), _descriptor(other._descriptor)
{
    other._plugin     = nullptr;
    other._descriptor = nullptr;
}

SelectionHeuristic& SelectionHeuristic::operator=(SelectionHeuristic&& other) noexcept
{
    if(this != &other)
    {
        // Clean up current descriptor
        if(_descriptor != nullptr && _plugin != nullptr)
        {
            try
            {
                _plugin->destroyPolicyDescriptor(_descriptor);
            }
            catch(...)
            {
                // Ignore exceptions in move assignment cleanup
            }
        }

        // Move from other
        _plugin           = other._plugin;
        _descriptor       = other._descriptor;
        other._plugin     = nullptr;
        other._descriptor = nullptr;
    }
    return *this;
}

void SelectionHeuristic::setEngineIds(const std::vector<int64_t>& engineIds)
{
    THROW_IF_FALSE(_descriptor != nullptr,
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "Policy descriptor not initialized");

    _plugin->setEngineIds(_descriptor, engineIds.data(), engineIds.size());
}

void SelectionHeuristic::setSerializedGraph(const hipdnnPluginConstData_t* serializedGraph)
{
    THROW_IF_FALSE(_descriptor != nullptr,
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "Policy descriptor not initialized");
    THROW_IF_FALSE(serializedGraph != nullptr,
                   HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                   "Serialized graph pointer cannot be null");

    _plugin->setSerializedGraph(_descriptor, serializedGraph);
}

bool SelectionHeuristic::finalize()
{
    THROW_IF_FALSE(_descriptor != nullptr,
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "Policy descriptor not initialized");

    // Call the plugin's finalize method
    // Returns true if policy succeeded (won the outer loop)
    // Returns false if not applicable or declined
    return _plugin->finalize(_descriptor);
}

std::vector<int64_t> SelectionHeuristic::getSortedEngineIds()
{
    THROW_IF_FALSE(_descriptor != nullptr,
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "Policy descriptor not initialized");

    // Call the plugin's getSortedEngineIds method
    // This is valid only after finalize() returned true
    return _plugin->getSortedEngineIds(_descriptor);
}

} // namespace hipdnn_backend::heuristics
