// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "EngineHeuristicDescriptor.hpp"
#include "BackendEnumStringUtils.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "EngineConfigDescriptor.hpp"
#include "EngineDescriptor.hpp"
#include "GraphDescriptor.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "ScopedDescriptor.hpp"
#include "handle/Handle.hpp"
#include "utilities/EngineOrdering.hpp"

// RFC 0007: Heuristics framework
#include "heuristics/DeviceProperties.hpp"
#include "heuristics/SelectionHeuristic.hpp"
#include "plugin/HeuristicPlugin.hpp"
#include "plugin/HeuristicPluginResourceManager.hpp"
#include "logging/Logging.hpp"

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

#include <cstdlib>
#include <set>
#include <sstream>
#include <unordered_map>

namespace hipdnn_backend
{

std::vector<int64_t> EngineHeuristicDescriptor::resolveHeuristicPolicyOrder()
{
    // RFC 0007 Section 5.3 & 15: Policy order resolution
    // Priority: descriptor attr > handle > env > default
    std::vector<std::string> policyNames;

    // 1. Descriptor attribute (highest priority)
    if(_policyOrderSet)
    {
        policyNames = _policyOrder;
        HIPDNN_BACKEND_LOG_DEBUG(
            "Using descriptor-level policy order: {} policies", policyNames.size());
    }
    // 2. Handle-level override (TODO: implement handle API)
    // else if (handle has override)
    // {
    //     policyNames = handle->getHeuristicPolicyOrder();
    // }
    // 3. Environment variable HIPDNN_HEURISTIC_POLICY_ORDER
    else if(const char* envPolicyOrder = std::getenv("HIPDNN_HEURISTIC_POLICY_ORDER"))
    {
        // Parse comma-separated policy names
        std::string envStr(envPolicyOrder);
        std::istringstream iss(envStr);
        std::string token;
        while(std::getline(iss, token, ','))
        {
            // Trim whitespace
            token.erase(0, token.find_first_not_of(" \t\n\r"));
            token.erase(token.find_last_not_of(" \t\n\r") + 1);
            if(!token.empty())
            {
                policyNames.push_back(token);
            }
        }
        HIPDNN_BACKEND_LOG_DEBUG(
            "Using environment variable policy order: {} policies", policyNames.size());
    }
    // 4. Default policy list per RFC 0007 Section 5.3
    else
    {
        policyNames = {
            "SelectionHeuristic::Config",
            "SelectionHeuristic::StaticOrdering"
        };
        HIPDNN_BACKEND_LOG_DEBUG("Using default policy order: {} policies", policyNames.size());
    }

    // Convert policy names to IDs using engineNameToId (FNV-1a hash)
    std::vector<int64_t> policyIds;
    policyIds.reserve(policyNames.size());
    for(const auto& name : policyNames)
    {
        policyIds.push_back(hipdnn_data_sdk::utilities::engineNameToId(name));
    }

    return policyIds;
}

void EngineHeuristicDescriptor::syncPolicySlots(const std::vector<int64_t>& orderedPolicyIds)
{
    // RFC 0007 Section 14.2: Ensure one SelectionHeuristic per policy slot
    // If the policy list changed, recreate the slots

    if(_orderedPolicyIds == orderedPolicyIds && !_policySlots.empty())
    {
        // Policy list unchanged and slots already created
        return;
    }

    _orderedPolicyIds = orderedPolicyIds;
    _policySlots.clear();

    auto handle = _graph->getHandle();
    auto heurRm = handle->getHeuristicPluginResourceManager();

    // Create one SelectionHeuristic per policy slot
    for(int64_t policyId : orderedPolicyIds)
    {
        auto pluginHandle = heurRm->getHeuristicHandleForPolicyId(policyId);
        if(pluginHandle == nullptr)
        {
            // Policy not loaded - add null placeholder
            _policySlots.push_back(nullptr);
            continue;
        }

        // Get the HeuristicPlugin for this policy
        auto plugin = heurRm->getPluginForPolicyId(policyId);
        if(plugin == nullptr)
        {
            // This shouldn't happen if handle exists, but be defensive
            _policySlots.push_back(nullptr);
            continue;
        }

        // Create SelectionHeuristic for this policy slot
        _policySlots.push_back(std::make_unique<heuristics::SelectionHeuristic>(plugin, pluginHandle));
    }
}

void EngineHeuristicDescriptor::finalize()
{
    // RFC 0007 Section 14.2: Outer loop policy selection
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "EngineHeuristicDescriptor::finalize() failed: Already finalized.");

    THROW_IF_NULL(_graph,
                  HIPDNN_STATUS_BAD_PARAM,
                  "EngineHeuristicDescriptor::finalize() failed: Graph is not set.");

    THROW_IF_FALSE(_heuristicModeSet,
                   HIPDNN_STATUS_BAD_PARAM,
                   "EngineHeuristicDescriptor::finalize() failed: Heuristic mode is not set.");

    auto handle = _graph->getHandle();
    auto engineRm = handle->getPluginResourceManager();
    auto heurRm = handle->getHeuristicPluginResourceManager();

    // Get candidate engine IDs from engine plugins
    auto candidates = engineRm->getApplicableEngineIds(_graph.get(), _findFirst);

    // RFC 0007 Section 6 & 13.2: Query and serialize device properties
    auto devProps = heuristics::queryDeviceProperties();
    auto devicePropsSerialized = heuristics::serializeDeviceProperties(devProps);
    hipdnnPluginConstData_t devicePropsWrapper =
        heuristics::wrapSerializedDeviceProperties(devicePropsSerialized);

    // RFC 0007 Section 13.1: Get serialized graph from GraphDescriptor
    hipdnnPluginConstData_t serializedGraph = _graph->getSerializedGraph();

    // Resolve ordered policy IDs
    auto orderedPolicyIds = resolveHeuristicPolicyOrder();

    // Ensure policy slots match the ordered policy list
    syncPolicySlots(orderedPolicyIds);

    // Set device properties on all distinct plugin handles (once per handle, not per slot)
    // NOTE: Multiple policies may share the same plugin handle if they come from the same
    // .so file (e.g., a single plugin providing multiple ordering strategies like Fast/Balanced/Accurate).
    // Use a map to deduplicate and call setDeviceProperties only once per unique handle.
    std::unordered_map<hipdnnHeuristicHandle_t, const plugin::HeuristicPlugin*> distinctHandles;
    for(int64_t policyId : orderedPolicyIds)
    {
        auto pluginHandle = heurRm->getHeuristicHandleForPolicyId(policyId);
        if(pluginHandle != nullptr)
        {
            auto plugin = heurRm->getPluginForPolicyId(policyId);
            if(plugin != nullptr)
            {
                distinctHandles[pluginHandle] = plugin;
            }
        }
    }

    // Call SetDeviceProperties on each distinct handle
    for(const auto& [pluginHandle, plugin] : distinctHandles)
    {
        plugin->setDeviceProperties(pluginHandle, &devicePropsWrapper);
    }

    // Outer policy loop: try each policy in order until one succeeds
    bool success = false;
    for(size_t i = 0; i < _policySlots.size(); ++i)
    {
        auto& selection = _policySlots[i];
        if(selection == nullptr)
        {
            // Policy plugin not loaded - continue to next policy
            continue;
        }

        try
        {
            // Set candidate engine IDs and serialized graph on the policy descriptor
            selection->setEngineIds(candidates);
            selection->setSerializedGraph(&serializedGraph);

            // Call finalize on this policy
            if(!selection->finalize())
            {
                // Policy declined or not applicable - continue to next policy
                continue;
            }

            // Policy succeeded! Get the sorted engine IDs
            candidates = selection->getSortedEngineIds();
            success = true;
            break;
        }
        catch(const HipdnnException& e)
        {
            // Policy threw an exception - log and continue to next policy
            HIPDNN_BACKEND_LOG_WARN(
                "Heuristic policy at slot {} (ID {}) threw exception: {}. Continuing to next policy.",
                i,
                _orderedPolicyIds[i],
                e.what());
            continue;
        }
    }

    // RFC 0007 Section 14.2: If no policy succeeded, throw exception
    // No hidden fallback to utilities::sortEngineIds
    if(!success)
    {
        throw HipdnnException(
            HIPDNN_STATUS_INTERNAL_ERROR,
            "EngineHeuristicDescriptor::finalize() failed: No heuristic policy succeeded.");
    }

    _engineIds = candidates;

    HipdnnBackendDescriptorImpl<EngineHeuristicDescriptor>::finalize();
}

void EngineHeuristicDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                             hipdnnBackendAttributeType_t attributeType,
                                             int64_t requestedElementCount,
                                             int64_t* elementCount,
                                             void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
                   "EngineHeuristicDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH:
        getGraph(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINEHEUR_MODE:
        getHeuristicMode(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINEHEUR_RESULTS:
        getEngineConfigs(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINEHEUR_FIND_FIRST_EXT:
        getFindFirst(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT:
        getPolicyOrder(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("EngineHeuristicDescriptor::getAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
    }
}

void EngineHeuristicDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                             hipdnnBackendAttributeType_t attributeType,
                                             int64_t elementCount,
                                             const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "EngineHeuristicDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_ENGINEHEUR_MODE:
        setHeuristicMode(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH:
        setGraph(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINEHEUR_FIND_FIRST_EXT:
        setFindFirst(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT:
        setPolicyOrder(attributeType, elementCount, arrayOfElements);
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("EngineHeuristicDescriptor::setAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
    }
}

void EngineHeuristicDescriptor::setHeuristicMode(hipdnnBackendAttributeType_t attributeType,
                                                 int64_t elementCount,
                                                 const void* arrayOfElements)
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_HEUR_MODE,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineHeuristicDescriptor failed to set heuristic mode: Invalid attribute type.");

    THROW_IF_NE(elementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineHeuristicDescriptor failed to set heuristic mode: Invalid element count.");

    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "EngineHeuristicDescriptor failed to set heuristic mode: Null pointer.");

    auto heurMode = static_cast<const hipdnnBackendHeurMode_t*>(arrayOfElements);
    THROW_IF_NULL(
        heurMode,
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
        "EngineHeuristicDescriptor failed to set heuristic mode: Heuristic mode is null.");

    auto heurModeValue = *heurMode;
    if(heurModeValue != HIPDNN_HEUR_MODE_FALLBACK)
    {
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "EngineHeuristicDescriptor::setHeuristicMode() is not "
                              "supported for the given heuristic mode.");
    }

    _heuristicMode = heurModeValue;
    _heuristicModeSet = true;
}

void EngineHeuristicDescriptor::setGraph(hipdnnBackendAttributeType_t attributeType,
                                         int64_t elementCount,
                                         const void* arrayOfElements)
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineHeuristicDescriptor failed to set graph: Invalid attribute type.");

    THROW_IF_NE(elementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineHeuristicDescriptor failed to set graph: Invalid element count.");

    auto graph = HipdnnBackendDescriptor::unpackDescriptor<const GraphDescriptor>(
        arrayOfElements,
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
        "EngineHeuristicDescriptor failed to set graph: Null pointer.");

    THROW_IF_FALSE(graph->isFinalized(),
                   HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
                   "EngineHeuristicDescriptor failed to set graph: Graph is not finalized.");

    _graph = graph;
}

void EngineHeuristicDescriptor::getGraph(hipdnnBackendAttributeType_t attributeType,
                                         int64_t requestedElementCount,
                                         int64_t* elementCount,
                                         void* arrayOfElements) const
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineHeuristicDescriptor failed to get graph: Invalid attribute type.");

    THROW_IF_NE(requestedElementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineHeuristicDescriptor failed to get graph: Invalid element count.");

    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "EngineHeuristicDescriptor failed to get graph: Null pointer.");

    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }

    HipdnnBackendDescriptor::packDescriptor(_graph, arrayOfElements);
}

void EngineHeuristicDescriptor::getEngineConfigs(hipdnnBackendAttributeType_t attributeType,
                                                 int64_t requestedElementCount,
                                                 int64_t* elementCount,
                                                 void* arrayOfElements) const
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineHeuristicDescriptor failed to get engine configs: Invalid attribute type.");

    // Return the number of engine configs if they aren't requesting any.
    if(requestedElementCount == 0)
    {
        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "EngineHeuristicDescriptor failed to get engine config count: Null pointer "
                      "for element count.");
        *elementCount = static_cast<int64_t>(_engineIds.size());
    }
    else
    {
        THROW_IF_NULL(arrayOfElements,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "EngineHeuristicDescriptor failed to get engine configs: Null pointer.");

        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "EngineHeuristicDescriptor failed to get engine config: Null pointer for "
                      "element count.");

        // Create engine config descriptors for each engine ID
        auto outputArray = static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements);
        for(size_t i = 0; i < _engineIds.size() && i < static_cast<size_t>(requestedElementCount);
            ++i)
        {
            auto config = HipdnnBackendDescriptor::unpackDescriptor<EngineConfigDescriptor>(
                outputArray[i],
                HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                "EngineHeuristicDescriptor failed to get engine config: Config "
                "descriptor is null.");

            auto engine = std::make_shared<EngineDescriptor>();

            engine->setAttribute(
                HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, &_engineIds[i]);

            ScopedDescriptor graphDesc(HipdnnBackendDescriptor::packDescriptor(_graph));
            engine->setAttribute(HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                 HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                 1,
                                 static_cast<const void*>(graphDesc.getPtr()));
            engine->finalize();

            ScopedDescriptor engineDesc(HipdnnBackendDescriptor::packDescriptor(engine));
            config->setAttribute(HIPDNN_ATTR_ENGINECFG_ENGINE,
                                 HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                 1,
                                 static_cast<const void*>(engineDesc.getPtr()));
        }

        *elementCount = std::min(requestedElementCount, static_cast<int64_t>(_engineIds.size()));
    }
}

void EngineHeuristicDescriptor::getHeuristicMode(hipdnnBackendAttributeType_t attributeType,
                                                 int64_t requestedElementCount,
                                                 int64_t* elementCount,
                                                 void* arrayOfElements) const
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_HEUR_MODE,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineHeuristicDescriptor failed to get heuristic mode: Invalid attribute type.");

    THROW_IF_NE(requestedElementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineHeuristicDescriptor failed to get heuristic mode: Invalid element count.");

    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "EngineHeuristicDescriptor failed to get heuristic mode: Null pointer.");

    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }

    auto heurModeOut = static_cast<hipdnnBackendHeurMode_t*>(arrayOfElements);
    *heurModeOut = _heuristicMode;
}

void EngineHeuristicDescriptor::setFindFirst(hipdnnBackendAttributeType_t attributeType,
                                             int64_t elementCount,
                                             const void* arrayOfElements)
{
    setScalar(_findFirst,
              HIPDNN_TYPE_BOOLEAN,
              attributeType,
              elementCount,
              arrayOfElements,
              "EngineHeuristicDescriptor::setAttribute()");
}

void EngineHeuristicDescriptor::getFindFirst(hipdnnBackendAttributeType_t attributeType,
                                             int64_t requestedElementCount,
                                             int64_t* elementCount,
                                             void* arrayOfElements) const
{
    getScalar(_findFirst,
              HIPDNN_TYPE_BOOLEAN,
              attributeType,
              requestedElementCount,
              elementCount,
              arrayOfElements,
              "EngineHeuristicDescriptor::getAttribute()");
}

std::shared_ptr<const GraphDescriptor> EngineHeuristicDescriptor::getGraph() const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_INTERNAL_ERROR,
                   "EngineHeuristicDescriptor::getGraph() failed: Not finalized.");

    return _graph;
}

hipdnnBackendDescriptorType_t EngineHeuristicDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR;
}

void EngineHeuristicDescriptor::setPolicyOrder(hipdnnBackendAttributeType_t attributeType,
                                               int64_t elementCount,
                                               const void* arrayOfElements)
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_CHAR,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineHeuristicDescriptor failed to set policy order: Invalid attribute type.");

    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "EngineHeuristicDescriptor failed to set policy order: Null pointer.");

    // Parse null-separated string array
    const char* data = static_cast<const char*>(arrayOfElements);
    _policyOrder.clear();

    size_t offset = 0;
    while(offset < static_cast<size_t>(elementCount))
    {
        std::string policyName(data + offset);
        if(policyName.empty())
        {
            break; // End of list
        }
        _policyOrder.push_back(policyName);
        offset += policyName.size() + 1; // Skip null terminator
    }

    _policyOrderSet = true;
    HIPDNN_BACKEND_LOG_DEBUG(
        "Set descriptor-level policy order: {} policies", _policyOrder.size());
}

void EngineHeuristicDescriptor::getPolicyOrder(hipdnnBackendAttributeType_t attributeType,
                                               int64_t requestedElementCount,
                                               int64_t* elementCount,
                                               void* arrayOfElements) const
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_CHAR,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineHeuristicDescriptor failed to get policy order: Invalid attribute type.");

    if(!_policyOrderSet)
    {
        // Return empty if not set
        if(elementCount != nullptr)
        {
            *elementCount = 0;
        }
        return;
    }

    // Calculate total size needed for null-separated string array
    // Format: "string1\0string2\0string3\0\0"
    // Each string ends in \0, final extra \0 marks end of array
    size_t totalSize = 0;
    for(const auto& name : _policyOrder)
    {
        totalSize += name.size() + 1; // Include null terminator
    }
    totalSize += 1; // Final null terminator

    // Return the count if they aren't requesting any
    if(requestedElementCount == 0)
    {
        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "EngineHeuristicDescriptor failed to get policy order count: Null pointer "
                      "for element count.");
        *elementCount = static_cast<int64_t>(totalSize);
        return;
    }

    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "EngineHeuristicDescriptor failed to get policy order: Null pointer.");

    THROW_IF_NULL(elementCount,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "EngineHeuristicDescriptor failed to get policy order: Null pointer for "
                  "element count.");

    // Copy null-separated string array
    char* output = static_cast<char*>(arrayOfElements);
    size_t offset = 0;
    for(const auto& name : _policyOrder)
    {
        if(offset + name.size() + 1 > static_cast<size_t>(requestedElementCount))
        {
            break; // Buffer too small
        }
        std::memcpy(output + offset, name.c_str(), name.size() + 1);
        offset += name.size() + 1;
    }

    // Add final null terminator
    if(offset < static_cast<size_t>(requestedElementCount))
    {
        output[offset] = '\0';
        offset++;
    }

    *elementCount = static_cast<int64_t>(offset);
}

std::string EngineHeuristicDescriptor::toString() const
{
    std::string str = "EngineHeuristicDescriptor: {heuristicMode=";
    str += _heuristicModeSet ? std::to_string(_heuristicMode) : "unset";
    str += _graph ? ", graph=" + fmt::format("{:p}", static_cast<const void*>(_graph.get()))
                  : ", graph=null";
    if(_policyOrderSet)
    {
        str += ", policyOrder=[";
        for(size_t i = 0; i < _policyOrder.size(); ++i)
        {
            if(i > 0) str += ", ";
            str += "\"" + _policyOrder[i] + "\"";
        }
        str += "]";
    }
    str += "}";
    return str;
}

} // namespace hipdnn_backend
