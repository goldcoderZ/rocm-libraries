# Config Heuristic Plugin

## Overview

The **Config** heuristic plugin implements configuration-based engine selection by interpreting graph-level preferences. This plugin allows users to directly specify engine ordering preferences in their operation graphs through the `preferred_engine_id` field.

## Purpose

This plugin provides user-directed engine selection:
- Honors `preferred_engine_id` field from the serialized graph
- Reorders candidates to prioritize the preferred engine
- Maintains relative order of non-preferred engines
- Returns `NOT_APPLICABLE` if no preference is set or preferred engine not in candidates

It serves as:
1. **First policy in default order** - user preferences take precedence
2. **Graph-level preference interpreter** - implements RFC 0007 Section 13.3
3. **User control mechanism** - allows applications to guide engine selection

## Well-Known Policy

**Policy Name**: `SelectionHeuristic::Config`
**Policy ID**: `0xABC123...` (computed via FNV-1a hash of name)

This is a well-known policy listed FIRST in the default policy order:
```cpp
std::vector<std::string> defaultPolicies = {
    "SelectionHeuristic::Config",           // Try graph preferences first
    "SelectionHeuristic::StaticOrdering"    // Fallback to static ordering
};
```

## C ABI Implementation

Implements the full heuristic C ABI from RFC 0007 Section 8 (same as StaticOrdering).

## Algorithm

```cpp
void finalize() {
    // Parse serialized graph FlatBuffer
    auto graph = GetGraph(serializedGraphBuffer);

    // Extract preferred_engine_id
    int64_t preferredEngineId = graph->preferred_engine_id();

    // Check if preferred engine is in candidates
    if (not in candidates) {
        return NOT_APPLICABLE;  // Policy declines
    }

    // Reorder: preferred first, others maintain relative order
    sortedEngineIds = [preferredEngineId] + (candidates - preferredEngineId);

    return SUCCESS;
}
```

## Applicability

Config applies **conditionally**:
- ✅ **Applies** if:
  - Graph has `preferred_engine_id` field set (non-null)
  - Preferred engine is in the candidate list

- ❌ **Not applicable** (`returns 0`) if:
  - No serialized graph provided
  - Graph has no `preferred_engine_id` set (null/default)
  - Preferred engine not in candidates
  - Empty candidate list

## Graph Schema

Uses the `preferred_engine_id` field from `graph.fbs`:

```flatbuffers
table Graph {
    name: string;
    compute_data_type: DataType;
    intermediate_data_type: DataType;
    io_data_type: DataType;
    tensors: [TensorAttributes];
    nodes: [Node];
    preferred_engine_id: long = null;  // <-- This field
}
```

## Example Usage

**Application code (using frontend)**:
```cpp
// Create operation graph
hipdnn_frontend::Graph graph;
// ... build operations ...

// Set preferred engine
graph.setPreferredEngineId(MIOPEN_ENGINE_ID);

// Finalize - Config policy will prioritize MIOPEN_ENGINE
graph.finalize();
auto engineConfigs = graph.getEngineConfigs();
// engineConfigs[0] will be MIOPEN_ENGINE (if applicable)
```

**Result**:
- **Without Config plugin**: Engines ordered by StaticOrdering
- **With Config plugin**: Preferred engine first, then StaticOrdering for rest

## Build

```bash
cd projects/hipdnn/build
cmake -GNinja ..
ninja hipdnn_heuristic_config
```

The plugin is built as:
- **Library**: `libhipdnn_heuristic_config.so`
- **Install path**: `${CMAKE_INSTALL_LIBDIR}/hipdnn_plugins/heuristics/`
- **Build copy**: Automatically copied to `build/lib/hipdnn_plugins/heuristics/` for testing

## Testing

Test graph with preferred engine:
```cpp
// Create graph with preferred_engine_id set
auto serializedGraph = createGraphWithPreferredEngine(MIOPEN_ENGINE_ID);

// Config should apply and reorder
auto heurDesc = createEngineHeuristicDescriptor(graph);
heurDesc->finalize();
auto results = heurDesc->getEngineIds();
// results[0] should be MIOPEN_ENGINE_ID
```

Test graph without preference:
```cpp
// Create graph with preferred_engine_id = null
auto serializedGraph = createGraphNoPreference();

// Config should decline (not applicable)
// StaticOrdering will apply instead
auto heurDesc = createEngineHeuristicDescriptor(graph);
heurDesc->finalize();  // Config returns 0, StaticOrdering returns 1
auto results = heurDesc->getEngineIds();
// results ordered by StaticOrdering
```

## Dependencies

- **hipdnn_plugin_sdk**: Provides C ABI types
- **hipdnn_data_sdk**: Provides FlatBuffers schema and `engineNameToId`
- **flatbuffers**: For parsing serialized graph

## Version Compatibility

- **Plugin API version**: 1.0.0
- **Minimum backend version**: Requires RFC 0007 implementation
- **Schema compatibility**: Requires `graph.fbs` with `preferred_engine_id` field

## Logging

Logs are emitted through the backend logging callback:
- `[Config] PolicyFinalize: reordered N engines with preferred engine 0x... first` - On success
- `[Config] PolicyFinalize: preferred_engine_id 0x... not in candidates - not applicable` - When declining
- Errors logged at `HIPDNN_SEVERITY_ERROR`
- Debug info logged at `HIPDNN_SEVERITY_DEBUG`

## RFC Reference

This plugin implements requirements from:
- **RFC 0007 Section 5.3**: Well-known policy names and IDs
- **RFC 0007 Section 9**: Policy plugins and outer loop
- **RFC 0007 Section 13.1**: Serialized graph input
- **RFC 0007 Section 13.3**: Graph-level preferences interpretation (critical requirement)
- **RFC 0007 Section 8**: C ABI for heuristic plugins

## Design Notes

**Why Config runs before StaticOrdering:**
- User preferences should take precedence over hard-coded defaults
- Config can decline (not applicable), allowing fallback to StaticOrdering
- Ensures user intent is respected when specified

**Why Config is separate from backend logic:**
- RFC 0007 Section 13.3 explicitly assigns preference interpretation to policies
- Keeps backend generic and extensible
- Allows future preference-based policies (cache-aware, ML-based, etc.)
