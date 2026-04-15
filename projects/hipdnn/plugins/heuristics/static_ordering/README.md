# StaticOrdering Heuristic Plugin

## Overview

The **StaticOrdering** heuristic plugin implements the legacy static engine ordering policy for hipDNN. This plugin wraps the existing `hipdnn_data_sdk::utilities::sortEngineIds` function and exposes it through the heuristic plugin C ABI defined in RFC 0007.

## Purpose

This plugin provides backward-compatible engine ordering semantics:
- Prioritizes `MIOPEN_ENGINE`
- Deprioritizes `MIOPEN_ENGINE_DETERMINISTIC`
- Maintains consistent ordering for other engines

It serves as:
1. **Default fallback policy** in the heuristics framework
2. **Reference implementation** for the heuristic plugin C ABI
3. **Regression baseline** ensuring ordering behavior matches pre-RFC implementation

## Well-Known Policy

**Policy Name**: `SelectionHeuristic::StaticOrdering`
**Policy ID**: `0x123ABC...` (computed via FNV-1a hash of name)

This is a well-known policy listed in the default policy order:
```cpp
std::vector<std::string> defaultPolicies = {
    "SelectionHeuristic::Config",           // Try graph preferences first
    "SelectionHeuristic::StaticOrdering"    // Fallback to static ordering
};
```

## C ABI Implementation

This plugin implements the full heuristic C ABI from RFC 0007 Section 8:

### Module Metadata
- `hipdnnHeuristicGetApiVersion()` - Returns "1.0.0"
- `hipdnnHeuristicGetPolicyId()` - Returns FNV-1a hash of policy name
- `hipdnnHeuristicGetPolicyName()` - Returns "SelectionHeuristic::StaticOrdering"
- `hipdnnHeuristicGetPluginVersion()` - Returns plugin implementation version

### Logging
- `hipdnnHeuristicSetLoggingCallback()` - Sets backend logging callback
- `hipdnnHeuristicSetLogLevel()` - Sets minimum log severity

### Handle Lifecycle
- `hipdnnHeuristicHandleCreate()` - Creates plugin session handle
- `hipdnnHeuristicHandleDestroy()` - Destroys plugin session handle
- `hipdnnHeuristicHandleSetDeviceProperties()` - Stores device properties (not used by StaticOrdering)

### Policy Descriptor Lifecycle
- `hipdnnHeuristicPolicyDescriptorCreate()` - Creates policy descriptor per slot
- `hipdnnHeuristicPolicyDescriptorDestroy()` - Destroys policy descriptor

### Policy Inputs
- `hipdnnHeuristicPolicySetEngineIds()` - Receives candidate engine IDs
- `hipdnnHeuristicPolicySetSerializedGraph()` - Receives serialized graph (not used by StaticOrdering)

### Selection
- `hipdnnHeuristicPolicyFinalize()` - Performs static sorting, returns success
- `hipdnnHeuristicPolicyGetSortedEngineIds()` - Returns sorted engine IDs

## Algorithm

```cpp
void finalize() {
    sortedEngineIds = candidateEngineIds;  // Copy input
    hipdnn_data_sdk::utilities::sortEngineIds(sortedEngineIds);  // Apply static ordering
}
```

The `sortEngineIds` function applies deterministic rules to prioritize certain engines.

## Applicability

StaticOrdering **always** applies (never returns `HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE`):
- Works with any graph
- Requires no device properties
- Accepts any candidate engine list (including empty - returns empty)

## Build

```bash
cd projects/hipdnn/build
cmake -GNinja ..
ninja hipdnn_heuristic_static_ordering
```

The plugin is built as:
- **Library**: `libhipdnn_heuristic_static_ordering.so`
- **Install path**: `${CMAKE_INSTALL_LIBDIR}/hipdnn_plugins/heuristics/`
- **Build copy**: Automatically copied to `build/lib/hipdnn_plugins/heuristics/` for testing

## Installation

```bash
ninja install
```

Installs the plugin to the default heuristic plugin search path used by `HeuristicPluginManager`.

## Testing

The plugin is loaded automatically when:
1. `HeuristicPluginResourceManager` is created
2. The plugin is in the search path (`hipdnn_plugins/heuristics/`)
3. The plugin passes validation (API version, policy ID uniqueness)

Test that it loads correctly:
```cpp
auto handle = std::make_shared<hipdnnHandle>();
auto heurRm = handle->getHeuristicPluginResourceManager();
auto policies = heurRm->getHeuristicPolicyInfos();
// Should include StaticOrdering in the list
```

## Dependencies

- **hipdnn_plugin_sdk**: Provides C ABI types and declarations
- **hipdnn_data_sdk**: Provides `utilities::sortEngineIds` and `engineNameToId`

## Version Compatibility

- **Plugin API version**: 1.0.0
- **Minimum backend version**: Requires RFC 0007 implementation
- **Schema compatibility**: N/A (doesn't parse device properties or graph)

## Logging

Logs are emitted through the backend logging callback:
- `[StaticOrdering] Handle created` - On handle creation
- `[StaticOrdering] PolicyFinalize: sorted N engines using static ordering` - On successful finalize
- Errors logged at `HIPDNN_SEVERITY_ERROR`
- Debug info logged at `HIPDNN_SEVERITY_DEBUG`

## RFC Reference

This plugin implements requirements from:
- **RFC 0007 Section 3**: Baseline current behavior
- **RFC 0007 Section 5.3**: Well-known policy names and IDs
- **RFC 0007 Section 8**: C ABI for heuristic plugins
- **RFC 0007 Section 9**: Policy plugins and outer loop
- **RFC 0007 Section 17**: Testing - regression test baseline
