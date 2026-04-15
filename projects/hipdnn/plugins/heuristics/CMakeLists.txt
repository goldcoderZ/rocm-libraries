# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Heuristic Plugins for RFC 0007 Engine Selection Framework

# Well-known default policies (in order)
add_subdirectory(config)           # First: honors graph preferences
add_subdirectory(static_ordering)  # Second: fallback static ordering
