# Split Button Functionality: Correlation Analysis Enhancement

## Overview

This document details the implementation of split correlation button functionality in the WBFM (Whole Body Fluorescence Microscopy) GUI system. The enhancement separates previously ambiguous correlation behavior into two distinct, predictable buttons.

## Problem Statement

### Before Implementation
- **Single Button**: "Correlation to current trace" with conditional logic
- **Unpredictable Behavior**: Sometimes used current neuron, sometimes reference trace
- **User Confusion**: No clear indication of which trace would be used
- **Mixed Responsibilities**: One method handling two different use cases

### After Implementation
- **Two Distinct Buttons**: Clear separation of functionality
- **Predictable Behavior**: Each button has a single, well-defined purpose
- **User Clarity**: Obvious choice between current neuron vs reference trace correlation

## Architecture Overview

### Information Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    WBFM GUI Main Interface                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐  ┌─────────────────────────────────────┐ │
│  │ Current Neuron      │  │ Reference Trace Dropdown            │ │
│  │ Selection Dropdown  │  │ ┌─────────────────────────────────┐ │ │
│  │ ┌─────────────────┐ │  │ │ • None                          │ │ │
│  │ │ • neuron_001    │ │  │ │ • neuron_001, neuron_002, ...   │ │ │
│  │ │ • neuron_002    │ │  │ │ • forward, backward, pause, ... │ │ │
│  │ │ • neuron_003    │ │  │ │ • custom:timeseries1, ...       │ │ │
│  │ │ • ...           │ │  │ └─────────────────────────────────┘ │ │
│  │ └─────────────────┘ │  └─────────────────────────────────────┘ │
│  └─────────────────────┘                                        │
│                                                                 │
│  ┌─────────────────────┐  ┌─────────────────────────────────────┐ │
│  │ [Correlation to     │  │ [Correlation to                     │ │
│  │  current trace]     │  │  reference trace]                   │ │
│  │        │            │  │            │                        │ │
│  └────────┼────────────┘  └────────────┼────────────────────────┘ │
│           │                            │                         │
└───────────┼────────────────────────────┼─────────────────────────┘
            │                            │
            ▼                            ▼
┌───────────────────────┐    ┌─────────────────────────────────────┐
│ Method:               │    │ Method:                             │
│ add_layer_colored_    │    │ add_layer_colored_by_correlation_   │
│ by_correlation_to_    │    │ to_reference_trace()               │
│ current_neuron()      │    │                                     │
│                       │    │ ┌─────────────────────────────────┐ │
│ ┌───────────────────┐ │    │ │ 1. Get dropdown selection       │ │
│ │ 1. Get current    │ │    │ │ 2. Validate selection != "None" │ │
│ │    neuron data    │ │    │ │ 3. Calculate trace data         │ │
│ │ 2. Use y_trace_   │ │    │ │ 4. Handle custom timeseries     │ │
│ │    mode directly  │ │    │ │ 5. Perform correlation          │ │
│ │ 3. Perform        │ │    │ │ 6. Create visualization layer   │ │
│ │    correlation    │ │    │ └─────────────────────────────────┘ │
│ │ 4. Create layer   │ │    └─────────────────────────────────────┘
│ └───────────────────┘ │                            │
└───────────────────────┘                            │
            │                                        │
            └────────────┬───────────────────────────┘
                         │
                         ▼
            ┌─────────────────────────────────────────┐
            │ Common Correlation Processing Pipeline  │
            │ ┌─────────────────────────────────────┐ │
            │ │ 1. Data validation & length check  │ │
            │ │ 2. Pandas DataFrame.corrwith(y)    │ │
            │ │ 3. val = val * abs(val) enhancement │ │
            │ │ 4. Heatmap generation               │ │
            │ │ 5. Napari layer creation            │ │
            │ │ 6. Layer reordering                 │ │
            │ └─────────────────────────────────────┘ │
            └─────────────────────────────────────────┘
```

### Data Flow Sequence

#### Button 1: "Correlation to current trace"
```
User Click → add_layer_colored_by_correlation_to_current_neuron()
    ↓
Current Neuron Selection → self.current_neuron_name
    ↓
Trace Data Retrieval → self.y_trace_mode
    ↓
Layer Name Generation → f'correlation_to_current_{neuron}_at_t_{time}'
    ↓
Correlation Calculation → df.corrwith(y)
    ↓
Napari Layer Creation → self.dat.add_layers_to_viewer()
```

#### Button 2: "Correlation to reference trace"
```
User Click → add_layer_colored_by_correlation_to_reference_trace()
    ↓
Dropdown Selection → self.changeReferenceTrace.currentText()
    ↓
Validation Check → if ref_trace_name == "None": return
    ↓
Trace Calculation → self.calculate_trace(trace_name=ref_trace_name)
    ↓
Custom Detection → if ref_trace_name.startswith('custom:')
    ↓
Layer Name Generation → f'correlation_to_reference_{trace}_at_t_{time}'
    ↓
Correlation Calculation → df.corrwith(y)
    ↓
Napari Layer Creation → self.dat.add_layers_to_viewer()
```

## Implementation Details

### File Modified
**Location**: `/Users/benjaminschaar/Documents/GitHub/wbfm/wbfm/gui/utils/napari_trace_explorer.py`

### Code Additions and Modifications

#### 1. GUI Button Setup (Lines 695-697)
**Method**: `_setup_layer_creation_buttons()`
**Location**: Within the "New layer creation" group box setup

```python
# ADDED: New button for reference trace correlation
self.addReferenceTraceHeatmap = QtWidgets.QPushButton("Add Layer")
self.addReferenceTraceHeatmap.pressed.connect(self.add_layer_colored_by_correlation_to_reference_trace)
self.formlayout8.addRow("Correlation to reference trace:", self.addReferenceTraceHeatmap)
```

**Information Flow**: 
- Button creation → Event handler assignment → Form layout integration
- User interaction → Qt signal/slot mechanism → Method execution

#### 2. Modified Current Neuron Correlation (Lines 2492-2530)
**Method**: `add_layer_colored_by_correlation_to_current_neuron()`

**Key Changes**:
```python
# REMOVED: Conditional logic checking for reference traces
# OLD:
# if hasattr(self, 'current_reference_trace_name') and ...
#     y = self.current_reference_trace_data
# else:
#     y = self.y_trace_mode

# NEW: Direct assignment - always use current neuron
y = self.y_trace_mode
correlation_with_name = self.current_neuron_name
```

**Information Flow**:
- Method call → Direct data access (`self.y_trace_mode`) → Correlation processing
- No conditional branching → Predictable behavior → Simplified debugging

#### 3. New Reference Trace Correlation (Lines 2544-2601)
**Method**: `add_layer_colored_by_correlation_to_reference_trace()` (New)

**Information Flow Breakdown**:

##### Step 1: Dropdown Selection Retrieval
```python
ref_trace_name = self.changeReferenceTrace.currentText()
```
- **Data Source**: Qt ComboBox current selection
- **Possible Values**: "None", neuron names, behavior aliases, custom timeseries names
- **Validation**: Early return if "None" selected

##### Step 2: Dynamic Trace Calculation
```python
try:
    t, y = self.calculate_trace(trace_name=ref_trace_name)
except Exception as e:
    # Error handling and graceful exit
```
- **Delegation**: Uses existing `calculate_trace()` infrastructure
- **Supports**: All trace types (neurons, behaviors, custom timeseries)
- **Error Handling**: Catches calculation failures and reports to user

##### Step 3: Correlation Processing
```python
val_to_plot = df.corrwith(y)
val_to_plot = val_to_plot * np.abs(val_to_plot)  # Enhancement
```
- **Algorithm**: Pearson correlation via pandas
- **Enhancement**: Square while preserving sign to de-emphasize small correlations
- **Output**: Correlation coefficients for all neurons vs reference trace

##### Step 4: Visualization Layer Creation
```python
which_layers = [('heatmap', 'custom_val_to_plot', f'correlation_to_reference_{correlation_with_name}_at_t_{self.t}')]
self.dat.add_layers_to_viewer(self.viewer, which_layers=which_layers, ...)
```
- **Layer Type**: Heatmap visualization in Napari
- **Naming Convention**: Includes "reference_" prefix for identification
- **Integration**: Uses existing layer creation infrastructure

### Integration Points

#### Reference Trace Dropdown Integration
**Location**: Lines 624-685 in `_setup_trace_filtering_buttons()`

**Existing Infrastructure Used**:
```python
self.changeReferenceTrace = QtWidgets.QComboBox()
neuron_names_and_none = self.dat.neuron_names.copy()
neuron_names_and_none.insert(0, "None")
# ... behavior aliases and custom timeseries added ...
```

**Information Flow**:
- Neuron data retrieval → Behavior availability checking → Custom timeseries detection → Dropdown population
- User selection → Qt signal → Reference trace update → Available for correlation

#### Trace Calculation Infrastructure
**Method**: `calculate_trace(trace_name)`
**Location**: Lines 2118-2150+ (existing method)

**Supported Trace Types**:
```python
if trace_name in self.dat.neuron_names:
    # Standard neuron trace calculation
elif trace_name in WormFullVideoPosture.beh_aliases_stable():
    # Behavioral data calculation  
elif trace_name.startswith("custom:"):
    # Custom timeseries data retrieval
```

**Information Flow**:
- Trace name input → Type detection → Appropriate data source access → Time series output

#### Napari Layer Management
**Method**: `self.dat.add_layers_to_viewer()`
**Integration**: Existing visualization infrastructure

**Layer Properties**:
- **Type**: Heatmap overlay
- **Data**: Correlation coefficient values
- **Opacity**: 1.0 for clear visualization
- **Ordering**: Manual ID layers moved to top to prevent obscuring

## Error Handling and Edge Cases

### Reference Trace Selection Validation
```python
if ref_trace_name == "None":
    print("❌ ERROR REFERENCE CORRELATION: No reference trace selected (dropdown is set to 'None')")
    return
```
**Information Flow**: Early validation → User feedback → Graceful method termination

### Trace Calculation Error Handling
```python
try:
    t, y = self.calculate_trace(trace_name=ref_trace_name)
    print(f"🔍 DEBUG REFERENCE CORRELATION: Successfully calculated reference trace data, length: {len(y)}")
except Exception as e:
    print(f"❌ ERROR REFERENCE CORRELATION: Failed to calculate reference trace '{ref_trace_name}': {e}")
    return
```
**Information Flow**: Calculation attempt → Exception catching → Error reporting → Graceful exit

### Data Length Validation
```python
if y is not None and df is not None:
    if len(y) != len(df):
        print(f"❌ ERROR REFERENCE CORRELATION: Length mismatch! y={len(y)}, df={len(df)}")
    else:
        print(f"✅ SUCCESS REFERENCE CORRELATION: Lengths match! y={len(y)}, df={len(df)}")
```
**Information Flow**: Data length checking → Mismatch detection → Warning output → Continued processing

## Debug and Monitoring Infrastructure

### Debug Output Enhancement
**Current Correlation**:
```
🔍 DEBUG CURRENT CORRELATION: Starting current neuron correlation calculation...
🔍 DEBUG CURRENT CORRELATION: Using current neuron trace for correlation: 'neuron_001'
✅ SUCCESS CURRENT CORRELATION: Lengths match! y=1500, df=1500
```

**Reference Correlation**:
```
🔍 DEBUG REFERENCE CORRELATION: Starting reference trace correlation calculation...
🔍 DEBUG REFERENCE CORRELATION: Selected reference trace: 'custom:worm1-2025-07-17_conc_at_0'
🔍 DEBUG REFERENCE CORRELATION: Successfully calculated reference trace data, length: 1500
🔍 DEBUG REFERENCE CORRELATION: Correlating with custom timeseries: custom:worm1-2025-07-17_conc_at_0
✅ SUCCESS REFERENCE CORRELATION: Lengths match! y=1500, df=1500
```

**Information Flow**: Method execution → Status checkpoints → Console output → User feedback

### Performance Monitoring
**Metrics Tracked**:
- Trace calculation success/failure rates
- Data length mismatches
- Custom timeseries detection
- Correlation calculation completion

## Layer Naming and Organization

### Naming Convention Changes
**Before**: `correlation_to_{trace_name}_at_t_{time}`
**After**: 
- Current: `correlation_to_current_{neuron_name}_at_t_{time}`
- Reference: `correlation_to_reference_{trace_name}_at_t_{time}`

### Information Flow Impact
```
Layer Creation → Naming Convention Application → Napari Layer List → User Layer Management
```
**Benefit**: Users can easily distinguish correlation types in the GUI layer panel

## Custom Timeseries Integration

### Detection and Handling
```python
if str(correlation_with_name).startswith('custom:'):
    print(f"🔍 DEBUG REFERENCE CORRELATION: Correlating with custom timeseries: {correlation_with_name}")
```

### Information Flow for Custom Data
```
CSV Files in /behavior/custom_timeseries/ 
    ↓
_load_custom_timeseries_csvs() → DataFrame creation
    ↓  
Dropdown population with "custom:" prefix
    ↓
User selection → calculate_trace() delegation
    ↓
Custom timeseries data retrieval → Correlation calculation
```

## Testing and Validation

### Syntax and Import Testing
```bash
python -m py_compile wbfm/gui/utils/napari_trace_explorer.py  # ✅ PASSED
python -c "from wbfm.gui.utils.napari_trace_explorer import NapariTraceExplorer"  # ✅ PASSED
```

### Expected Information Flow Validation
1. **Button clicks** → Correct method execution
2. **Current neuron changes** → Button 1 correlation updates
3. **Reference dropdown changes** → Button 2 correlation availability  
4. **Layer creation** → Proper naming and visualization
5. **Error conditions** → Graceful handling and user feedback

## Future Enhancement Opportunities

### Additional Information Flow Paths
1. **Real-time Updates**: Auto-refresh correlations when data changes
2. **Batch Processing**: Multiple correlation calculations in sequence
3. **Correlation Comparison**: Side-by-side analysis of current vs reference
4. **Export Functionality**: Save correlation data to files
5. **Statistical Summaries**: Correlation distribution analysis

### Architectural Extensibility
- **Plugin Architecture**: Additional correlation algorithms
- **Data Source Integration**: Remote data streams
- **Visualization Options**: Alternative heatmap styles
- **Performance Optimization**: Caching and lazy evaluation

## Conclusion

The split button functionality provides clear separation of concerns with well-defined information flow paths. Users now have predictable, controllable access to both current neuron and reference trace correlation analysis, with comprehensive error handling and debug support throughout the entire data processing pipeline.