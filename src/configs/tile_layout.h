// Per-tile heterogeneity infrastructure.
//
// This header defines a per-tile type tag and a per-die role tag that lets
// the rest of the simulator account for energy and (eventually) latency
// per tile type. The shipped MuchiSim configurations are homogeneous — by
// default all tiles are tagged TILE_TYPE_GPU and the per-type energy
// coefficients in param_energy.h are equal across types, so existing apps
// produce bit-identical output until an app actively overrides the layout.
//
// Heterogeneous apps (e.g. src/apps/llm_inference.h) call
// init_tile_types_per_chiplet({...roles per die...}) before simulation
// starts, which sets the role of every tile inside die N to roles[N]. The
// energy model in calc_energy.h then uses the per-tile-type coefficient
// arrays to attribute dynamic energy correctly across tile types and dies.
//
// This header must be included AFTER configs/config_system.h (which
// resolves GRID_SIZE / DIES / DIE_W / DIE_H / DIE_FACTOR via common/macros.h).
//
// IMPORTANT: this header defines globals. Include it from exactly one
// translation unit (main.cpp).

#ifndef __TILE_LAYOUT_H__
#define __TILE_LAYOUT_H__

// NOTE: this header relies on GRID_SIZE / DIES / DIE_W / DIE_H / DIE_FACTOR
// and the `global(x,y)` / `die_id(x,y)` macros from common/macros.h. We do
// NOT re-include macros.h here because that header has no include guard
// and would conflict with main.cpp's earlier inclusion. The single TU
// that includes this header (main.cpp) is guaranteed to have already
// included common/macros.h.

// ----------------------------------------------------------------------
// Tile type enum
// ----------------------------------------------------------------------

enum tile_type_e {
    TILE_TYPE_CPU   = 0,
    TILE_TYPE_GPU   = 1,
    TILE_TYPE_ACCEL = 2,
    TILE_TYPE_HBM   = 3,
    NUM_TILE_TYPES  = 4,
};

const char * const tile_type_names[NUM_TILE_TYPES] = {
    "cpu", "gpu", "accel", "hbm",
};

// ----------------------------------------------------------------------
// Globals: per-tile and per-die role assignment
// ----------------------------------------------------------------------

// One byte per tile, indexed by the global tile id computed via
// `global(tX, tY)` (see common/macros.h). Default: TILE_TYPE_GPU
// everywhere (preserves homogeneous shipping behavior).
u_int8_t tile_type_array[GRID_SIZE];

// One byte per die, indexed by `die_id(tX, tY)`. Tracks the role chosen
// for each die so per-die energy aggregation can attribute results.
// Default: TILE_TYPE_GPU.
u_int8_t die_role[DIES];

// Whether heterogeneity has been explicitly engaged. Used by
// calc_energy.h to decide whether to emit the new per-type / per-die
// log sections (avoids cluttering homogeneous runs).
bool heterogeneous_layout_enabled = false;

// ----------------------------------------------------------------------
// Per-die / per-tile-type counter aggregates (populated each ACUM print)
// ----------------------------------------------------------------------
//
// Sized using GLOBAL_COUNTERS from common/macros.h (24 entries) and
// either DIES or NUM_TILE_TYPES. Populated in calc_stats.h's
// print_counter_stats() while it walks the per-tile counters[i][j][c]
// arrays, then read by calc_energy.h's print_energy() to emit per-die
// and per-tile-type energy sections in the log.
//
// We use 32 as a generous upper bound for GLOBAL_COUNTERS (actually 24)
// so we don't depend on macros.h having been included before this
// header (it is — main.cpp includes macros.h first — but keeping the
// dependency loose makes future ordering changes safer).
u_int64_t per_die_counters[256][32];        // [die_index][counter_id]
u_int64_t per_type_counters[NUM_TILE_TYPES][32];

inline void reset_hetero_counters() {
    for (u_int32_t d = 0; d < DIES; d++)
        for (u_int32_t c = 0; c < 32; c++)
            per_die_counters[d][c] = 0;
    for (u_int32_t t = 0; t < NUM_TILE_TYPES; t++)
        for (u_int32_t c = 0; c < 32; c++)
            per_type_counters[t][c] = 0;
}

// ----------------------------------------------------------------------
// Initializers
// ----------------------------------------------------------------------

// Default homogeneous initializer: every tile and die is TILE_TYPE_GPU.
// Called automatically at simulation startup. Apps that want
// heterogeneous behavior call init_tile_types_per_chiplet AFTER this.
inline void init_tile_types_homogeneous(u_int8_t default_type = TILE_TYPE_GPU) {
    for (u_int32_t i = 0; i < GRID_SIZE; i++) {
        tile_type_array[i] = default_type;
    }
    for (u_int32_t d = 0; d < DIES; d++) {
        die_role[d] = default_type;
    }
    heterogeneous_layout_enabled = false;
}

// Per-die initializer: assign every tile in die N to roles[N].
// Length of `roles` must equal DIES; values must be in [0, NUM_TILE_TYPES).
//
// Example: init_tile_types_per_chiplet({TILE_TYPE_CPU, TILE_TYPE_GPU,
// TILE_TYPE_ACCEL, TILE_TYPE_HBM}) on a DIE_FACTOR=2 layout assigns the
// four 16x16 dies to four distinct roles, matching the PackageSim
// hetero_4chiplet packages.
inline void init_tile_types_per_chiplet(const u_int8_t * roles) {
    for (u_int32_t d = 0; d < DIES; d++) {
        u_int8_t role = roles[d];
        die_role[d] = role;
    }
    for (u_int32_t tY = 0; tY < GRID_Y; tY++) {
        for (u_int32_t tX = 0; tX < GRID_X; tX++) {
            u_int32_t d = die_id(tX, tY);
            tile_type_array[global(tX, tY)] = roles[d];
        }
    }
    heterogeneous_layout_enabled = true;
}

// Lookup helpers used by accounting code.
inline u_int8_t tile_type_of(u_int32_t tX, u_int32_t tY) {
    return tile_type_array[global(tX, tY)];
}
inline u_int8_t die_role_of(u_int32_t d) {
    return die_role[d];
}

#endif // __TILE_LAYOUT_H__
