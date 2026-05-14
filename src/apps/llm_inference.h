// Synthetic role-aware workload (LLM-inference-flavoured).
//
// Each tile runs a role-specific amount of work LOCALLY (no cross-tile
// dispatch), producing per-die energy that cleanly differentiates by
// role:
//   CPU   tiles do light dispatch  (LLM_T1_CYCLES of int ops)
//   GPU   tiles do heavy matmul    (LLM_T2_CYCLES, counted as FLOPS)
//   ACCEL tiles do memory-bound decode (LLM_T3_CYCLES + HBM-like loads)
//   HBM   tiles stay nearly idle   (LLM_THBM_CYCLES, minimal int ops)
//
// This isn't a faithful LLM data-flow model — it's a stress fixture that
// produces visibly heterogeneous per-die activity in MuchiSim's existing
// queue/router framework without fighting the proxy-routing assumptions.
// The PackageSim thermal pipeline (which is what we're validating) only
// cares about per-die / per-type energy ratios, which this captures.
//
// The graph dataset is required by the loader framework but its contents
// are unused; any small dataset (Kron16) works as a no-op placeholder.

#include <vector>

// =====================================================================
// Workload tunables — cycle counts model the per-role activity intensity
// =====================================================================
const u_int64_t LLM_ITERS_PER_TILE = 8;   // total iterations per tile
const int LLM_T1_CYCLES  = 8;     // CPU per-iter cost
const int LLM_T2_CYCLES  = 64;    // GPU per-iter cost (counted as FLOPS)
const int LLM_T3_CYCLES  = 24;    // ACCEL per-iter cost
const int LLM_THBM_CYCLES = 2;    // HBM per-iter cost
const int LLM_T3_LOADS   = 4;     // ACCEL HBM-like reads per iter

// =====================================================================
// Globals (populated in config_app from tile_type_array)
// =====================================================================
static std::vector<u_int32_t> cpu_tile_ids;
static std::vector<u_int32_t> gpu_tile_ids;
static std::vector<u_int32_t> accel_tile_ids;
static std::vector<u_int32_t> hbm_tile_ids;

// =====================================================================
// Required interface
// =====================================================================

bool compare_out(char* output_name) {
    (void)output_name;
    return true;  // synthetic — no reference to compare
}

void initialize_dataset_structures() {
    initialize_proxys();
    cout << "proxys initialized\n" << flush;
    ret = (int *) malloc(sizeof(int) * graph->nodes);
    for (int i = 0; i < graph->nodes; i++) ret[i] = 0;
}

void config_dataset(string dataset_filename) {
    dataset_has_edge_val = 0;
    graph = new graph_loader(dataset_filename, 1);
}

void config_app() {
    cpu_tile_ids.clear();
    gpu_tile_ids.clear();
    accel_tile_ids.clear();
    hbm_tile_ids.clear();
    for (u_int32_t i = 0; i < GRID_SIZE; i++) {
        switch (tile_type_array[i]) {
            case TILE_TYPE_CPU:   cpu_tile_ids.push_back(i); break;
            case TILE_TYPE_GPU:   gpu_tile_ids.push_back(i); break;
            case TILE_TYPE_ACCEL: accel_tile_ids.push_back(i); break;
            case TILE_TYPE_HBM:   hbm_tile_ids.push_back(i); break;
        }
    }
    cout << "[llm_inference] role tile counts: cpu=" << cpu_tile_ids.size()
         << " gpu=" << gpu_tile_ids.size()
         << " accel=" << accel_tile_ids.size()
         << " hbm=" << hbm_tile_ids.size() << endl;

    proxy_default = 0;
    task1_dest = 1;
    max_task_chunk = LOOP_CHUNK;

    // Leave queue sizes at UINT32_MAX so config_queue applies its
    // default formulas — they all pass the >=16 assertion that way.

    dataset_words_per_tile = nodePerTile + edgePerTile;
}

int task_init(int tX, int tY) {
    // Every tile bootstraps LLM_ITERS_PER_TILE iterations of self-fed
    // T1 work. The kernel below picks its workload size based on the
    // tile's role.
    for (u_int64_t i = 0; i < LLM_ITERS_PER_TILE; i++) {
        routers[tX][tY]->output_q[C][0].enqueue(Msg(i, MONO, 0));
    }
    return (int)LLM_ITERS_PER_TILE;
}

int task1_kernel(int tX, int tY, u_int64_t timer, u_int64_t & compute_cycles) {
    (void)compute_cycles;
    u_int32_t this_tile = global(tX, tY);
    u_int8_t role = tile_type_array[this_tile];

    Msg msg = IQ(0).dequeue();
    u_int64_t iter = msg.data;

    int penalty;
    switch (role) {
        case TILE_TYPE_CPU:
            // Light dispatch: integer ops, all SRAM-resident.
            penalty = LLM_T1_CYCLES;
            break;
        case TILE_TYPE_GPU:
            // Heavy compute: FLOPS dominate. Some SRAM activity for register
            // spills / activation reads, but the data stays on-die.
            penalty = LLM_T2_CYCLES;
            flop(LLM_T2_CYCLES);
            break;
        case TILE_TYPE_ACCEL: {
            // Memory-bound decode: KV-cache reads exceed any reasonable
            // DCACHE size. We issue LLM_T3_LOADS dcache lookups per iter
            // against the loader-provided edge_array (contents unused;
            // the array just gives us a valid pointer that produces a
            // distinct cache tag). Under DCACHE=1 with dataset_cached=
            // false (the typical config), check_dcache's else branch
            // counts every call as an HBM miss -> mc_transactions++ ->
            // HBM read energy, which calc_energy.h's heterogeneity patch
            // re-attributes to the HBM-tagged die.
            penalty = LLM_T3_CYCLES;
            u_int64_t base_idx = (u_int64_t)this_tile * 4096 + iter * 1024;
            u_int64_t max_idx = (u_int64_t)graph->edges;
            for (int k = 0; k < LLM_T3_LOADS; k++) {
                u_int64_t idx = (base_idx + (u_int64_t)k * 32) % max_idx;
                penalty += check_dcache(tX, tY, graph->edge_array, idx,
                                        timer + penalty);
            }
            break;
        }
        case TILE_TYPE_HBM:
        default:
            // HBM controller die: near-idle PU. Real HBM read energy is
            // accounted system-wide in calc_energy.h (mc_transactions ->
            // hbm_access_energy_pj) and re-attributed to this die in the
            // Per-Die Energy log section via the heterogeneity patch.
            penalty = LLM_THBM_CYCLES;
            break;
    }
    return penalty;
}

int task2_kernel(int tX, int tY, u_int64_t timer, u_int64_t & compute_cycles) {
    (void)tX; (void)tY; (void)timer; (void)compute_cycles;
    return 1;  // unused
}

int task3_kernel(int tX, int tY, u_int64_t timer, u_int64_t & compute_cycles) {
    (void)tX; (void)tY; (void)timer; (void)compute_cycles;
    return 1;  // unused
}

int task3bis_kernel(int tX, int tY, u_int64_t timer, u_int64_t & compute_cycles) {
    (void)tX; (void)tY; (void)timer; (void)compute_cycles;
    return 1;  // unused
}
