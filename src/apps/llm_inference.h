// Phase 1 of proper LLM inference modeling in MuchiSim, modeled after
// src/apps/fft.h — the cleanest example of an app that owns its own data
// and doesn't pretend to be a graph workload.
//
// Goal of Phase 1: validate the app integrates cleanly with main.cpp and
// the TSU loop, and runs to completion under both homo and hetero die
// layouts. No matmul, no all-reduce, no HBM access yet — just the
// structural skeleton.
//
// Subsequent phases (per the design plan):
//   Phase 2: matmul via flop()/penalty for a single layer
//   Phase 3: ring all-reduce via T3 messages (real inter-die traffic)
//   Phase 4: HBM weight fetch via check_dcache (real DRAM access)
//   Phase 5: KV-cache reads for decode + prefill/decode phase switch
//   Phase 6: multi-layer loop and final per-die energy validation
//
// Model defaults match LLaMA-7B; override any of them at compile time
// with -DLLM_HIDDEN_DIM=... etc.

// =====================================================================
// Architecture parameters (LLaMA-7B defaults)
// =====================================================================
#ifndef LLM_HIDDEN_DIM
#define LLM_HIDDEN_DIM    4096
#endif
#ifndef LLM_FFN_DIM
#define LLM_FFN_DIM       11008
#endif
#ifndef LLM_NUM_HEADS
#define LLM_NUM_HEADS     32
#endif
#ifndef LLM_NUM_LAYERS
#define LLM_NUM_LAYERS    32
#endif
#ifndef LLM_SEQ_LEN
#define LLM_SEQ_LEN       2048
#endif
#ifndef LLM_PHASE
#define LLM_PHASE         1   // 0 = PREFILL, 1 = DECODE
#endif
#define LLM_HEAD_DIM      (LLM_HIDDEN_DIM / LLM_NUM_HEADS)

// How many FP ops a single tile retires per cycle. Heuristic that converts
// per-tile FLOP counts into a cycle penalty. Defaults to 16 (a vector-tile
// throughput); raise for tensor-core-like tiles, lower for scalar tiles.
#ifndef LLM_FLOPS_PER_CYCLE
#define LLM_FLOPS_PER_CYCLE 16
#endif

// =====================================================================
// Per-tile data slabs
// =====================================================================
// In later phases:
//   llm_weights  — walked by check_dcache during weight fetch (Phase 4)
//   llm_kv_cache — read for decode (per-token), written for prefill (Phase 5)
//   llm_acts     — current hidden state, sender buffer for all-reduce (Phase 3)
// Phase 1 just proves the allocation chain works — the values are unused.
float * llm_weights = NULL;
float * llm_kv_cache = NULL;
float * llm_acts = NULL;
const u_int32_t llm_weights_words_per_tile = 1024;
const u_int32_t llm_kv_words_per_tile      = 1024;
const u_int32_t llm_acts_words_per_tile    = 256;

// =====================================================================
// Per-layer GEMM model (computed in config_app, consumed by task1_kernel)
// =====================================================================
// Decoder layer flop counts (one forward pass through one layer):
//   QKV projections        : 3 × 2 × T × H × H        (T tokens this pass)
//   Attention QK^T         : 2 × T × H × S            (S = effective KV length)
//   Attention V multiply   : 2 × T × S × H
//   Attention output proj  : 2 × T × H × H
//   FFN gate / up / down   : 3 × 2 × T × H × F
// Prefill: T = SEQ_LEN, S = SEQ_LEN (attention is O(S²))
// Decode : T = 1,       S = SEQ_LEN (one new token vs full cache)
u_int64_t llm_flops_per_layer_total = 0;       // across whole grid
u_int64_t llm_flops_per_layer_per_tile = 0;    // per-tile slice
int       llm_penalty_per_layer = 0;            // derived cycle count

// =====================================================================
// Per-layer HBM weight-fetch model (Phase 4)
// =====================================================================
// Each compute tile reads its slice of the layer's weights from HBM via
// check_dcache. The number of reads is derived from the actual weight
// bytes:
//   QKV projections:    3 × H × H × 4B
//   Output projection:    H × H × 4B
//   FFN gate / up / down: 3 × H × F × 4B
// Total per layer: (4H² + 3HF) × sizeof(float) bytes,
// split across (GRID_SIZE - HBM_tile_count) compute tiles.
// We divide by cache-line bytes to get reads-per-tile-per-layer.
u_int64_t llm_hbm_reads_per_layer = 0;

// =====================================================================
// Required app interface
// =====================================================================

bool compare_out(char* output_name) {
    (void)output_name;
    return true;  // no functional reference to compare in Phase 1
}

void config_dataset(string dataset_filename) {
    (void)dataset_filename;
    // Empty, like FFT. No graph loaded; the dataset path passed on the
    // command line is consumed by main.cpp and ignored here.
}

void initialize_dataset_structures() {
    u_int64_t w_total = (u_int64_t)GRID_SIZE * llm_weights_words_per_tile;
    u_int64_t k_total = (u_int64_t)GRID_SIZE * llm_kv_words_per_tile;
    u_int64_t a_total = (u_int64_t)GRID_SIZE * llm_acts_words_per_tile;
    llm_weights  = (float *) calloc(w_total, sizeof(float));
    llm_kv_cache = (float *) calloc(k_total, sizeof(float));
    llm_acts     = (float *) calloc(a_total, sizeof(float));

    cout << "[llm] Phase 1 skeleton initialized\n";
    cout << "[llm]   model: hidden=" << LLM_HIDDEN_DIM
         << " ffn="      << LLM_FFN_DIM
         << " heads="    << LLM_NUM_HEADS
         << " head_dim=" << LLM_HEAD_DIM
         << " layers="   << LLM_NUM_LAYERS
         << " seq_len="  << LLM_SEQ_LEN
         << " phase="    << (LLM_PHASE == 0 ? "PREFILL" : "DECODE")
         << endl;
    cout << "[llm]   per-tile buffers: weights=" << llm_weights_words_per_tile
         << "w, kv=" << llm_kv_words_per_tile
         << "w, acts=" << llm_acts_words_per_tile << "w" << endl;
    cout << "[llm]   total host alloc: "
         << ((w_total + k_total + a_total) * sizeof(float) / 1024 / 1024)
         << " MiB" << endl;
}

void config_app() {
    ALWAYS_ASSERT_MSG(PROXY_FACTOR == 1,
        "PROXY must be disabled for LLM (set PROXY_W = GRID_X)");

    proxy_default = 0;
    // task1_dest selects which OQ's capacity is consulted in the TSU's
    // runnability check (must hold > 8 entries). Even though Phase 1 never
    // writes to any OQ, the check still runs — point it at OQ[2] which is
    // sized to 16 below, matching FFT's pattern.
    task1_dest = 2;
    max_task_chunk = LOOP_CHUNK;

    // Phase 3 uses channel 2 (T3) for the ring all-reduce step after each
    // layer's matmul. Channel 1 (T2) and 3 (T3') stay unused. Sizing of
    // oq_sizes[2] / iq_sizes[2] follows FFT's pattern (oq=16, iq=128).
    iq_sizes[1] = unused_buffer;
    oq_sizes[1] = unused_buffer;
    iq_sizes[2] = 128;
    oq_sizes[2] = 16;
    iq_sizes[3] = unused_buffer;
    oq_sizes[3] = unused_buffer;

    // Approximate per-tile SRAM footprint (in 32-bit words). This is small
    // for Phase 1 — Phase 4 will adjust it once HBM weight access is modeled.
    dataset_words_per_tile = llm_acts_words_per_tile;

    // ----- Phase 2: per-layer GEMM cost -----
    const u_int64_t H = LLM_HIDDEN_DIM;
    const u_int64_t F = LLM_FFN_DIM;
    const u_int64_t S = LLM_SEQ_LEN;
#if LLM_PHASE == 0
    // Prefill: T = S, attention is O(S²)
    const u_int64_t T = S;
    const u_int64_t attn_flops = 4ULL * T * S * H;
#else
    // Decode: T = 1, attention is one row against the cached S keys/values
    const u_int64_t T = 1ULL;
    const u_int64_t attn_flops = 4ULL * T * S * H;
#endif
    const u_int64_t proj_flops = 8ULL * T * H * H;       // QKV + output proj
    const u_int64_t ffn_flops  = 6ULL * T * H * F;       // gate + up + down
    llm_flops_per_layer_total    = proj_flops + attn_flops + ffn_flops;
    llm_flops_per_layer_per_tile = llm_flops_per_layer_total / (u_int64_t)GRID_SIZE;
    if (llm_flops_per_layer_per_tile == 0) llm_flops_per_layer_per_tile = 1;
    llm_penalty_per_layer = (int)((llm_flops_per_layer_per_tile + LLM_FLOPS_PER_CYCLE - 1)
                                  / LLM_FLOPS_PER_CYCLE);

    cout << "[llm]   per-layer FLOPs total : " << llm_flops_per_layer_total << endl;
    cout << "[llm]   per-layer FLOPs/tile  : " << llm_flops_per_layer_per_tile << endl;
    cout << "[llm]   per-layer cycles/tile : " << llm_penalty_per_layer
         << "  (at " << LLM_FLOPS_PER_CYCLE << " FLOPs/cycle)" << endl;
    cout << "[llm]   est. total cycles/tile: "
         << ((u_int64_t)llm_penalty_per_layer * LLM_NUM_LAYERS) << endl;

    // ----- Phase 4: HBM weight-fetch rate -----
    // Count how many tiles will act as compute (everything not HBM). With
    // the homogeneous default all tiles are GPU; with a 4-die hetero
    // layout, three of four dies are compute (3*256 = 768 tiles).
    u_int32_t n_compute_tiles = 0;
    for (u_int32_t i = 0; i < GRID_SIZE; i++) {
        if (tile_type_array[i] != TILE_TYPE_HBM) n_compute_tiles++;
    }
    if (n_compute_tiles == 0) n_compute_tiles = 1;  // guard

    const u_int64_t weight_bytes_per_layer_total =
        (4ULL * H * H + 3ULL * H * F) * sizeof(float);
    const u_int64_t weight_bytes_per_tile =
        weight_bytes_per_layer_total / (u_int64_t)n_compute_tiles;
    const u_int64_t cache_line_bytes =
        (u_int64_t)dcache_words_in_line * sizeof(u_int32_t);  // typically 64 B
    llm_hbm_reads_per_layer = weight_bytes_per_tile / cache_line_bytes;
    if (llm_hbm_reads_per_layer == 0) llm_hbm_reads_per_layer = 1;

    cout << "[llm]   weight bytes/layer   : " << weight_bytes_per_layer_total << endl;
    cout << "[llm]   weight bytes/tile/lyr: " << weight_bytes_per_tile
         << "  (" << n_compute_tiles << " compute tiles)" << endl;
    cout << "[llm]   HBM reads/tile/layer : " << llm_hbm_reads_per_layer
         << "  (cache line = " << cache_line_bytes << " B)" << endl;
}

int task_init(int tX, int tY) {
    // Each tile bootstraps with one message in its local IQ[0], carrying
    // layer index 0. The kernel re-enqueues each subsequent layer until
    // LLM_NUM_LAYERS have been processed.
    routers[tX][tY]->output_q[C][0].enqueue(Msg(0, MONO, 0));
    return 1;
}

// Helper: convert global tile id back to (x, y) for the current topology.
// Inverse of the `global(x,y)` macro in common/macros.h.
inline void global_to_xy(int g, int & tX, int & tY) {
#if TORUS==1
    tX = g & GRID_Xm1;
    tY = g >> GRID_X_LOG2;
#else
    tY = g >> GRID_X_LOG2;
    tX = g & GRID_Xm1;
    if ((tY % 2) == 1) tX = GRID_Xm1 - tX;
#endif
}

int task1_kernel(int tX, int tY, u_int64_t timer, u_int64_t & compute_cycles) {
    (void)compute_cycles;

    Msg msg = IQ(0).dequeue();
    int layer = msg.data;
    int penalty = 0;

    int my_tile = global(tX, tY);
    u_int8_t role = tile_type_array[my_tile];

    // ----- HBM weight fetch (Phase 4) -----
    // Compute tiles (everything except HBM-tagged tiles) read their slice
    // of the layer weights from HBM via cache-missing accesses. Each miss
    // increments mc_transactions[], which calc_energy.h re-attributes
    // entirely to HBM-tagged die(s).
    if (role != TILE_TYPE_HBM) {
        const u_int64_t max_idx =
            (u_int64_t)GRID_SIZE * (u_int64_t)llm_weights_words_per_tile;
        // Use coprime offsets (primes) per tile and per layer so reads
        // from different tiles / different layers map to different cache
        // lines. The stride = cache-line size guarantees each access
        // lands in a new tag.
        u_int64_t base_idx =
            ((u_int64_t)my_tile * 7919ULL + (u_int64_t)layer * 257ULL) % max_idx;
        for (u_int64_t k = 0; k < llm_hbm_reads_per_layer; k++) {
            u_int64_t idx = (base_idx + k * (u_int64_t)dcache_words_in_line) % max_idx;
            penalty += check_dcache(tX, tY, llm_weights, idx, timer + penalty);
        }

        // ----- Matmul (Phase 2) -----
        flop((u_int32_t)llm_flops_per_layer_per_tile);
        penalty += llm_penalty_per_layer;
    } else {
        // HBM tiles act as memory devices: they do no compute and no
        // weight fetch. Their participation in the ring (below) keeps
        // the all-reduce topology consistent. A small idle penalty
        // approximates the controller overhead of forwarding.
        penalty += 4;
    }

    // ----- Ring all-reduce send (Phase 3) -----
    // Every tile participates in the ring, including HBM-tagged tiles
    // (they forward partial sums without adding to them in this model).
    int next_tile = (my_tile + 1) % GRID_SIZE;
    int next_tX, next_tY;
    global_to_xy(next_tile, next_tX, next_tY);
    u_int32_t head_flit = XYHeadFlit(next_tX, next_tY);
    OQ(2).enqueue(Msg(head_flit, HEAD, timer + penalty));
    OQ(2).enqueue(Msg(layer,     TAIL, timer + penalty));
    store(2);
    penalty += 4;

    if (layer + 1 < LLM_NUM_LAYERS) {
        IQ(0).enqueue(Msg(layer + 1, MONO, timer + penalty));
    }

    return penalty;
}

int task2_kernel(int tX, int tY, u_int64_t timer, u_int64_t & compute_cycles) {
    (void)tX; (void)tY; (void)timer; (void)compute_cycles;
    return 1;
}

// Phase 3: receive one ring all-reduce partial-sum (2-flit T3 message).
// Phase 3 is purely structural — we count traffic, no real accumulation.
int task3_kernel(int tX, int tY, u_int64_t timer, u_int64_t & compute_cycles) {
    (void)tX; (void)tY; (void)timer; (void)compute_cycles;
    IQ(2).dequeue();  // HEAD
    IQ(2).dequeue();  // TAIL
    load(1);
    return 3;
}

int task3bis_kernel(int tX, int tY, u_int64_t timer, u_int64_t & compute_cycles) {
    (void)tX; (void)tY; (void)timer; (void)compute_cycles;
    return 1;
}
