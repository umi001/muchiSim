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
// Combined-run mode (Phase 7-C): one MuchiSim run that switches phase
// mid-simulation, producing time-resolved per-die data spanning the
// prefill->decode handoff in real LLM inference.
//
// When LLM_COMBINED_RUN=0 (default), LLM_PHASE selects a single mode
// for the entire run (existing Phase 1-6 behavior, fully backwards
// compatible).
//
// When LLM_COMBINED_RUN=1, the kernel switches per-layer-iteration:
//   passes [0, LLM_PREFILL_PASSES)        run in prefill mode
//   passes [LLM_PREFILL_PASSES, total)    run in decode mode
// The IQ[0] message carries iter = pass_idx * LLM_NUM_LAYERS + layer.
// =====================================================================
#ifndef LLM_COMBINED_RUN
#define LLM_COMBINED_RUN 0
#endif
#ifndef LLM_PREFILL_PASSES
#define LLM_PREFILL_PASSES 1
#endif
#ifndef LLM_DECODE_PASSES
#define LLM_DECODE_PASSES 1
#endif
#if LLM_COMBINED_RUN
#define LLM_TOTAL_PASSES (LLM_PREFILL_PASSES + LLM_DECODE_PASSES)
#else
#define LLM_TOTAL_PASSES 1
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
// =====================================================================
// Role-based work assignment (Phase 9)
// =====================================================================
// Each tile role does a qualitatively different slice of the decoder
// layer, matching how heterogeneous LLM inference systems actually
// partition work:
//   GPU   tiles : bulk matmul = QKV proj + output proj + FFN
//                 (compute-heavy, weight reads from HBM)
//   ACCEL tiles : attention   = QK^T + softmax + V multiply
//                 (memory-heavy in decode -- reads KV cache;
//                  compute-heavy in prefill -- O(S^2) attention)
//   CPU   tiles : layernorm + softmax + residual + control
//                 (very small per layer; included for completeness)
//   HBM   tiles : memory device; no compute, no reads issued
//
// Within each role the work is split evenly across that role's tiles.
// Below we precompute everything in a phase x role lookup table; the
// kernel just does work_table[phase_idx][role].
enum LlmPhaseId { LLM_PHASE_PREFILL = 0, LLM_PHASE_DECODE = 1, LLM_NUM_PHASE_IDS = 2 };
const char * const llm_phase_names[LLM_NUM_PHASE_IDS] = {"prefill", "decode"};

struct LlmRoleWork {
    u_int64_t flops_per_layer_per_tile;
    int       penalty_per_layer;
    u_int64_t weight_reads_per_layer;   // HBM weight cache lines per tile per layer
    u_int64_t kv_reads_per_layer;       // HBM KV cache lines per tile per layer
};

LlmRoleWork work_table[LLM_NUM_PHASE_IDS][NUM_TILE_TYPES] = {};
u_int32_t   tile_count_by_role[NUM_TILE_TYPES] = {0};

// =====================================================================
// Compute-only ring topology (Phase 9.1)
// =====================================================================
// Real tensor-parallel all-reduce runs only across compute peers; HBM
// dies are memory devices and never act as ring participants. This
// table is the next-compute-tile-in-global-id-order for each tile, so
// task1_kernel's ring-send hop is O(1) instead of scanning at runtime.
// Built once in config_app() after tile_type_array is final.
// Value UINT32_MAX means "no non-HBM successor" (degenerate; not
// expected at runtime).
u_int32_t llm_next_compute_tile[GRID_SIZE];

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

    // ----- Phase 9: Role-based work-table builder -----
    // Count tiles per role. Used to split per-role total work evenly
    // across the tiles that belong to that role.
    for (u_int32_t r = 0; r < NUM_TILE_TYPES; r++) tile_count_by_role[r] = 0;
    for (u_int32_t i = 0; i < GRID_SIZE; i++) {
        tile_count_by_role[tile_type_array[i]]++;
    }

    // ----- Phase 9.1: build compute-only ring topology -----
    // For each tile t, find the next non-HBM tile in global-id order
    // (mod GRID_SIZE). HBM tiles are memory devices and skipped.
    for (u_int32_t t = 0; t < GRID_SIZE; t++) {
        u_int32_t next = (t + 1) % GRID_SIZE;
        // Walk forward at most GRID_SIZE-1 steps. If we wrap all the way
        // back to t without finding a non-HBM tile, mark as no successor.
        u_int32_t steps = 0;
        while (steps < GRID_SIZE && tile_type_array[next] == TILE_TYPE_HBM) {
            next = (next + 1) % GRID_SIZE;
            steps++;
        }
        llm_next_compute_tile[t] =
            (steps == GRID_SIZE) ? UINT32_MAX : next;
    }
    {
        u_int32_t n_with_succ = 0;
        for (u_int32_t t = 0; t < GRID_SIZE; t++)
            if (llm_next_compute_tile[t] != UINT32_MAX) n_with_succ++;
        cout << "[llm] ring topology: "
             << n_with_succ << " tiles have a compute successor, "
             << (GRID_SIZE - n_with_succ) << " do not" << endl;
    }
    u_int32_t n_gpu   = tile_count_by_role[TILE_TYPE_GPU];
    u_int32_t n_accel = tile_count_by_role[TILE_TYPE_ACCEL];
    u_int32_t n_cpu   = tile_count_by_role[TILE_TYPE_CPU];
    // Guards: avoid division-by-zero when a role has no tiles (e.g. a
    // homogeneous run with only GPU tiles). The role still gets a slot
    // in work_table but the FLOPs/reads come out as 0 because the
    // role doesn't actually exist in the grid.
    if (n_gpu   == 0) n_gpu   = 1;
    if (n_accel == 0) n_accel = 1;
    if (n_cpu   == 0) n_cpu   = 1;

    const u_int64_t H = LLM_HIDDEN_DIM;
    const u_int64_t F = LLM_FFN_DIM;
    const u_int64_t S = LLM_SEQ_LEN;
    const u_int64_t cache_line_bytes =
        (u_int64_t)dcache_words_in_line * sizeof(u_int32_t);  // 64 B typical

    // Per-role TOTAL work per layer (across the whole grid).
    // GPU does bulk matmul: QKV (3*H^2) + output proj (H^2) + FFN gate/up/down (3*H*F),
    //   each multiplied by T (tokens this pass) and 2 (mul+add). So per layer:
    //     bulk_flops(T) = 2 * T * (4*H^2 + 3*H*F) = 8 T H^2 + 6 T H F
    // ACCEL does attention: QK^T (T S H) + softmax*V (T S H), each x2:
    //     attn_flops(T) = 4 * T * S * H
    // CPU does layernorm + softmax + residual (small, but non-zero):
    //     cpu_flops(T) ~ 4 * T * H  (placeholder; real ops are layernorm gamma/beta
    //                                fused multiply-add + softmax exp/normalize)
    // GPU reads layer weights from HBM: (4*H^2 + 3*H*F) * 4 bytes
    // ACCEL reads KV cache in decode: 2 * S * H * 4 bytes; prefill = 0 (KV is fresh)
    // CPU reads are tiny (layernorm scale/bias); we model as 0 for now.

    auto fill = [&](LlmPhaseId phase, u_int64_t T, bool decode_kv) {
        // Totals across the grid
        u_int64_t gpu_flops_total   = 8ULL * T * H * H + 6ULL * T * H * F;
        u_int64_t accel_flops_total = 4ULL * T * S * H;
        u_int64_t cpu_flops_total   = 4ULL * T * H;
        u_int64_t weight_bytes      = (4ULL * H * H + 3ULL * H * F) * sizeof(float);
        u_int64_t kv_bytes          = decode_kv ? (2ULL * S * H * sizeof(float)) : 0ULL;

        auto set_role = [&](u_int8_t role, u_int64_t flops_total, u_int32_t n_tiles,
                            u_int64_t weight_bytes_role, u_int64_t kv_bytes_role) {
            LlmRoleWork & w = work_table[phase][role];
            u_int64_t f_per_tile = flops_total / (u_int64_t)n_tiles;
            if (f_per_tile == 0) f_per_tile = 1;
            w.flops_per_layer_per_tile = f_per_tile;
            w.penalty_per_layer = (int)((f_per_tile + LLM_FLOPS_PER_CYCLE - 1)
                                        / LLM_FLOPS_PER_CYCLE);
            w.weight_reads_per_layer =
                weight_bytes_role > 0
                    ? (weight_bytes_role / (u_int64_t)n_tiles) / cache_line_bytes
                    : 0;
            w.kv_reads_per_layer =
                kv_bytes_role > 0
                    ? (kv_bytes_role / (u_int64_t)n_tiles) / cache_line_bytes
                    : 0;
        };

        // GPU gets weight reads but no KV reads.
        set_role(TILE_TYPE_GPU,   gpu_flops_total,   n_gpu,   weight_bytes, 0);
        // ACCEL gets KV reads (decode only) but no weight reads.
        set_role(TILE_TYPE_ACCEL, accel_flops_total, n_accel, 0,            kv_bytes);
        // CPU gets neither (its bytes are negligible at this granularity).
        set_role(TILE_TYPE_CPU,   cpu_flops_total,   n_cpu,   0,            0);
        // HBM tiles do nothing at all (they're memory devices, not processors).
        // Leave HBM entry at zeros — task1_kernel branches on role and skips work.
    };

    fill(LLM_PHASE_PREFILL, /*T=*/S, /*decode_kv=*/false);
    fill(LLM_PHASE_DECODE,  /*T=*/1, /*decode_kv=*/true);

    cout << "[llm] ===== Role-based work table =====" << endl;
    cout << "[llm]   tile counts: gpu=" << tile_count_by_role[TILE_TYPE_GPU]
         << " cpu="   << tile_count_by_role[TILE_TYPE_CPU]
         << " accel=" << tile_count_by_role[TILE_TYPE_ACCEL]
         << " hbm="   << tile_count_by_role[TILE_TYPE_HBM] << endl;
    for (u_int32_t p = 0; p < LLM_NUM_PHASE_IDS; p++) {
        cout << "[llm]   phase=" << llm_phase_names[p] << " role-by-role:" << endl;
        for (u_int32_t r = 0; r < NUM_TILE_TYPES; r++) {
            const LlmRoleWork & w = work_table[p][r];
            cout << "[llm]     " << tile_type_names[r]
                 << ": flops/tile=" << w.flops_per_layer_per_tile
                 << " penalty="     << w.penalty_per_layer
                 << " W-rd="        << w.weight_reads_per_layer
                 << " KV-rd="       << w.kv_reads_per_layer << endl;
        }
    }
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
    int iter = msg.data;
    int penalty = 0;

    int my_tile = global(tX, tY);
    u_int8_t role = tile_type_array[my_tile];

    // ----- Phase 9: pick (phase, role) -> per-tile work -----
    // In single-phase mode, every layer-iter uses the same phase from
    // LLM_PHASE. In combined-run mode, iter encodes both pass_idx and
    // layer; passes [0, LLM_PREFILL_PASSES) are prefill, rest decode.
    int phase_idx;
#if LLM_COMBINED_RUN
    int pass_idx = iter / LLM_NUM_LAYERS;
    phase_idx = (pass_idx < LLM_PREFILL_PASSES)
              ? LLM_PHASE_PREFILL : LLM_PHASE_DECODE;
#else
    phase_idx = (LLM_PHASE == 0) ? LLM_PHASE_PREFILL : LLM_PHASE_DECODE;
#endif
    const LlmRoleWork & w = work_table[phase_idx][role];

    // ----- Weight reads (Phase 4): only roles with weight_reads > 0 -----
    // In Phase 9 only GPU dies issue weight reads (they own bulk matmul).
    // Each miss increments mc_transactions[], which calc_energy.h
    // re-attributes entirely to HBM-tagged die(s).
    if (w.weight_reads_per_layer > 0) {
        const u_int64_t w_max_idx =
            (u_int64_t)GRID_SIZE * (u_int64_t)llm_weights_words_per_tile;
        // Coprime stride/offsets so reads from different tiles / layers
        // map to different cache lines.
        u_int64_t w_base =
            ((u_int64_t)my_tile * 7919ULL + (u_int64_t)iter * 257ULL) % w_max_idx;
        for (u_int64_t k = 0; k < w.weight_reads_per_layer; k++) {
            u_int64_t idx = (w_base + k * (u_int64_t)dcache_words_in_line) % w_max_idx;
            penalty += check_dcache(tX, tY, llm_weights, idx, timer + penalty);
        }
    }

    // ----- KV-cache reads (Phase 5, decode-only ACCEL): -----
    // In Phase 9 only ACCEL dies do attention, and only in decode mode
    // does attention read the cached K and V from HBM. Prefill computes
    // K/V fresh in registers.
    if (w.kv_reads_per_layer > 0) {
        const u_int64_t kv_max_idx =
            (u_int64_t)GRID_SIZE * (u_int64_t)llm_kv_words_per_tile;
        u_int64_t kv_base =
            ((u_int64_t)my_tile * 6553ULL + (u_int64_t)iter * 313ULL) % kv_max_idx;
        for (u_int64_t k = 0; k < w.kv_reads_per_layer; k++) {
            u_int64_t idx = (kv_base + k * (u_int64_t)dcache_words_in_line) % kv_max_idx;
            penalty += check_dcache(tX, tY, llm_kv_cache, idx, timer + penalty);
        }
    }

    // ----- Matmul (Phase 2/9): role-specific FLOPs/cycles -----
    // HBM tiles are memory devices: no compute, no reads, no ring
    // participation, no layer re-enqueue. They process their bootstrap
    // IQ[0] message and then go idle for the rest of the simulation.
    if (role == TILE_TYPE_HBM) {
        return penalty + 1;  // tiny advance so core_timer is non-zero
    }
    flop((u_int32_t)w.flops_per_layer_per_tile);
    penalty += w.penalty_per_layer;

    // ----- Ring all-reduce send (Phase 9.1: compute-only) -----
    // Only compute (non-HBM) tiles participate in the tensor-parallel
    // ring. The next ring-neighbor is the next non-HBM tile in global-id
    // order, precomputed in llm_next_compute_tile[] during config_app.
    // Inter-die hops happen wherever this lookup crosses a die boundary,
    // including the case where the path goes around the HBM die.
    u_int32_t next_tile_u = llm_next_compute_tile[my_tile];
    if (next_tile_u != UINT32_MAX) {
        int next_tX, next_tY;
        global_to_xy((int)next_tile_u, next_tX, next_tY);
        u_int32_t head_flit = XYHeadFlit(next_tX, next_tY);
        OQ(2).enqueue(Msg(head_flit, HEAD, timer + penalty));
        OQ(2).enqueue(Msg(iter,      TAIL, timer + penalty));
        store(2);
        penalty += 4;
    }

    const int max_iter = LLM_TOTAL_PASSES * LLM_NUM_LAYERS;
    if (iter + 1 < max_iter) {
        IQ(0).enqueue(Msg(iter + 1, MONO, timer + penalty));
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
