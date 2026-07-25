// Agentic AI workload (APP=10) for MuchiSim.
//
// This is workload #3 of three independent workloads (the others are LLM
// inference, APP=9, and graph processing). It is a STANDALONE app: it does
// not reuse the graph machinery and is not a composition of the other two.
// It does reuse the *code patterns and FLOP formulas* of llm_inference.h,
// because the compute core of an agent is still transformer inference.
//
// What makes it distinct from one-shot LLM inference:
//   An agent runs an OUTER CONTROL LOOP with tool use. Each turn it
//   (1) reasons over its accumulated context (prefill + decode), (2) calls
//   a tool, (3) observes the result and appends it to the context. The
//   active IP -- and therefore the hotspot -- MIGRATES across the loop, and
//   the context GROWS every turn, so load escalates. This is a single-agent
//   ReAct loop (multi-agent planner+workers is a documented v2; the
//   work_table[turn][stage][role] structure generalizes to it).
//
// Sizing (see tools/agentbench_sizing.py + docs/what_is_needed.md):
//   Group 1 (model arch)  : LLaMA-7B defaults, reused from the LLM app.
//   Group 2 (loop behavior): AGENT_NUM_TURNS / AGENT_BASE_SEQ_LEN /
//                            AGENT_TOOL_CPU_RATIO are grounded in AgentBench
//                            (config round caps + tool schemas + tokenized
//                            task data) and set per environment via -D.
//   Group 3 (hw cost)     : AGENT_TOOL_CPU_FLOPS / AGENT_TOOL_STALL_CYCLES /
//                            AGENT_OBSERVE_FLOPS / AGENT_GEN_TOKENS /
//                            AGENT_OBS_TOKENS are swept (no static source).
//
// Override any macro at compile time with -DAGENT_...=value.

// =====================================================================
// Group 1 -- model architecture (LLaMA-7B defaults; same basis as APP=9)
// =====================================================================
#ifndef AGENT_HIDDEN_DIM
#define AGENT_HIDDEN_DIM    4096
#endif
#ifndef AGENT_FFN_DIM
#define AGENT_FFN_DIM       11008
#endif
#ifndef AGENT_NUM_HEADS
#define AGENT_NUM_HEADS     32
#endif
// Layers a single reason stage walks. Kept smaller than a real 32-layer
// model so the per-turn iteration count (and one MuchiSim run) stays
// bounded while each stage still spans >= 1 ACUM sample window.
#ifndef AGENT_NUM_LAYERS
#define AGENT_NUM_LAYERS    8
#endif
#ifndef AGENT_FLOPS_PER_CYCLE
#define AGENT_FLOPS_PER_CYCLE 16
#endif
#define AGENT_HEAD_DIM (AGENT_HIDDEN_DIM / AGENT_NUM_HEADS)

// =====================================================================
// Group 2 -- agent-loop behavior (grounded in AgentBench, set per env)
// =====================================================================
// Defaults below match the AgentBench "dbbench" environment
// (max_round=15, execute_sql -> local compute tool, ~775-token base
// context). Override per environment, e.g.:
//   os      : -DAGENT_NUM_TURNS=8  -DAGENT_BASE_SEQ_LEN=128 -DAGENT_TOOL_CPU_RATIO=100
//   webshop : -DAGENT_NUM_TURNS=20 -DAGENT_BASE_SEQ_LEN=512 -DAGENT_TOOL_CPU_RATIO=0
#ifndef AGENT_NUM_TURNS
#define AGENT_NUM_TURNS     15
#endif
#ifndef AGENT_BASE_SEQ_LEN
#define AGENT_BASE_SEQ_LEN  768
#endif
// Percent of turns whose tool call is a LOCAL compute burst (vs an external
// latency stall). 100 = all CPU bursts (os/dbbench), 0 = all stalls (webshop).
#ifndef AGENT_TOOL_CPU_RATIO
#define AGENT_TOOL_CPU_RATIO 100
#endif

// =====================================================================
// Group 3 -- hardware cost of non-LLM stages (swept; see what_is_needed.md)
// =====================================================================
// Per-turn generation + observation token counts. They drive context growth
// (S_turn) only; not derivable from a static AgentBench checkout -> sweep.
#ifndef AGENT_GEN_TOKENS
#define AGENT_GEN_TOKENS    256
#endif
#ifndef AGENT_OBS_TOKENS
#define AGENT_OBS_TOKENS    128
#endif
// Compute a local (CPU) tool invocation performs -- shell / SQL / code exec.
#ifndef AGENT_TOOL_CPU_FLOPS
#define AGENT_TOOL_CPU_FLOPS (4ULL * AGENT_HIDDEN_DIM * 256ULL)
#endif
// External tool latency (web / API round-trip) in cycles. Whole package
// idles for the window, so this must be >= sample_time to show a distinct
// cool frame. The stall-to-compute ratio is a headline thermal-recovery knob.
#ifndef AGENT_TOOL_STALL_CYCLES
#define AGENT_TOOL_STALL_CYCLES 200000ULL
#endif
// Cost of parsing the tool output and appending it to the context (CPU).
#ifndef AGENT_OBSERVE_FLOPS
#define AGENT_OBSERVE_FLOPS (64ULL * AGENT_HIDDEN_DIM)
#endif
// Per-HBM-miss latency allowance used to size the per-stage lockstep barrier
// (see below). A safe upper bound keeps the compute-heavy role padding rather
// than overshooting the barrier, so stages stay time-synchronized.
#ifndef AGENT_MEM_LATENCY_EST
#define AGENT_MEM_LATENCY_EST 500
#endif

// =====================================================================
// Per-tile data slabs (HBM-resident; walked by check_dcache).
// ag_weights / ag_kv_cache are also forward-declared in mem/data_cache.h
// so its specialized cache_tag can identify them.
// =====================================================================
float * ag_weights  = NULL;
float * ag_kv_cache = NULL;
float * ag_acts     = NULL;
const u_int32_t ag_weights_words_per_tile = 1024;
const u_int32_t ag_kv_words_per_tile      = 1024;
const u_int32_t ag_acts_words_per_tile    = 256;

// =====================================================================
// Stage model -- a single agent turn cycles through these stages.
// =====================================================================
//   AG_REASON_PREFILL : ingest full (growing) context. GPU matmul-heavy,
//                       ACCEL attention O(S^2). Weight reads from HBM.
//   AG_REASON_DECODE  : generate a thought + tool call. KV reads from HBM
//                       (ACCEL + HBM activity), GPU per-token matmul.
//   AG_TOOL_EXEC_CPU  : local tool (shell/SQL/code). CPU burst; GPU/ACCEL idle.
//   AG_TOOL_EXEC_STALL: external tool (web/API). Whole complex idles.
//   AG_OBSERVE        : append/parse tool output. Small CPU.
// HBM tiles are memory devices in every stage (no compute).
enum AgenticStage {
    AG_REASON_PREFILL = 0,
    AG_REASON_DECODE  = 1,
    AG_TOOL_EXEC_CPU  = 2,
    AG_TOOL_EXEC_STALL= 3,
    AG_OBSERVE        = 4,
    AG_NUM_STAGES     = 5
};
const char * const ag_stage_names[AG_NUM_STAGES] =
    {"reason_prefill", "reason_decode", "tool_cpu", "tool_stall", "observe"};

struct AgRoleWork {
    u_int64_t flops_per_call_per_tile;
    int       penalty_per_call;       // compute cycles = flops / FLOPS_PER_CYCLE
    u_int64_t weight_reads_per_call;  // HBM weight cache lines per tile
    u_int64_t kv_reads_per_call;      // HBM KV cache lines per tile
};

// Per-turn x per-stage x per-role work, plus the per-stage lockstep barrier
// duration. Built once in config_app(); growing context makes later turns
// heavier. Sizes are compile-time constants (tiny).
AgRoleWork ag_work_table[AGENT_NUM_TURNS][AG_NUM_STAGES][NUM_TILE_TYPES] = {};
u_int64_t  ag_stage_cycles[AGENT_NUM_TURNS][AG_NUM_STAGES] = {};
u_int32_t  ag_tile_count_by_role[NUM_TILE_TYPES] = {0};

// Compute-only ring (skips HBM dies), as in llm_inference.h.
u_int32_t  ag_next_compute_tile[GRID_SIZE];

// Flat schedule: one entry per kernel micro-iteration. Per turn:
//   REASON_PREFILL x L, REASON_DECODE x L, one TOOL stage, one OBSERVE.
#define AG_ITERS_PER_TURN (2 * AGENT_NUM_LAYERS + 2)
#define AG_MAX_ITER       (AGENT_NUM_TURNS * AG_ITERS_PER_TURN)
u_int8_t   ag_stage_of_iter[AG_MAX_ITER];
u_int32_t  ag_turn_of_iter[AG_MAX_ITER];

// Decide whether turn `t`'s tool call is a local CPU burst or an external
// stall, per the AgentBench-derived tool-type ratio.
static inline bool ag_tool_is_cpu(int t) {
#if AGENT_TOOL_CPU_RATIO >= 100
    (void)t; return true;
#elif AGENT_TOOL_CPU_RATIO <= 0
    (void)t; return false;
#else
    return ((t * 37 + 11) % 100) < AGENT_TOOL_CPU_RATIO;
#endif
}

// =====================================================================
// Required app interface
// =====================================================================
bool compare_out(char* output_name) {
    (void)output_name;
    return true;  // no functional reference
}

void config_dataset(string dataset_filename) {
    (void)dataset_filename;  // synthetic / parameterized; no graph loaded
}

void initialize_dataset_structures() {
    u_int64_t w_total = (u_int64_t)GRID_SIZE * ag_weights_words_per_tile;
    u_int64_t k_total = (u_int64_t)GRID_SIZE * ag_kv_words_per_tile;
    u_int64_t a_total = (u_int64_t)GRID_SIZE * ag_acts_words_per_tile;
    ag_weights  = (float *) calloc(w_total, sizeof(float));
    ag_kv_cache = (float *) calloc(k_total, sizeof(float));
    ag_acts     = (float *) calloc(a_total, sizeof(float));

    cout << "[agentic] single-agent ReAct loop initialized\n";
    cout << "[agentic]   model: hidden=" << AGENT_HIDDEN_DIM
         << " ffn="    << AGENT_FFN_DIM
         << " heads="  << AGENT_NUM_HEADS
         << " layers=" << AGENT_NUM_LAYERS << endl;
    cout << "[agentic]   loop: turns="    << AGENT_NUM_TURNS
         << " base_seq=" << AGENT_BASE_SEQ_LEN
         << " gen="      << AGENT_GEN_TOKENS
         << " obs="      << AGENT_OBS_TOKENS
         << " tool_cpu_ratio=" << AGENT_TOOL_CPU_RATIO << "%" << endl;
}

void config_app() {
    ALWAYS_ASSERT_MSG(PROXY_FACTOR == 1,
        "PROXY must be disabled for AGENTIC (set PROXY_W = GRID_X)");

    proxy_default  = 0;
    task1_dest     = 2;            // OQ[2] capacity is consulted by the TSU
    max_task_chunk = LOOP_CHUNK;

    iq_sizes[1] = unused_buffer;  oq_sizes[1] = unused_buffer;
    iq_sizes[2] = 128;            oq_sizes[2] = 16;   // ring all-reduce channel
    iq_sizes[3] = unused_buffer;  oq_sizes[3] = unused_buffer;

    dataset_words_per_tile = ag_acts_words_per_tile;

    // ----- tiles per role -----
    for (u_int32_t r = 0; r < NUM_TILE_TYPES; r++) ag_tile_count_by_role[r] = 0;
    for (u_int32_t i = 0; i < GRID_SIZE; i++)
        ag_tile_count_by_role[tile_type_array[i]]++;

    // ----- compute-only ring (skip HBM dies), identical to the LLM app -----
    for (u_int32_t t = 0; t < GRID_SIZE; t++) {
        u_int32_t next = (t + 1) % GRID_SIZE, steps = 0;
        while (steps < GRID_SIZE && tile_type_array[next] == TILE_TYPE_HBM) {
            next = (next + 1) % GRID_SIZE; steps++;
        }
        ag_next_compute_tile[t] = (steps == GRID_SIZE) ? UINT32_MAX : next;
    }

    u_int32_t n_gpu   = ag_tile_count_by_role[TILE_TYPE_GPU];
    u_int32_t n_accel = ag_tile_count_by_role[TILE_TYPE_ACCEL];
    u_int32_t n_cpu   = ag_tile_count_by_role[TILE_TYPE_CPU];
    if (n_gpu   == 0) n_gpu   = 1;
    if (n_accel == 0) n_accel = 1;
    if (n_cpu   == 0) n_cpu   = 1;

    const u_int64_t H = AGENT_HIDDEN_DIM;
    const u_int64_t F = AGENT_FFN_DIM;
    const u_int64_t cache_line_bytes =
        (u_int64_t)dcache_words_in_line * sizeof(u_int32_t);

    // ----- build the per-turn work table (growing context) -----
    for (int turn = 0; turn < AGENT_NUM_TURNS; turn++) {
        // Accumulated trajectory length entering this turn.
        const u_int64_t S = (u_int64_t)AGENT_BASE_SEQ_LEN
            + (u_int64_t)turn * ((u_int64_t)AGENT_GEN_TOKENS + (u_int64_t)AGENT_OBS_TOKENS);

        // Fill one (turn,stage): set per-role work and the lockstep barrier.
        auto fill_stage = [&](int stage,
                              u_int64_t gpu_total, u_int64_t accel_total,
                              u_int64_t cpu_total, u_int64_t weight_bytes,
                              u_int64_t kv_bytes) {
            u_int64_t max_pen = 0, max_reads = 0;
            auto set_role = [&](u_int8_t role, u_int64_t flops_total,
                                u_int32_t n_tiles, u_int64_t wbytes, u_int64_t kvbytes) {
                AgRoleWork & w = ag_work_table[turn][stage][role];
                u_int64_t f = flops_total / (u_int64_t)n_tiles;
                w.flops_per_call_per_tile = f;
                u_int64_t pen64 = (f + AGENT_FLOPS_PER_CYCLE - 1) / AGENT_FLOPS_PER_CYCLE;
                // Kernel penalties are int (count_delay takes int). A per-tile
                // penalty above INT_MAX means too few tiles for the model size
                // -- use a bigger grid (more tiles/role), smaller dims, or a
                // larger AGENT_FLOPS_PER_CYCLE. Fail loud rather than overflow.
                ALWAYS_ASSERT_MSG(pen64 < 2000000000ULL,
                    "agentic per-tile penalty overflows int: increase grid/tiles "
                    "per role, reduce AGENT_HIDDEN_DIM/FFN/turns, or raise "
                    "AGENT_FLOPS_PER_CYCLE");
                w.penalty_per_call = (int)pen64;
                w.weight_reads_per_call =
                    wbytes  > 0 ? (wbytes  / (u_int64_t)n_tiles) / cache_line_bytes : 0;
                w.kv_reads_per_call =
                    kvbytes > 0 ? (kvbytes / (u_int64_t)n_tiles) / cache_line_bytes : 0;
                if ((u_int64_t)w.penalty_per_call > max_pen)
                    max_pen = (u_int64_t)w.penalty_per_call;
                u_int64_t rds = w.weight_reads_per_call + w.kv_reads_per_call;
                if (rds > max_reads) max_reads = rds;
            };
            set_role(TILE_TYPE_GPU,   gpu_total,   n_gpu,   weight_bytes, 0);
            set_role(TILE_TYPE_ACCEL, accel_total, n_accel, 0,            kv_bytes);
            set_role(TILE_TYPE_CPU,   cpu_total,   n_cpu,   0,            0);
            // HBM stays zero (memory device).
            // Barrier = bottleneck compute + a safe read-latency allowance so
            // the reading role pads (stays in lockstep) rather than overshoots.
            u_int64_t sc = max_pen + max_reads * (u_int64_t)AGENT_MEM_LATENCY_EST;
            sc = sc ? sc : 1;
            // The kernel snaps penalty to stage_cycles via (int)stage_cycles,
            // so the barrier must also fit int.
            ALWAYS_ASSERT_MSG(sc < 2000000000ULL,
                "agentic stage_cycles overflows int: increase grid/tiles per "
                "role or reduce model dims / read counts");
            ag_stage_cycles[turn][stage] = sc;
        };

        // Reason: prefill ingests the whole context (T = S); decode is one
        // representative token (T = 1) reading the KV cache (grows with S).
        const u_int64_t weight_bytes = (4ULL * H * H + 3ULL * H * F) * sizeof(float);
        fill_stage(AG_REASON_PREFILL,
                   8ULL * S * H * H + 6ULL * S * H * F,   // GPU bulk matmul
                   4ULL * S * S * H,                      // ACCEL attention O(S^2)
                   4ULL * S * H,                          // CPU norm/softmax/residual
                   weight_bytes, /*kv=*/0);
        fill_stage(AG_REASON_DECODE,
                   8ULL * H * H + 6ULL * H * F,           // GPU per-token matmul
                   4ULL * S * H,                          // ACCEL attention over cache
                   4ULL * H,                              // CPU
                   weight_bytes,
                   2ULL * S * H * sizeof(float));         // KV cache read (K and V)

        // Tool stages: only one is scheduled per turn (ag_tool_is_cpu), but
        // both are filled. CPU burst lights the CPU die; stall idles everyone.
        fill_stage(AG_TOOL_EXEC_CPU,  0, 0, (u_int64_t)AGENT_TOOL_CPU_FLOPS, 0, 0);
        fill_stage(AG_TOOL_EXEC_STALL, 0, 0, 0, 0, 0);
        ag_stage_cycles[turn][AG_TOOL_EXEC_STALL] = (u_int64_t)AGENT_TOOL_STALL_CYCLES;

        fill_stage(AG_OBSERVE, 0, 0, (u_int64_t)AGENT_OBSERVE_FLOPS, 0, 0);
    }

    // ----- build the flat stage schedule -----
    int it = 0;
    for (int turn = 0; turn < AGENT_NUM_TURNS; turn++) {
        for (int l = 0; l < AGENT_NUM_LAYERS; l++) {
            ag_stage_of_iter[it] = AG_REASON_PREFILL; ag_turn_of_iter[it] = turn; it++;
        }
        for (int l = 0; l < AGENT_NUM_LAYERS; l++) {
            ag_stage_of_iter[it] = AG_REASON_DECODE;  ag_turn_of_iter[it] = turn; it++;
        }
        ag_stage_of_iter[it] = ag_tool_is_cpu(turn) ? AG_TOOL_EXEC_CPU
                                                    : AG_TOOL_EXEC_STALL;
        ag_turn_of_iter[it] = turn; it++;
        ag_stage_of_iter[it] = AG_OBSERVE;            ag_turn_of_iter[it] = turn; it++;
    }
    ALWAYS_ASSERT_MSG(it == AG_MAX_ITER, "agentic schedule length mismatch");

    // ----- diagnostics -----
    cout << "[agentic] ===== schedule =====" << endl;
    cout << "[agentic]   turns="   << AGENT_NUM_TURNS
         << " iters/turn="         << AG_ITERS_PER_TURN
         << " total iters="        << AG_MAX_ITER << endl;
    cout << "[agentic]   tile counts: gpu=" << ag_tile_count_by_role[TILE_TYPE_GPU]
         << " cpu="   << ag_tile_count_by_role[TILE_TYPE_CPU]
         << " accel=" << ag_tile_count_by_role[TILE_TYPE_ACCEL]
         << " hbm="   << ag_tile_count_by_role[TILE_TYPE_HBM] << endl;
    for (int turn = 0; turn < AGENT_NUM_TURNS; turn++) {
        const u_int64_t S = (u_int64_t)AGENT_BASE_SEQ_LEN
            + (u_int64_t)turn * ((u_int64_t)AGENT_GEN_TOKENS + (u_int64_t)AGENT_OBS_TOKENS);
        cout << "[agentic]   turn " << turn << " S=" << S
             << " tool=" << (ag_tool_is_cpu(turn) ? "cpu" : "stall")
             << " stage_cycles[prefill/decode/tool/observe]="
             << ag_stage_cycles[turn][AG_REASON_PREFILL] << "/"
             << ag_stage_cycles[turn][AG_REASON_DECODE]  << "/"
             << (ag_tool_is_cpu(turn) ? ag_stage_cycles[turn][AG_TOOL_EXEC_CPU]
                                      : ag_stage_cycles[turn][AG_TOOL_EXEC_STALL]) << "/"
             << ag_stage_cycles[turn][AG_OBSERVE] << endl;
    }
}

int task_init(int tX, int tY) {
    // Each tile bootstraps with iter 0 in its local IQ[0]; the kernel walks
    // the schedule by re-enqueuing iter+1 until AG_MAX_ITER.
    routers[tX][tY]->output_q[C][0].enqueue(Msg(0, MONO, 0));
    return 1;
}

// Inverse of the global(x,y) macro, identical to the LLM app.
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

    int my_tile   = global(tX, tY);
    u_int8_t role = tile_type_array[my_tile];

    int stage = ag_stage_of_iter[iter];
    int turn  = (int)ag_turn_of_iter[iter];
    const AgRoleWork & w = ag_work_table[turn][stage][role];
    const u_int64_t stage_cycles = ag_stage_cycles[turn][stage];

    // HBM tiles are memory devices: process the bootstrap message once and go
    // idle. Their power comes from being the target of others' reads (HBM
    // transactions), not from their own activity. They do not re-enqueue.
    if (role == TILE_TYPE_HBM) {
        return penalty + 1;
    }

    // ----- external tool stall: nobody computes; the whole complex idles -----
    // Expressed as mem-wait so the TDP adapter (active = Task - Mem_wait)
    // reads ~0 active on every die for the window.
    if (stage == AG_TOOL_EXEC_STALL) {
        mem_wait_add(stage_cycles);
        penalty += (int)stage_cycles;
        if (iter + 1 < AG_MAX_ITER)
            IQ(0).enqueue(Msg(iter + 1, MONO, timer + penalty));
        return penalty;
    }

    // ----- HBM reads (weights for GPU; KV cache for ACCEL in decode) -----
    if (w.weight_reads_per_call > 0) {
        const u_int64_t w_max_idx =
            (u_int64_t)GRID_SIZE * (u_int64_t)ag_weights_words_per_tile;
        u_int64_t w_base =
            ((u_int64_t)my_tile * 7919ULL + (u_int64_t)iter * 257ULL) % w_max_idx;
        for (u_int64_t k = 0; k < w.weight_reads_per_call; k++) {
            u_int64_t idx = (w_base + k * (u_int64_t)dcache_words_in_line) % w_max_idx;
            penalty += check_dcache(tX, tY, ag_weights, idx, timer + penalty);
        }
    }
    if (w.kv_reads_per_call > 0) {
        const u_int64_t kv_max_idx =
            (u_int64_t)GRID_SIZE * (u_int64_t)ag_kv_words_per_tile;
        u_int64_t kv_base =
            ((u_int64_t)my_tile * 6553ULL + (u_int64_t)iter * 313ULL) % kv_max_idx;
        for (u_int64_t k = 0; k < w.kv_reads_per_call; k++) {
            u_int64_t idx = (kv_base + k * (u_int64_t)dcache_words_in_line) % kv_max_idx;
            penalty += check_dcache(tX, tY, ag_kv_cache, idx, timer + penalty);
        }
    }

    // ----- compute (role-specific FLOPs) -----
    if (w.flops_per_call_per_tile > 0) {
        flop(w.flops_per_call_per_tile);
        penalty += w.penalty_per_call;
    }

    // ----- ring all-reduce send (reason stages only; compute peers) -----
    if (stage == AG_REASON_PREFILL || stage == AG_REASON_DECODE) {
        u_int32_t next_tile_u = ag_next_compute_tile[my_tile];
        if (next_tile_u != UINT32_MAX) {
            int next_tX, next_tY;
            global_to_xy((int)next_tile_u, next_tX, next_tY);
            u_int32_t head_flit = XYHeadFlit(next_tX, next_tY);
            OQ(2).enqueue(Msg(head_flit, HEAD, timer + penalty));
            OQ(2).enqueue(Msg(iter,      TAIL, timer + penalty));
            store(2);
            penalty += 4;
        }
    }

    // ----- lockstep barrier: every non-HBM tile takes exactly stage_cycles -----
    // The active role's compute fills the window; idle roles pad the remainder
    // into mem-wait (modeling a stall while they wait for the active role).
    // Net: per-die active = compute, and stages stay time-synchronized so the
    // hotspot migration is visible across ACUM windows.
    if ((u_int64_t)penalty < stage_cycles) {
        mem_wait_add(stage_cycles - (u_int64_t)penalty);
        penalty = (int)stage_cycles;
    }

    if (iter + 1 < AG_MAX_ITER) {
        IQ(0).enqueue(Msg(iter + 1, MONO, timer + penalty));
    }
    return penalty;
}

int task2_kernel(int tX, int tY, u_int64_t timer, u_int64_t & compute_cycles) {
    (void)tX; (void)tY; (void)timer; (void)compute_cycles;
    return 1;
}

// Receive one ring all-reduce partial-sum (2-flit T3 message), like the LLM app.
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
