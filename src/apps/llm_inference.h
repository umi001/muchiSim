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

    // Phase 1 does no cross-tile communication. Most queues stay small,
    // but oq_sizes[dest_qid] (i.e. oq_sizes[2] when PROXY_FACTOR==1) has
    // an assertion in config_queue() requiring at least 16 entries.
    iq_sizes[1] = unused_buffer;
    oq_sizes[1] = unused_buffer;
    iq_sizes[2] = unused_buffer;
    oq_sizes[2] = 16;
    iq_sizes[3] = unused_buffer;
    oq_sizes[3] = unused_buffer;

    // Approximate per-tile SRAM footprint (in 32-bit words). This is small
    // for Phase 1 — Phase 4 will adjust it once HBM weight access is modeled.
    dataset_words_per_tile = llm_acts_words_per_tile;
}

int task_init(int tX, int tY) {
    // Each tile bootstraps with one message in its local IQ[0], carrying
    // layer index 0. The kernel re-enqueues each subsequent layer until
    // LLM_NUM_LAYERS have been processed.
    routers[tX][tY]->output_q[C][0].enqueue(Msg(0, MONO, 0));
    return 1;
}

int task1_kernel(int tX, int tY, u_int64_t timer, u_int64_t & compute_cycles) {
    (void)compute_cycles;

    Msg msg = IQ(0).dequeue();
    int layer = msg.data;

    // Phase 1: trivial per-layer penalty, no real computation.
    // Later phases inject real matmul ops, HBM reads, and all-reduce here.
    int penalty = 10;

    if (layer + 1 < LLM_NUM_LAYERS) {
        IQ(0).enqueue(Msg(layer + 1, MONO, timer + penalty));
    }

    return penalty;
}

// Stub kernels — populated in later phases.
int task2_kernel(int tX, int tY, u_int64_t timer, u_int64_t & compute_cycles) {
    (void)tX; (void)tY; (void)timer; (void)compute_cycles;
    return 1;
}
int task3_kernel(int tX, int tY, u_int64_t timer, u_int64_t & compute_cycles) {
    (void)tX; (void)tY; (void)timer; (void)compute_cycles;
    return 1;
}
int task3bis_kernel(int tX, int tY, u_int64_t timer, u_int64_t & compute_cycles) {
    (void)tX; (void)tY; (void)timer; (void)compute_cycles;
    return 1;
}
