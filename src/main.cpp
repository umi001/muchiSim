#include "common/macros.h"
#if USE_OMP
  #include <omp.h>
#else
  #include <pthread.h>
#endif
// System Configuration Parameters
#include "configs/config_system.h"
#include "configs/config_queue.h"
// Heterogeneity infrastructure: per-tile and per-die role tags. Defines
// globals (tile_type_array, die_role), so include exactly here in the
// single translation unit. Defaults to homogeneous (TILE_TYPE_GPU
// everywhere) unless an app explicitly calls init_tile_types_per_chiplet.
#include "configs/tile_layout.h"
// Global structures and defined parameters
#include "common/global.h"
#include "mem/memory_util.h"
#if APP<ALTERNATIVE
  #include "dataset_loaders/graph_loader.h"
#else
  #include "dataset_loaders/no_loader.h"
#endif
graph_loader * graph = NULL;
#include "network/router.h"
#include "common/calc_area.h"
#include "common/calc_cost.h"
#include "common/util_stats.h"
#include "common/calc_stats.h"
#include "mem/memory_system.h"
#include "apps/frontier_common.h"
#include "configs/config_app.h"
#include "network/network.h"


void *thread_function(void *arg) {
    long tid = (long)arg;
    // cout << tid << flush;
    if (tid < COLUMNS) {
        router_thread(tid);
    } else {
        tsu_core_thread(tid - COLUMNS);
    }
    return NULL;
}

int main(int argc, char** argv) {
  //print_processor_info();
  calculate_derived_param();
  // ==== Check Configurations Allowed ====
  assert(pow(2,log2(GRID_X))==GRID_X );
  assert(GRID_X>=BOARD_W);
  assert(BOARD_W>=DIE_W);
  assert(MUX_BUS<=BOARD_W);
  #if BOARD_W < GRID_X
  assert(COLUMNS_PER_TH>=MUX_BUS);
  #endif

  // ==== ALLOCATE Memory for the DATASET =====
  dataset_filename = "datasets/Kron16/";
  binary_filename = "0";
  bool dry_run = false;
  if (argc >= 2) dataset_filename = argv[1];
  if (argc >= 3) binary_filename = argv[2];
  if (argc >= 4) dry_run = (bool) atoi(argv[3]);
  cout << "Dataset: " << dataset_filename << endl;
  cout << "Dry run: " << dry_run << endl;
  // ==== CONFIGURATIONS ====
  config_dataset(dataset_filename);
  // Initialize the per-tile role layout BEFORE config_app() so the app's
  // hook can read tile_type_array (e.g., llm_inference partitions tiles
  // into cpu/gpu/accel/hbm role lists during config_app).
  init_tile_types_homogeneous();
  if (const char* layout_env = std::getenv("MUCHI_HETERO_LAYOUT")) {
    u_int8_t roles[DIES];
    for (u_int32_t d = 0; d < DIES; d++) roles[d] = TILE_TYPE_GPU;
    std::string s(layout_env);
    u_int32_t d = 0;
    size_t pos = 0;
    while (pos < s.size() && d < DIES) {
      size_t next = s.find(',', pos);
      std::string token = s.substr(pos, next == std::string::npos ? std::string::npos : next - pos);
      if      (token == "cpu")   roles[d] = TILE_TYPE_CPU;
      else if (token == "gpu")   roles[d] = TILE_TYPE_GPU;
      else if (token == "accel") roles[d] = TILE_TYPE_ACCEL;
      else if (token == "hbm")   roles[d] = TILE_TYPE_HBM;
      else { std::cerr << "Unknown role in MUCHI_HETERO_LAYOUT: " << token << "\n"; return 1; }
      d++;
      if (next == std::string::npos) break;
      pos = next + 1;
    }
    if (d != DIES) {
      std::cerr << "MUCHI_HETERO_LAYOUT had " << d << " roles but DIES=" << DIES << "\n";
      return 1;
    }
    init_tile_types_per_chiplet(roles);
    init_heterogeneous_pu_coefficients();
    std::cout << "Heterogeneous layout engaged via MUCHI_HETERO_LAYOUT: " << layout_env << std::endl;
  }
  config_app();
  config_queue();

  // ==== CALCULATE STORAGE, AREA and COST ====
  // NOTE: Don't change the order of these functions as there are values that are used in the next ones
  calculate_storage_per_tile();
  print_configuration(cout);
  area_calculation();
  cost_calculation();
  if (dry_run) return 0;

  // (tile_type_array and per-type coefficients were initialized earlier
  // — before config_app() — so the app could read tile_type_array during
  // its own setup.)

  init_perf_counters(); cout << "Perf counters initialized\n"<<flush;
  connect_mesh(); cout << "Mesh connected\n"<<flush;

  // ==== ALLOCATE Sync, Cache and Dataset Structures =====
  intialize_sync_structures();
  initialize_cache_structures(); cout << "Cache structures initialized\n"<<flush;
  initialize_dataset_structures(); cout << "Dataset structures initialized\n"<<flush;

  // ==== START THE SIMULATION ====
  cout << std::setprecision(2) << std::fixed << "\n\nStarting Simulation\n" << std::flush;
  auto start = chrono::system_clock::now();
  #if USE_OMP
    // int max_threads = omp_get_max_threads(); cout << "Max Available threads: " << max_threads << "\n"; assert(MAX_THREADS <= max_threads);
    omp_set_num_threads(MAX_THREADS);
    #pragma omp parallel default(shared)
    {
      long tid = omp_get_thread_num();
      thread_function((void*)tid);
    }
  #else
    pthread_t threads[MAX_THREADS]; int rc; long t;
    for(t = 0; t < MAX_THREADS; t++) {
        rc = pthread_create(&threads[t], NULL, thread_function, (void *)t);
        if (rc) {std::cout << "Error: unable to create thread," << rc << "\n"; exit(-1);}
    }
    for(t = 0; t < MAX_THREADS; t++) pthread_join(threads[t], NULL);
  #endif

  auto end = std::chrono::system_clock::now();
  chrono::duration<double> elapsed_seconds = end-start;
  double sim_time = elapsed_seconds.count();

  // ==== PRINT THE PERFORMANCE COUNTERS and RESULTS ====
  print_stats_acum(true, sim_time);
  bool result_correct = (argc >= 5) ? result_correct = compare_out(argv[4]) : true;
  cout << "\nVersion 32\n\n";

  // ==== FREE MEMORY ALLOCATED ====
  destroy_cache_structures();
  destroy_dataset_structures();
  destroy_sync_structures();

  return result_correct ? 0 : 1;
}