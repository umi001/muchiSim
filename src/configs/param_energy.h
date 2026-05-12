
const double transistor_node = 7;  // nanometers
#define get_vdd(freq) (0.06 + 0.13 * freq + 0.06 * transistor_node)

// PU freq and NoC freq are parameters
// Freq design determines the area of the circuit as it is the target frequency for that design.
// The actual frequency can be scaled up or down from the design frequency to save power, as the voltage is reduced.
// Freq should not be larger than the design one as we would be accounting for less area than the actual one.
const double pu_freq = 1.0;
const double pu_freq_design = pu_freq > 1.0 ? pu_freq : 1.0;

const double pu_vdd = get_vdd(pu_freq);
const double noc_freq = 1.0;
const double noc_freq_design = noc_freq;

const double noc_vdd = get_vdd(noc_freq);

// This is SRAM logic, should not be changed unless we have good data
// Current one https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9162985
const double mem_freq_design = 1.0;
const double mem_freq = 1.0;
const double mem_vdd = get_vdd(mem_freq);

const double noc_to_pu_ratio = pu_freq/noc_freq;
const double pu_to_noc_ratio = noc_freq/pu_freq;
const double pu_mem_ratio = pu_freq/mem_freq;

const double reference_vdd = 0.7; // Vdd considered for these enegy values
const double reference_freq = 1.0; // Frequency considered for these enegy values
// Function to compute the energy of a circuit for voltage vdd_ref, given the energy of that circuit for voltage v1
#define calc_energy_at(known_energy, v2) ( ((known_energy) * (v2 * v2)) / (reference_vdd * reference_vdd) )
#define ceil_macro(x) ( (x) - (int)(x) > 0.1 ? (int)(x) + 1 : (int)(x) )

#define pu_area(known_area) (known_area * (1 + 0.5 * (pu_freq_design - reference_freq)) )
#define mem_area(known_area) (known_area * (1 + 0.5 * (mem_freq_design - reference_freq)) )
#define noc_area(known_area) (known_area * (1 + 0.5 * (noc_freq_design - reference_freq)) )


// === Energy of SRAM ===
// Energy at reference voltage and frequency, scaled to particular vdd. Adjusted for SRAM size at util_energy.h
const double sram_read_word_energy  = calc_energy_at(5.8, mem_vdd);
const double sram_write_word_energy = calc_energy_at(9.1, mem_vdd);
const double sram_leakage_nw_per_kib = 625; // nW per KiB

// === Energy of DRAM ===
// Refresh time uses as double, so that the calculation is done in double precision
const double hbm_refresh_time_ns = 32000000;
const double hbm_read_energy_pj_bit = 3.7; // pJ/bit for HBM read data, including IO (fine_grain paper)
const double hbm_activation_energy_pj_bit = 0.011;
const double hbm_precharge_energy_pj_bit = 0.011;

// === Energy of PU operations ===
const double pj_per_intop_ref = calc_energy_at(6,pu_vdd); //pJ per int_op
const double pj_per_flop_ref = calc_energy_at(30,pu_vdd); //pJ per fp_op

// === Per-tile-type energy coefficients (heterogeneity patch) ===
// One entry per value in tile_layout.h::tile_type_e. Default values are
// equal across types so that the heterogeneity infrastructure stays a
// no-op until a heterogeneous app sets distinct values OR an experiment
// explicitly overrides these constants. NUM_TILE_TYPES is defined in
// tile_layout.h; we duplicate the size literal here to avoid a header
// ordering dependency (param_energy.h is included before tile_layout.h).
//
// Indices: 0=CPU, 1=GPU, 2=ACCEL, 3=HBM.
//
// Rationale for the chosen non-default values (used when an app
// explicitly opts in via init_heterogeneous_pu_coefficients()):
//
//   CPU   - simpler in-order-ish pipeline, lower per-op energy than GPU,
//           but more memory accesses per op (cache-miss-heavy work).
//   GPU   - reference value (same as the homogeneous baseline). SIMT
//           scheduler + register-file accesses add overhead per op.
//   ACCEL - specialized dataflow, fewer pipeline stages -> very low
//           per-op energy on the ops it does do; but it does fewer ops
//           per cycle because it relies on long-latency memory streams.
//   HBM   - "tile" representing the HBM-stack-side controller; near-zero
//           compute, mostly idle leakage. We set its compute coefficients
//           low; the HBM dynamic energy is accounted for separately in
//           the existing DRAM section of calc_energy.h.

// Default (no-op): every type uses the reference value.
double pj_per_intop_by_type[4] = {
    pj_per_intop_ref, pj_per_intop_ref, pj_per_intop_ref, pj_per_intop_ref,
};
double pj_per_flop_by_type[4] = {
    pj_per_flop_ref, pj_per_flop_ref, pj_per_flop_ref, pj_per_flop_ref,
};
// Per-tile-type cycles-per-op multiplier (used by Phase C). Default 1.0
// so simulation timing is unchanged until heterogeneity is engaged.
double cycles_per_op_by_type[4] = {1.0, 1.0, 1.0, 1.0};

// Opt-in initializer that an app calls to activate non-uniform PU
// coefficients. Apps that want the homogeneous baseline simply don't
// call this.
inline void init_heterogeneous_pu_coefficients() {
    // CPU: 0.6x per-op energy of GPU, but 2.0x cycles-per-op (slower pipeline)
    pj_per_intop_by_type[0] = 0.6 * pj_per_intop_ref;
    pj_per_flop_by_type[0]  = 0.6 * pj_per_flop_ref;
    cycles_per_op_by_type[0] = 2.0;
    // GPU: reference (matches the homogeneous baseline)
    pj_per_intop_by_type[1] = pj_per_intop_ref;
    pj_per_flop_by_type[1]  = pj_per_flop_ref;
    cycles_per_op_by_type[1] = 1.0;
    // ACCEL: 0.4x per-op energy (specialized dataflow), 0.5x cycles (faster)
    pj_per_intop_by_type[2] = 0.4 * pj_per_intop_ref;
    pj_per_flop_by_type[2]  = 0.4 * pj_per_flop_ref;
    cycles_per_op_by_type[2] = 0.5;
    // HBM: near-zero compute (HBM controller, not a real PU)
    pj_per_intop_by_type[3] = 0.05 * pj_per_intop_ref;
    pj_per_flop_by_type[3]  = 0.05 * pj_per_flop_ref;
    cycles_per_op_by_type[3] = 4.0;  // mostly idle
}


// === Energy of NoC ===
// Max length is 5mm
const double d2d_interposer_hop_energy_pj_per_bit = calc_energy_at(0.500, noc_vdd);
// Max length is 25mm
const double d2d_substrate_hop_energy_pj_per_bit = calc_energy_at(0.550, noc_vdd);
const double in_die_wire_pj_bit_mm = calc_energy_at(0.150, noc_vdd);

// Energy of the router per flit (wire energy accounted separetely) depends on the radix
#if RUCHE_X == 2 || RUCHE_X == 3
    const u_int64_t energy_per_routing_flit = calc_energy_at(6, noc_vdd); // radix 9
#else
    const u_int64_t energy_per_routing_flit = calc_energy_at(3, noc_vdd); // radix 5
#endif

const double inter_board_pJ_bit = calc_energy_at(1.17,noc_vdd); //hipchips slides, adjusted
const double inter_pack_pJ_bit = inter_board_pJ_bit;