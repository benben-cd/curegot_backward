#pragma once

#include <cstddef>
#include <memory>
#include <cuda_runtime.h>

#include "linsolve.h"

// Debug/benchmark-only timing breakdown of launch_low_rank_search_direc()
// (Stage 3C-2A). Defined here so that the kernel implementation
// (sinkhorn_splr_kernel_low_rank.cu), the forward solver declaration
// (sinkhorn_splr.cu) and SPLRBackwardContext share a single definition.
//
// Real implementation structure (confirmed against the kernel source):
//   1. cudaHostGetDevicePointer + search_direc_dot_kernel launch (async);
//   2. cudaStreamSynchronize(stream)  <-- blocking: the dot kernel is queued
//      after the cuDSS solve on the same stream, so this wait absorbs any
//      still-running solve work plus the dot kernel;
//   3. host partial-sum reduction + scalar arithmetic (sg_ys, term1, term2);
//   4. update_direc_kernel launch (async).
struct LowRankSearchDirecProfile
{
    // Host wall times (std::chrono::steady_clock, microseconds)
    double setup_launch_host_us = 0.0;      // step 1
    double sync_wait_host_us = 0.0;         // step 2; absorbs the GPU solve
                                            // wait when the solve is still
                                            // running -- NOT pure sync overhead
    double host_scalar_us = 0.0;            // step 3
    double correction_launch_host_us = 0.0; // step 4
    // GPU times (CUDA events on the working stream, microseconds), queried
    // by the caller after synchronizing the stream
    double gpu_reduce_us = 0.0;             // search_direc_dot_kernel
    double gpu_correction_us = 0.0;         // update_direc_kernel
    // Events recorded by the helper. They are created, queried and destroyed
    // by the CALLER (the context profiling path); all nullptr when the
    // helper runs unprofiled (production).
    cudaEvent_t ev_reduce_start = nullptr;
    cudaEvent_t ev_reduce_stop = nullptr;
    cudaEvent_t ev_corr_start = nullptr;
    cudaEvent_t ev_corr_stop = nullptr;
};

// Backward-propagation context for the SPLR solver.
//
// The backward pass of entropic-regularized OT requires solving linear
// systems with the reduced Hessian H = (1/reg) * Q(T*) by implicit
// differentiation. Depending on the ownership mode, the context either reuses
// the forward cuDSS factorization of the selected approximate Hessian A or
// factorizes a private CSR snapshot:
//
//     B = A + (reg/ys) * y * y^T - (A s)(A s)^T / (s^T A s)   [low_rank enabled]
//     B = A                                                    [low_rank disabled]
//
// where A = reg * H_Omega + (shift * reg) * I is the sparsified Hessian in
// the stored scale, s and y are the last secant pair, and ys = y^T s. The
// rank-2 solve uses the Woodbury formula
//
//     B^{-1} r = q - (s^T r / ys) * u
//                + [(1/reg + y^T u / ys) * (s^T r / ys) - y^T q / ys] * s
//
// with q = A^{-1} r and u = A^{-1} y. This is the forward search-direction
// update rule, minus the final -reg scaling.
//
// Ownership modes:
//
//   1. Snapshot mode (fallback/debug baseline): the forward provides a CSR
//      snapshot and low-rank materials; the context performs a fresh
//      analyze + factorization on its private buffers.
//   2. Adopted-forward-state mode (Stage-3B.2-C2 normal path): the context
//      takes ownership of the forward SparseCholeskySolver and the exact
//      backing buffers referenced by its cuDSS wrappers, then reuses the
//      existing forward factorization with zero extra factorization.
//
// Lifetime invariant:
//
//   1. m_solver is destroyed first, which destroys all cudssMatrix_t
//      wrappers, cudssData_t, cudssConfig_t and cudssHandle_t;
//   2. only after that are the CSR/rhs/solution/low-rank GPU buffers freed.
//
// In adopted mode, d_y and d_invA_y are non-owning aliases into the second
// columns of d_rhs and d_sol, respectively. Ownership flags below make this
// distinction explicit and prevent double-free in the destructor.
class SPLRBackwardContext
{
public:
    // problem_size = n + m - 1 (dimension of the reduced dual variable)
    // max_nnz      = Kmax + problem_size (capacity of the CSR buffers)
    SPLRBackwardContext(int problem_size, int max_nnz);
    ~SPLRBackwardContext();

    // Non-copyable: the context owns CUDA buffers and an opaque cuDSS state.
    SPLRBackwardContext(const SPLRBackwardContext&) = delete;
    SPLRBackwardContext& operator=(const SPLRBackwardContext&) = delete;

    // Movable: needed when returning the context through pybind. Ownership
    // of the solver and all buffers is transferred; the source becomes empty.
    SPLRBackwardContext(SPLRBackwardContext&& other) noexcept;
    SPLRBackwardContext& operator=(SPLRBackwardContext&& other) noexcept;

    // ---------------- Forward-side interface ----------------
    // The functions below are called by cuda_sinkhorn_splr_bwd() only.

    // Snapshot a CSR triple produced by the forward pass. This snapshot mode
    // is retained for immediate-convergence fallback and debug/ablation
    // baselines. The normal Stage-3B.2-C2 path adopts the forward state
    // instead of snapshotting it.
    void snapshot_sphess(
        const double* d_values, const int* d_colind, const int* d_rowptr,
        size_t nnz
    );

    // Initialize the context from the stored CSR snapshot. This performs a
    // fresh analyze + factorization inside the context. The low-rank
    // materials are copied from the forward pass but the forward solver and
    // its cuDSS factorization remain owned by the forward solver.
    void initialize_from_snapshot(
        const double* d_s_fwd, const double* d_y_fwd,
        const double* d_invA_y_fwd,
        double ys, bool low_rank, double reg,
        const double* d_ctx_pt
    );

    // Fallback path for immediate convergence: the snapshot was taken at the
    // final point and is factorized here without low-rank terms.
    void factorize_fallback(double reg, const double* d_ctx_pt);

    // Adopt a forward-owned solver and its exact backing buffers. This is the
    // Stage-3B.2-C2 normal-path ownership API; it reuses the existing forward
    // cuDSS factorization and performs no new analyze/factorization.
    //
    // Ownership transferred to this context:
    //   - solver (including its cuDSS factorization state);
    //   - d_values / d_colind / d_rowptr;
    //   - d_rhs_fwd / d_sol_fwd (each allocation has two columns);
    //   - d_s_fwd.
    //
    // In low-rank mode, d_y and d_invA_y are aliases:
    //   d_y      = d_rhs_fwd + problem_size
    //   d_invA_y = d_sol_fwd + problem_size
    // and are therefore not freed separately.
    void adopt_forward_state(
        std::unique_ptr<SparseCholeskySolver> solver,
        double* d_values, int* d_colind, int* d_rowptr,
        int dim, size_t nnz,
        double* d_rhs_fwd, double* d_sol_fwd,
        double* d_s_fwd,
        double ys, bool low_rank, double reg, bool fallback,
        const double* d_ctx_pt
    );

    // ---------------- Backward-side interface ----------------

    // Compute z = B^{-1} r. d_r and d_z are device pointers of length
    // problem_size. Synchronous with respect to the host.
    void apply_preconditioner(const double* d_r, double* d_z);

    // -------- Debug/benchmark-only profiling interface (Stage 3C-2A) ------
    // The methods below instrument the SAME execution path as
    // apply_preconditioner(); the production method above is unchanged and
    // gains no extra synchronization or events.

    // Timing breakdown of a single apply_preconditioner call.
    // Host times: std::chrono::steady_clock wall time of each blocking step.
    // GPU times: CUDA events recorded on the solver's working stream.
    struct PreconditionerProfile
    {
        // Host wall time per step (microseconds)
        double host_memcpy_in_us = 0.0;       // cudaMemcpy d_rhs <- d_r
        double host_solve_launch_us = 0.0;    // cuDSS solve API call (async)
        double host_lowrank_total_us = 0.0;   // whole launch_low_rank_search_direc call
        double host_final_sync_wait_us = 0.0; // final cudaStreamSynchronize
        double host_memcpy_out_us = 0.0;      // cudaMemcpy d_z <- d_sol
        double host_total_us = 0.0;           // whole C++ function
        // GPU elapsed time on the solver stream (microseconds, CUDA events)
        double gpu_cudss_solve_us = 0.0;
        // Low-rank stage split; all zeros when low-rank is disabled.
        // Attribution note: in low-rank mode the GPU solve is normally waited
        // on INSIDE the helper's internal cudaStreamSynchronize (reported as
        // lowrank.sync_wait_host_us), so host_final_sync_wait_us then only
        // covers the correction kernel. The true GPU solve time is
        // gpu_cudss_solve_us regardless of where the host wait happens.
        bool low_rank_active = false;
        LowRankSearchDirecProfile lowrank;
    };

    // Debug/benchmark-only: identical math and execution order to
    // apply_preconditioner(), plus chrono/CUDA-event instrumentation. Unlike
    // the production path, this variant creates and destroys CUDA events
    // inside the call, so its host_total is slightly inflated relative to
    // production; the GPU event times are the primary output.
    void apply_preconditioner_profiled(
        const double* d_r, double* d_z, PreconditionerProfile& profile
    );

    // Debug/benchmark-only: call the production apply_preconditioner()
    // `repeats` times in a C++ loop on the same r/z buffers and return the
    // total host wall time in microseconds. Compared against a Python loop
    // over the public API, this isolates the Python/PyBind/tensor-wrapper
    // per-call overhead. Note that the production per-call
    // cudaStreamSynchronize and memcpys remain inside the loop; the ratio
    // does not capture those.
    double benchmark_preconditioner_repeated(
        const double* d_r, double* d_z, int repeats
    );

    // Metadata
    bool   is_valid()       const { return m_valid; }
    bool   is_fallback()    const { return m_fallback; }
    bool   low_rank()       const { return m_low_rank; }
    // true when the context adopted the forward solver's existing cuDSS
    // factorization; false for snapshot/refactorization fallback paths.
    bool   adopted_forward_state() const { return m_adopted_forward_state; }
    int    problem_size()   const { return m_Hsize; }
    double reg()            const { return m_reg; }
    double ys()             const { return m_ys; }

    // Validation-only accessors (device pointers owned by this context)
    const double* csr_values()  const { return d_Avalues; }
    const int*    csr_colind()  const { return d_Acolind; }
    const int*    csr_rowptr()  const { return d_Arowptr; }
    size_t        csr_nnz()     const { return m_nnz; }
    const double* secant_s()    const { return d_s; }
    const double* secant_y()    const { return d_y; }
    const double* ctx_point()   const { return d_ctx_point; }
    bool          has_ctx_point() const { return d_ctx_point != nullptr; }

private:
    // Free GPU buffers that can be replaced by adopted forward state. This
    // preserves d_ctx_point and h_pinned, which remain context-owned.
    void release_replaceable_buffers();

    // Free GPU/pinned buffers only. This function must not touch cuDSS
    // handles; the solver is reset separately before these buffers are freed.
    void release_buffers();

    // Create the solver if needed and factorize the owned CSR snapshot.
    void factorize_snapshot();

    int    m_Hsize = 0;        // n + m - 1
    int    m_max_nnz = 0;      // capacity of d_Avalues / d_Acolind
    size_t m_nnz = 0;          // actual nnz of the snapshot
    double m_reg = 0.0;        // regularization parameter (eta)
    double m_ys = 0.0;         // y^T s of the last secant pair
    bool   m_low_rank = false; // whether the rank-2 correction is active
    bool   m_fallback = false; // true if the fallback path was used
    bool   m_valid = false;    // true after the context has usable state
    bool   m_adopted_forward_state = false; // true after adopt_forward_state()

    // Allocation ownership flags. In snapshot mode all replaceable buffers
    // are context-owned. In adopted mode, CSR/rhs/sol/s are transferred and
    // owned, while y and invA_y are aliases into rhs/sol.
    bool m_owns_csr = false;
    bool m_owns_rhs = false;
    bool m_owns_sol = false;
    bool m_owns_s = false;
    bool m_owns_y = false;
    bool m_owns_invA_y = false;

    // Solver and active factorization state. A unique_ptr is used so that
    // destruction order is explicit: solver first, backing buffers second.
    std::unique_ptr<SparseCholeskySolver> m_solver;

    // CSR backing buffers (context-owned snapshot or adopted forward buffers)
    double* d_Avalues = nullptr;
    int*    d_Acolind = nullptr;
    int*    d_Arowptr = nullptr;

    // Fixed rhs/solution buffers, persistently bound to m_solver
    double* d_rhs = nullptr;
    double* d_sol = nullptr;

    // Low-rank materials. In adopted mode, d_y aliases d_rhs + m_Hsize and
    // d_invA_y aliases d_sol + m_Hsize; the ownership flags reflect this.
    double* d_s = nullptr;
    double* d_y = nullptr;
    double* d_invA_y = nullptr;

    // Validation: the point associated with the snapshot
    double* d_ctx_point = nullptr;

    // Pinned host workspace for the reduction kernels
    double* h_pinned = nullptr;
};
