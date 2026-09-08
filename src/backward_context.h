#pragma once

#include <cstddef>
#include <memory>
#include <cuda_runtime.h>

#include "linsolve.h"

// Backward-propagation context for the SPLR solver.
//
// The backward pass of entropic-regularized OT requires solving linear
// systems with the reduced Hessian H = (1/reg) * Q(T*) by implicit
// differentiation. As a Stage-3B.1 correctness baseline, this context stores
// a private CSR snapshot of the selected approximate Hessian A and creates
// its OWN cuDSS factorization of that snapshot:
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
// Important ownership note:
//
//   This version does NOT move or rebind the forward solver's cuDSS
//   factorization. The forward only provides a CSR snapshot and low-rank
//   materials; the context performs a fresh analyze + factorization on its
//   own snapshot. This avoids relying on replacing cudssMatrix_t after a
//   numerical factorization. True zero-extra-factorization ownership
//   transfer is left for a later stage.
//
// Lifetime invariant:
//
//   1. m_solver is destroyed first, which destroys all cudssMatrix_t
//      wrappers, cudssData_t, cudssConfig_t and cudssHandle_t;
//   2. only after that are the CSR/rhs/solution/low-rank GPU buffers freed.
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

    // Snapshot a CSR triple produced by the forward pass. In the normal path
    // this is called immediately after each compute_search_direc(), so the
    // snapshot corresponds to the matrix consumed by that iteration's
    // forward factorization, even though the forward working buffers are
    // overwritten later by dual_sphess().
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

    // ---------------- Backward-side interface ----------------

    // Compute z = B^{-1} r. d_r and d_z are device pointers of length
    // problem_size. Synchronous with respect to the host.
    void apply_preconditioner(const double* d_r, double* d_z);

    // Metadata
    bool   is_valid()       const { return m_valid; }
    bool   is_fallback()    const { return m_fallback; }
    bool   low_rank()       const { return m_low_rank; }
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
    bool   m_fallback = false; // true if factorize_fallback() was used
    bool   m_valid = false;    // true after the owned snapshot is factorized

    // The context owns its own solver and factorization of the snapshot.
    // A unique_ptr is used so that destruction order is explicit:
    // solver first, backing buffers second.
    std::unique_ptr<SparseCholeskySolver> m_solver;

    // CSR backing buffers (private copies of the selected matrix)
    double* d_Avalues = nullptr;
    int*    d_Acolind = nullptr;
    int*    d_Arowptr = nullptr;

    // Fixed rhs/solution buffers, persistently bound to m_solver
    double* d_rhs = nullptr;
    double* d_sol = nullptr;

    // Low-rank materials
    double* d_s = nullptr;
    double* d_y = nullptr;
    double* d_invA_y = nullptr;

    // Validation: the point associated with the snapshot
    double* d_ctx_point = nullptr;

    // Pinned host workspace for the reduction kernels
    double* h_pinned = nullptr;
};
