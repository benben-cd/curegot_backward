#include "backward_context.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Must match the value used in sinkhorn_splr_kernel_low_rank.cu
#define MAX_NUM_BLOCK_1D 256

// Helper function to compute the search direction with low-rank terms.
// Defined in sinkhorn_splr_kernel_low_rank.cu. Given d_direc = A^{-1} g on
// input, it applies the rank-2 Woodbury correction in place:
//   sg = s^T g, yinvAy = y^T invA_y, yinvAg = y^T (A^{-1} g)
//   sg_ys = sg / ys
//   direc += ((1/reg + yinvAy/ys) * sg_ys - yinvAg/ys) * s - sg_ys * invA_y
// The trailing profile argument is debug/benchmark-only (Stage 3C-2A):
// when non-null, the helper records chrono host timings and CUDA events
// (created/owned by the caller) around its internal stages. Production
// call sites pass nothing and run exactly as before.
void launch_low_rank_search_direc(
    double* d_direc,
    const double* d_invA_y,
    const double* d_g,
    const double* d_y,
    const double* d_s,
    double ys,
    double reg,
    int size,
    double* h_pinned,
    cudaStream_t stream,
    LowRankSearchDirecProfile* profile = nullptr
);

// Stage 3C-3 diagonal preconditioner kernels. Defined in
// sinkhorn_splr_kernel_low_rank.cu (keeps this file free of <<< >>> so it
// stays host-compiler syntax-checkable); no build-system changes needed.

// One thread per row scans rowptr[row]..rowptr[row+1] for col == row and
// writes inv_diag[row] = 1 / value. On any violation (missing diagonal,
// non-positive or non-finite value) the kernel records the smallest failing
// (row + 1) into d_err via atomicMin; d_err must be initialized to INT_MAX
// by the caller (0 rows failing leaves it at INT_MAX).
void launch_extract_inv_diag(
    const double* d_values,
    const int* d_colind,
    const int* d_rowptr,
    double* d_inv_diag,
    int dim,
    int* d_err,
    cudaStream_t stream
);

// z[i] = inv_diag[i] * r[i]; asynchronous on `stream`.
void launch_apply_diag_precond(
    const double* d_inv_diag,
    const double* d_r,
    double* d_z,
    int dim,
    cudaStream_t stream
);

SPLRBackwardContext::SPLRBackwardContext(int problem_size, int max_nnz) :
    m_Hsize(problem_size), m_max_nnz(max_nnz)
{
    // CSR backing buffers
    CUDA_CHECK(cudaMalloc(&d_Avalues, m_max_nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Acolind, m_max_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_Arowptr, (m_Hsize + 1) * sizeof(int)));

    // Persistent rhs/solution buffers
    CUDA_CHECK(cudaMalloc(&d_rhs, m_Hsize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sol, m_Hsize * sizeof(double)));

    // Low-rank materials
    CUDA_CHECK(cudaMalloc(&d_s, m_Hsize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, m_Hsize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_invA_y, m_Hsize * sizeof(double)));

    // Diagonal preconditioner buffer (Stage 3C-3): always context-owned,
    // filled once by setup_diagonal_preconditioner() after the CSR is
    // finalized. Not freed by release_replaceable_buffers().
    CUDA_CHECK(cudaMalloc(&d_inv_diag, m_Hsize * sizeof(double)));

    // Snapshot mode owns all replaceable buffers allocated above.
    m_owns_csr = true;
    m_owns_rhs = true;
    m_owns_sol = true;
    m_owns_s = true;
    m_owns_y = true;
    m_owns_invA_y = true;

    // Pinned host workspace for the reduction kernels
    CUDA_CHECK(cudaHostAlloc(
        &h_pinned,
        (4 * MAX_NUM_BLOCK_1D) * sizeof(double),
        cudaHostAllocMapped
    ));
}

SPLRBackwardContext::~SPLRBackwardContext()
{
    // First destroy all cuDSS wrappers and solver state. After reset(),
    // no cudssMatrix_t refers to the backing buffers below.
    m_solver.reset();

    // Then free the user buffers referenced by the now-destroyed wrappers.
    release_buffers();
}

void SPLRBackwardContext::release_replaceable_buffers()
{
    // Do not abort on errors: this helper is also used during destruction,
    // when the CUDA context may already be torn down. Ownership flags are
    // essential in adopted mode because d_y/d_invA_y are aliases.
    if (m_owns_csr)
    {
        if (d_Avalues != nullptr) cudaFree(d_Avalues);
        if (d_Acolind != nullptr) cudaFree(d_Acolind);
        if (d_Arowptr != nullptr) cudaFree(d_Arowptr);
    }
    if (m_owns_rhs && d_rhs != nullptr) cudaFree(d_rhs);
    if (m_owns_sol && d_sol != nullptr) cudaFree(d_sol);
    if (m_owns_s && d_s != nullptr) cudaFree(d_s);
    if (m_owns_y && d_y != nullptr) cudaFree(d_y);
    if (m_owns_invA_y && d_invA_y != nullptr) cudaFree(d_invA_y);

    d_Avalues = nullptr;
    d_Acolind = nullptr;
    d_Arowptr = nullptr;
    d_rhs = nullptr;
    d_sol = nullptr;
    d_s = nullptr;
    d_y = nullptr;
    d_invA_y = nullptr;

    m_owns_csr = false;
    m_owns_rhs = false;
    m_owns_sol = false;
    m_owns_s = false;
    m_owns_y = false;
    m_owns_invA_y = false;
}

void SPLRBackwardContext::release_buffers()
{
    // Do not abort on errors: the CUDA context may already be torn down
    // when the Python interpreter finalizes. cuDSS handles are not touched
    // here; m_solver must already have been reset by the caller.
    release_replaceable_buffers();
    if (d_ctx_point != nullptr) { cudaFree(d_ctx_point);   d_ctx_point = nullptr; }
    if (d_inv_diag  != nullptr) { cudaFree(d_inv_diag);    d_inv_diag = nullptr; }
    m_inv_diag_ready = false;
    if (h_pinned    != nullptr) { cudaFreeHost(h_pinned);  h_pinned = nullptr; }
}

SPLRBackwardContext::SPLRBackwardContext(SPLRBackwardContext&& other) noexcept :
    m_solver(std::move(other.m_solver))
{
    // Steal scalar state
    m_Hsize    = other.m_Hsize;
    m_max_nnz  = other.m_max_nnz;
    m_nnz      = other.m_nnz;
    m_reg      = other.m_reg;
    m_ys       = other.m_ys;
    m_low_rank = other.m_low_rank;
    m_fallback = other.m_fallback;
    m_valid    = other.m_valid;
    m_adopted_forward_state = other.m_adopted_forward_state;
    m_owns_csr = other.m_owns_csr;
    m_owns_rhs = other.m_owns_rhs;
    m_owns_sol = other.m_owns_sol;
    m_owns_s = other.m_owns_s;
    m_owns_y = other.m_owns_y;
    m_owns_invA_y = other.m_owns_invA_y;

    // Steal buffers
    d_Avalues   = other.d_Avalues;
    d_Acolind   = other.d_Acolind;
    d_Arowptr   = other.d_Arowptr;
    d_rhs       = other.d_rhs;
    d_sol       = other.d_sol;
    d_s         = other.d_s;
    d_y         = other.d_y;
    d_invA_y    = other.d_invA_y;
    d_ctx_point = other.d_ctx_point;
    h_pinned    = other.h_pinned;
    d_inv_diag  = other.d_inv_diag;
    m_inv_diag_ready = other.m_inv_diag_ready;

    // Leave the source empty
    other.m_Hsize = 0;
    other.m_max_nnz = 0;
    other.m_nnz = 0;
    other.m_reg = 0.0;
    other.m_ys = 0.0;
    other.m_low_rank = false;
    other.m_fallback = false;
    other.m_valid = false;
    other.m_adopted_forward_state = false;
    other.m_owns_csr = false;
    other.m_owns_rhs = false;
    other.m_owns_sol = false;
    other.m_owns_s = false;
    other.m_owns_y = false;
    other.m_owns_invA_y = false;
    other.d_Avalues = nullptr;
    other.d_Acolind = nullptr;
    other.d_Arowptr = nullptr;
    other.d_rhs = nullptr;
    other.d_sol = nullptr;
    other.d_s = nullptr;
    other.d_y = nullptr;
    other.d_invA_y = nullptr;
    other.d_ctx_point = nullptr;
    other.h_pinned = nullptr;
    other.d_inv_diag = nullptr;
    other.m_inv_diag_ready = false;
}

SPLRBackwardContext& SPLRBackwardContext::operator=(SPLRBackwardContext&& other) noexcept
{
    if (this != &other)
    {
        // Destroy the old cuDSS state before freeing the buffers it references.
        m_solver.reset();
        release_buffers();

        // Steal the source solver and scalar state
        m_solver   = std::move(other.m_solver);
        m_Hsize    = other.m_Hsize;
        m_max_nnz  = other.m_max_nnz;
        m_nnz      = other.m_nnz;
        m_reg      = other.m_reg;
        m_ys       = other.m_ys;
        m_low_rank = other.m_low_rank;
        m_fallback = other.m_fallback;
        m_valid    = other.m_valid;
        m_adopted_forward_state = other.m_adopted_forward_state;
        m_owns_csr = other.m_owns_csr;
        m_owns_rhs = other.m_owns_rhs;
        m_owns_sol = other.m_owns_sol;
        m_owns_s = other.m_owns_s;
        m_owns_y = other.m_owns_y;
        m_owns_invA_y = other.m_owns_invA_y;

        // Steal buffers
        d_Avalues   = other.d_Avalues;
        d_Acolind   = other.d_Acolind;
        d_Arowptr   = other.d_Arowptr;
        d_rhs       = other.d_rhs;
        d_sol       = other.d_sol;
        d_s         = other.d_s;
        d_y         = other.d_y;
        d_invA_y    = other.d_invA_y;
        d_ctx_point = other.d_ctx_point;
        h_pinned    = other.h_pinned;
        d_inv_diag  = other.d_inv_diag;
        m_inv_diag_ready = other.m_inv_diag_ready;

        // Leave the source empty
        other.m_Hsize = 0;
        other.m_max_nnz = 0;
        other.m_nnz = 0;
        other.m_reg = 0.0;
        other.m_ys = 0.0;
        other.m_low_rank = false;
        other.m_fallback = false;
        other.m_valid = false;
        other.m_adopted_forward_state = false;
        other.m_owns_csr = false;
        other.m_owns_rhs = false;
        other.m_owns_sol = false;
        other.m_owns_s = false;
        other.m_owns_y = false;
        other.m_owns_invA_y = false;
        other.d_Avalues = nullptr;
        other.d_Acolind = nullptr;
        other.d_Arowptr = nullptr;
        other.d_rhs = nullptr;
        other.d_sol = nullptr;
        other.d_s = nullptr;
        other.d_y = nullptr;
        other.d_invA_y = nullptr;
        other.d_ctx_point = nullptr;
        other.h_pinned = nullptr;
        other.d_inv_diag = nullptr;
        other.m_inv_diag_ready = false;
    }
    return *this;
}

void SPLRBackwardContext::snapshot_sphess(
    const double* d_values, const int* d_colind, const int* d_rowptr,
    size_t nnz
)
{
    if (m_valid || m_solver || m_adopted_forward_state)
    {
        fprintf(stderr, "SPLRBackwardContext: cannot snapshot after initialization/adoption\n");
        exit(EXIT_FAILURE);
    }
    if ((int)nnz > m_max_nnz)
    {
        fprintf(stderr, "SPLRBackwardContext: CSR snapshot exceeds capacity\n");
        exit(EXIT_FAILURE);
    }

    // Synchronous copies: ordered with respect to prior work on the blocking
    // solver stream and the legacy default stream at this point in forward.
    CUDA_CHECK(cudaMemcpy(
        d_Avalues, d_values, nnz * sizeof(double), cudaMemcpyDeviceToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        d_Acolind, d_colind, nnz * sizeof(int), cudaMemcpyDeviceToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        d_Arowptr, d_rowptr, (m_Hsize + 1) * sizeof(int), cudaMemcpyDeviceToDevice
    ));
    m_nnz = nnz;
}

void SPLRBackwardContext::factorize_snapshot()
{
    if (m_nnz == 0)
    {
        fprintf(stderr, "SPLRBackwardContext: cannot factorize an empty CSR snapshot\n");
        exit(EXIT_FAILURE);
    }
    if (m_valid)
    {
        fprintf(stderr, "SPLRBackwardContext: context has already been initialized\n");
        exit(EXIT_FAILURE);
    }

    if (!m_owns_csr || !m_owns_rhs || !m_owns_sol)
    {
        fprintf(stderr, "SPLRBackwardContext: snapshot factorization requires owned CSR/rhs/sol buffers\n");
        exit(EXIT_FAILURE);
    }

    m_adopted_forward_state = false;

    // Create the solver lazily so that an unused context does not create a
    // cuDSS handle. The solver is owned by this context and factorizes the
    // private CSR snapshot; no forward solver state is moved or rebound.
    if (!m_solver)
    {
        m_solver = std::make_unique<SparseCholeskySolver>();
    }

    m_solver->set_A(d_Avalues, d_Acolind, d_Arowptr, m_Hsize, m_nnz);
    m_solver->set_b(d_rhs, m_Hsize, 1);
    m_solver->set_x(d_sol, m_Hsize, 1);
    m_solver->analyze();
    m_solver->factorize();
}

void SPLRBackwardContext::initialize_from_snapshot(
    const double* d_s_fwd, const double* d_y_fwd,
    const double* d_invA_y_fwd,
    double ys, bool low_rank, double reg,
    const double* d_ctx_pt
)
{
    m_ys = ys;
    m_reg = reg;
    m_low_rank = low_rank;
    m_fallback = false;

    // Copy the low-rank materials. The context does not take ownership of
    // the forward buffers, so private copies are required.
    if (low_rank)
    {
        CUDA_CHECK(cudaMemcpy(
            d_s, d_s_fwd, m_Hsize * sizeof(double), cudaMemcpyDeviceToDevice
        ));
        CUDA_CHECK(cudaMemcpy(
            d_y, d_y_fwd, m_Hsize * sizeof(double), cudaMemcpyDeviceToDevice
        ));
        CUDA_CHECK(cudaMemcpy(
            d_invA_y, d_invA_y_fwd, m_Hsize * sizeof(double),
            cudaMemcpyDeviceToDevice
        ));
    }

    // Copy the snapshot point for validation
    if (d_ctx_pt != nullptr)
    {
        if (d_ctx_point == nullptr)
        {
            CUDA_CHECK(cudaMalloc(&d_ctx_point, m_Hsize * sizeof(double)));
        }
        CUDA_CHECK(cudaMemcpy(
            d_ctx_point, d_ctx_pt, m_Hsize * sizeof(double),
            cudaMemcpyDeviceToDevice
        ));
    }

    // Fresh factorization of the owned snapshot. This path is retained for
    // fallback/debug validation; the normal Stage-3B.2-C2 path instead adopts
    // the forward factorization directly.
    factorize_snapshot();
    m_valid = true;
}

void SPLRBackwardContext::factorize_fallback(double reg, const double* d_ctx_pt)
{
    m_reg = reg;
    m_ys = 0.0;
    m_low_rank = false;
    m_fallback = true;

    // Copy the final point for validation
    if (d_ctx_pt != nullptr)
    {
        if (d_ctx_point == nullptr)
        {
            CUDA_CHECK(cudaMalloc(&d_ctx_point, m_Hsize * sizeof(double)));
        }
        CUDA_CHECK(cudaMemcpy(
            d_ctx_point, d_ctx_pt, m_Hsize * sizeof(double),
            cudaMemcpyDeviceToDevice
        ));
    }

    factorize_snapshot();
    m_valid = true;
}

void SPLRBackwardContext::adopt_forward_state(
    std::unique_ptr<SparseCholeskySolver> solver,
    double* d_values, int* d_colind, int* d_rowptr,
    int dim, size_t nnz,
    double* d_rhs_fwd, double* d_sol_fwd,
    double* d_s_fwd,
    double ys, bool low_rank, double reg, bool fallback,
    const double* d_ctx_pt
)
{
    if (m_valid || m_solver || m_adopted_forward_state)
    {
        fprintf(stderr, "SPLRBackwardContext: cannot adopt into an initialized context\n");
        exit(EXIT_FAILURE);
    }
    if (!solver || d_values == nullptr || d_colind == nullptr ||
        d_rowptr == nullptr || d_rhs_fwd == nullptr ||
        d_sol_fwd == nullptr || d_s_fwd == nullptr)
    {
        fprintf(stderr, "SPLRBackwardContext: incomplete forward state adoption\n");
        exit(EXIT_FAILURE);
    }
    if (dim != m_Hsize)
    {
        fprintf(stderr, "SPLRBackwardContext: adopted dimension mismatch\n");
        exit(EXIT_FAILURE);
    }
    if ((int)nnz > m_max_nnz)
    {
        fprintf(stderr, "SPLRBackwardContext: adopted CSR exceeds capacity\n");
        exit(EXIT_FAILURE);
    }

    // This context was constructed with snapshot-mode buffers. They are no
    // longer needed once the exact forward-owned buffers are adopted. The
    // ownership-aware release prevents double-freeing aliases if this API is
    // called on a partially initialized object.
    release_replaceable_buffers();

    // Adopt the solver and all base allocations referenced by its cuDSS
    // wrappers. The caller must null its corresponding owning pointers after
    // this call returns.
    m_solver = std::move(solver);

    d_Avalues = d_values;
    d_Acolind = d_colind;
    d_Arowptr = d_rowptr;
    d_rhs = d_rhs_fwd;
    d_sol = d_sol_fwd;
    d_s = d_s_fwd;

    m_owns_csr = true;
    m_owns_rhs = true;
    m_owns_sol = true;
    m_owns_s = true;

    // In the forward solver, rhs/sol are always allocated with two columns:
    //   rhs = [grad, y], sol = [direc, invA_y].
    // The second columns exist even when low_rank is false, but they are only
    // meaningful when low_rank is true. They remain non-owning aliases here.
    if (low_rank)
    {
        d_y = d_rhs + m_Hsize;
        d_invA_y = d_sol + m_Hsize;
    }
    else
    {
        d_y = nullptr;
        d_invA_y = nullptr;
    }
    m_owns_y = false;
    m_owns_invA_y = false;

    m_nnz = nnz;
    m_ys = ys;
    m_reg = reg;
    m_low_rank = low_rank;
    m_fallback = fallback;
    m_adopted_forward_state = true;

    // Copy the validation point before the source forward solver is allowed
    // to release its remaining forward-only buffers.
    if (d_ctx_pt != nullptr)
    {
        if (d_ctx_point == nullptr)
        {
            CUDA_CHECK(cudaMalloc(&d_ctx_point, m_Hsize * sizeof(double)));
        }
        CUDA_CHECK(cudaMemcpy(
            d_ctx_point, d_ctx_pt, m_Hsize * sizeof(double),
            cudaMemcpyDeviceToDevice
        ));
    }

    // The adopted solver must already have completed analyze + factorization.
    // No reorder/symfac/factorize is performed here.
    m_valid = true;
}

void SPLRBackwardContext::apply_preconditioner(const double* d_r, double* d_z)
{
    if (!m_valid || !m_solver)
    {
        fprintf(stderr, "SPLRBackwardContext: invalid context\n");
        exit(EXIT_FAILURE);
    }

    cudaStream_t stream = m_solver->get_cuda_stream();

    // Copy r into the persistent rhs buffer. The synchronous memcpy is
    // ordered after prior work on the legacy default stream (e.g. the torch
    // operations that produced r), and the cuDSS solve below is issued after
    // the copy has completed.
    CUDA_CHECK(cudaMemcpy(
        d_rhs, d_r, m_Hsize * sizeof(double), cudaMemcpyDeviceToDevice
    ));

    // Cholesky solve: d_sol = A^{-1} r
    m_solver->solve();

    // Rank-2 Woodbury correction, identical to the forward search-direction
    // update rule but without the final -reg scaling:
    //   B^{-1} r = q - (s^T r / ys) * u
    //              + [(1/reg + y^T u / ys) * (s^T r / ys) - y^T q / ys] * s
    // with q = A^{-1} r and u = A^{-1} y.
    if (m_low_rank)
    {
        launch_low_rank_search_direc(
            d_sol, d_invA_y, d_rhs, d_y, d_s,
            m_ys, m_reg, m_Hsize, h_pinned, stream
        );
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(
        d_z, d_sol, m_Hsize * sizeof(double), cudaMemcpyDeviceToDevice
    ));
}

// ---------------------------------------------------------------------------
// Stage 3C-3: native diagonal preconditioner
// ---------------------------------------------------------------------------

// File-local helper: run the validated extraction once on `stream`.
// Returns the smallest failing row index (0-based), or -1 on success.
// Uses a temporary device int for the error flag; synchronizes `stream`
// once to read it back. One-time setup cost only -- never on the PCG path.
static int extract_inv_diag_validated(
    const double* d_values, const int* d_colind, const int* d_rowptr,
    double* d_inv_diag, int dim, cudaStream_t stream
)
{
    int* d_err = nullptr;
    CUDA_CHECK(cudaMalloc(&d_err, sizeof(int)));
    const int h_init = INT_MAX; // sentinel: "no failing row"
    CUDA_CHECK(cudaMemcpyAsync(
        d_err, &h_init, sizeof(int), cudaMemcpyHostToDevice, stream
    ));
    launch_extract_inv_diag(
        d_values, d_colind, d_rowptr, d_inv_diag, dim, d_err, stream
    );
    int h_err = INT_MAX;
    CUDA_CHECK(cudaMemcpyAsync(
        &h_err, d_err, sizeof(int), cudaMemcpyDeviceToHost, stream
    ));
    // Ordered on the same stream; safe for both default and non-default
    // PyTorch current streams (unlike a legacy-stream synchronous memcpy).
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_err));
    return (h_err == INT_MAX) ? -1 : (h_err - 1);
}

void SPLRBackwardContext::setup_diagonal_preconditioner(cudaStream_t stream)
{
    if (!m_valid)
    {
        fprintf(stderr, "SPLRBackwardContext: diagonal setup on invalid context\n");
        exit(EXIT_FAILURE);
    }
    if (m_inv_diag_ready)
    {
        return; // idempotent: extracted once per context
    }
    const int bad_row = extract_inv_diag_validated(
        d_Avalues, d_Acolind, d_Arowptr, d_inv_diag, m_Hsize, stream
    );
    if (bad_row >= 0)
    {
        // Guaranteed not to happen by construction (sparsification keeps the
        // diagonal and the shift makes it positive); a violation is a bug.
        fprintf(stderr,
                "SPLRBackwardContext: CSR diagonal missing/non-positive/"
                "non-finite at row %d\n", bad_row);
        exit(EXIT_FAILURE);
    }
    m_inv_diag_ready = true;
}

void SPLRBackwardContext::apply_diagonal_preconditioner(
    const double* d_r, double* d_z, cudaStream_t stream
) const
{
    if (!m_valid || !m_inv_diag_ready)
    {
        fprintf(stderr, "SPLRBackwardContext: diagonal preconditioner not set up\n");
        exit(EXIT_FAILURE);
    }
    // Single elementwise kernel; no memcpy through d_rhs, no cuDSS, no
    // stream synchronization, no temporary allocation. Ordering with the
    // producer/consumer of d_r/d_z is inherited from `stream`.
    launch_apply_diag_precond(d_inv_diag, d_r, d_z, m_Hsize, stream);
}

double SPLRBackwardContext::benchmark_diag_setup_repeated(
    cudaStream_t stream, int repeats
)
{
    if (!m_valid)
    {
        fprintf(stderr, "SPLRBackwardContext: invalid context\n");
        exit(EXIT_FAILURE);
    }
    if (repeats < 1)
    {
        fprintf(stderr, "benchmark_diag_setup_repeated: repeats must be >= 1\n");
        exit(EXIT_FAILURE);
    }
    std::vector<double> samples;
    samples.reserve(repeats);
    for (int i = 0; i < repeats; ++i)
    {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto t0 = std::chrono::steady_clock::now();
        const int bad_row = extract_inv_diag_validated(
            d_Avalues, d_Acolind, d_Arowptr, d_inv_diag, m_Hsize, stream
        );
        const auto t1 = std::chrono::steady_clock::now();
        if (bad_row >= 0)
        {
            fprintf(stderr,
                    "SPLRBackwardContext: CSR diagonal invalid at row %d\n",
                    bad_row);
            exit(EXIT_FAILURE);
        }
        samples.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count()
        );
    }
    std::sort(samples.begin(), samples.end());
    m_inv_diag_ready = true;
    return samples[samples.size() / 2]; // median, microseconds
}

// Debug/benchmark-only (Stage 3C-2A). Identical math and execution order to
// apply_preconditioner(); the only additions are std::chrono host timing and
// CUDA events on the solver stream.
//
// Caller-owned events: every event used anywhere along the call (including
// the ones recorded INSIDE launch_low_rank_search_direc) is created here,
// passed to the helper through profile.lowrank.ev_*, queried here after the
// stream has been synchronized, and destroyed here. This is required because
// cudaEventElapsedTime needs both events to have completed, while the
// correction kernel is still in flight when the helper returns.
//
// Attribution note (measured structure of the real code path):
// launch_low_rank_search_direc contains an INTERNAL cudaStreamSynchronize
// between its dot kernel and its host-side scalar merge. In low-rank mode
// that internal sync is where the host actually waits for the cuDSS solve
// (the dot kernel is queued behind the solve on the same stream), so
// lowrank.sync_wait_host_us absorbs the remaining GPU solve time. The
// final cudaStreamSynchronize below then only waits for the correction
// kernel. gpu_cudss_solve_us measures the true GPU solve time regardless
// of which host call happens to wait for it.
void SPLRBackwardContext::apply_preconditioner_profiled(
    const double* d_r, double* d_z, PreconditionerProfile& profile
)
{
    if (!m_valid || !m_solver)
    {
        fprintf(stderr, "SPLRBackwardContext: invalid context\n");
        exit(EXIT_FAILURE);
    }

    cudaStream_t stream = m_solver->get_cuda_stream();

    profile.low_rank_active = m_low_rank;
    profile.lowrank = LowRankSearchDirecProfile{};  // zero-reset for reuse

    cudaEvent_t ev_solve_start, ev_solve_stop;
    CUDA_CHECK(cudaEventCreate(&ev_solve_start));
    CUDA_CHECK(cudaEventCreate(&ev_solve_stop));
    if (m_low_rank)
    {
        CUDA_CHECK(cudaEventCreate(&profile.lowrank.ev_reduce_start));
        CUDA_CHECK(cudaEventCreate(&profile.lowrank.ev_reduce_stop));
        CUDA_CHECK(cudaEventCreate(&profile.lowrank.ev_corr_start));
        CUDA_CHECK(cudaEventCreate(&profile.lowrank.ev_corr_stop));
    }

    using clock = std::chrono::steady_clock;
    auto us = [](const clock::time_point& a, const clock::time_point& b) {
        return std::chrono::duration<double, std::micro>(b - a).count();
    };

    const auto t_total0 = clock::now();

    // Copy r into the persistent rhs buffer (same blocking call as the
    // production path).
    const auto t0 = clock::now();
    CUDA_CHECK(cudaMemcpy(
        d_rhs, d_r, m_Hsize * sizeof(double), cudaMemcpyDeviceToDevice
    ));
    const auto t1 = clock::now();

    // Cholesky solve: d_sol = A^{-1} r (async on the solver stream)
    CUDA_CHECK(cudaEventRecord(ev_solve_start, stream));
    m_solver->solve();
    CUDA_CHECK(cudaEventRecord(ev_solve_stop, stream));
    const auto t2 = clock::now();

    // Rank-2 Woodbury correction (same call as the production path; the
    // profile argument only turns on instrumentation inside the helper).
    if (m_low_rank)
    {
        launch_low_rank_search_direc(
            d_sol, d_invA_y, d_rhs, d_y, d_s,
            m_ys, m_reg, m_Hsize, h_pinned, stream,
            &profile.lowrank
        );
    }
    const auto t3 = clock::now();

    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto t4 = clock::now();

    CUDA_CHECK(cudaMemcpy(
        d_z, d_sol, m_Hsize * sizeof(double), cudaMemcpyDeviceToDevice
    ));
    const auto t5 = clock::now();

    // All recorded events have completed at this point (the stream was
    // synchronized above).
    float solve_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&solve_ms, ev_solve_start, ev_solve_stop));
    if (m_low_rank)
    {
        float reduce_ms = 0.0f, corr_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(
            &reduce_ms, profile.lowrank.ev_reduce_start,
            profile.lowrank.ev_reduce_stop
        ));
        CUDA_CHECK(cudaEventElapsedTime(
            &corr_ms, profile.lowrank.ev_corr_start,
            profile.lowrank.ev_corr_stop
        ));
        profile.lowrank.gpu_reduce_us = (double)reduce_ms * 1e3;
        profile.lowrank.gpu_correction_us = (double)corr_ms * 1e3;
    }

    CUDA_CHECK(cudaEventDestroy(ev_solve_start));
    CUDA_CHECK(cudaEventDestroy(ev_solve_stop));
    if (m_low_rank)
    {
        CUDA_CHECK(cudaEventDestroy(profile.lowrank.ev_reduce_start));
        CUDA_CHECK(cudaEventDestroy(profile.lowrank.ev_reduce_stop));
        CUDA_CHECK(cudaEventDestroy(profile.lowrank.ev_corr_start));
        CUDA_CHECK(cudaEventDestroy(profile.lowrank.ev_corr_stop));
        profile.lowrank.ev_reduce_start = nullptr;
        profile.lowrank.ev_reduce_stop = nullptr;
        profile.lowrank.ev_corr_start = nullptr;
        profile.lowrank.ev_corr_stop = nullptr;
    }

    profile.host_memcpy_in_us = us(t0, t1);
    profile.host_solve_launch_us = us(t1, t2);
    // Whole helper call. In low-rank mode this interval CONTAINS the host
    // wait for (the remainder of) the GPU solve, because of the helper's
    // internal cudaStreamSynchronize -- see the attribution note above.
    profile.host_lowrank_total_us = m_low_rank ? us(t2, t3) : 0.0;
    // In low-rank mode this only waits for the correction kernel; in
    // non-low-rank mode it waits for the whole GPU solve.
    profile.host_final_sync_wait_us = us(t3, t4);
    profile.host_memcpy_out_us = us(t4, t5);
    profile.host_total_us = us(t_total0, t5);
    profile.gpu_cudss_solve_us = (double)solve_ms * 1e3;
}

// Debug/benchmark-only (Stage 3C-2A). C++ loop over the production
// apply_preconditioner(); returns the total host wall time in microseconds.
double SPLRBackwardContext::benchmark_preconditioner_repeated(
    const double* d_r, double* d_z, int repeats
)
{
    if (!m_valid || !m_solver)
    {
        fprintf(stderr, "SPLRBackwardContext: invalid context\n");
        exit(EXIT_FAILURE);
    }
    if (repeats <= 0)
    {
        return 0.0;
    }

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    for (int i = 0; i < repeats; i++)
    {
        apply_preconditioner(d_r, d_z);
    }
    const auto t1 = clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

// Debug/benchmark-only (Stage 3C-2B). Pure A^{-1} r: identical to
// apply_preconditioner() minus the low-rank Woodbury correction, plus
// chrono/CUDA-event instrumentation. Source-level fact (linsolve.cu):
// SparseCholeskySolver::solve() is a single cudssExecute(CUDSS_PHASE_SOLVE)
// -- no descriptor rebinding, no config/data update, no sync inside.
void SPLRBackwardContext::apply_base_solve_profiled(
    const double* d_r, double* d_z, BaseSolveProfile& profile
)
{
    if (!m_valid || !m_solver)
    {
        fprintf(stderr, "SPLRBackwardContext: invalid context\n");
        exit(EXIT_FAILURE);
    }

    cudaStream_t stream = m_solver->get_cuda_stream();

    cudaEvent_t ev_solve_start, ev_solve_stop;
    CUDA_CHECK(cudaEventCreate(&ev_solve_start));
    CUDA_CHECK(cudaEventCreate(&ev_solve_stop));

    using clock = std::chrono::steady_clock;
    auto us = [](const clock::time_point& a, const clock::time_point& b) {
        return std::chrono::duration<double, std::micro>(b - a).count();
    };

    const auto t_total0 = clock::now();

    const auto t0 = clock::now();
    CUDA_CHECK(cudaMemcpy(
        d_rhs, d_r, m_Hsize * sizeof(double), cudaMemcpyDeviceToDevice
    ));
    const auto t1 = clock::now();

    // Cholesky solve: d_sol = A^{-1} r. host_cudss_execute_us is the host
    // wall time inside the cuDSS API call; it is NOT pure launch overhead
    // (cuDSS may block the host until part or all of the GPU solve is done).
    CUDA_CHECK(cudaEventRecord(ev_solve_start, stream));
    const auto t2 = clock::now();
    m_solver->solve();
    const auto t3 = clock::now();
    CUDA_CHECK(cudaEventRecord(ev_solve_stop, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto t4 = clock::now();

    CUDA_CHECK(cudaMemcpy(
        d_z, d_sol, m_Hsize * sizeof(double), cudaMemcpyDeviceToDevice
    ));
    const auto t5 = clock::now();

    float solve_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&solve_ms, ev_solve_start, ev_solve_stop));
    CUDA_CHECK(cudaEventDestroy(ev_solve_start));
    CUDA_CHECK(cudaEventDestroy(ev_solve_stop));

    profile.host_memcpy_in_us = us(t0, t1);
    profile.host_cudss_execute_us = us(t2, t3);
    profile.host_sync_wait_us = us(t3, t4);
    profile.host_memcpy_out_us = us(t4, t5);
    profile.host_total_us = us(t_total0, t5);
    profile.gpu_solve_us = (double)solve_ms * 1e3;
}

// Debug/benchmark-only (Stage 3C-2B). C++ loop over the uninstrumented
// base-solve sequence; returns the total host wall time in microseconds.
double SPLRBackwardContext::benchmark_base_solve_repeated(
    const double* d_r, double* d_z, int repeats
)
{
    if (!m_valid || !m_solver)
    {
        fprintf(stderr, "SPLRBackwardContext: invalid context\n");
        exit(EXIT_FAILURE);
    }
    if (repeats <= 0)
    {
        return 0.0;
    }

    cudaStream_t stream = m_solver->get_cuda_stream();

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    for (int i = 0; i < repeats; i++)
    {
        // Same sequence as the non-low-rank portion of the production
        // apply_preconditioner().
        CUDA_CHECK(cudaMemcpy(
            d_rhs, d_r, m_Hsize * sizeof(double), cudaMemcpyDeviceToDevice
        ));
        m_solver->solve();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(
            d_z, d_sol, m_Hsize * sizeof(double), cudaMemcpyDeviceToDevice
        ));
    }
    const auto t1 = clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
}
