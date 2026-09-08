#include "backward_context.h"

#include <cstdio>
#include <cstdlib>
#include <utility>

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

void SPLRBackwardContext::release_buffers()
{
    // Do not abort on errors: the CUDA context may already be torn down
    // when the Python interpreter finalizes. cuDSS handles are not touched
    // here; m_solver must already have been reset by the caller.
    if (d_Avalues   != nullptr) { cudaFree(d_Avalues);     d_Avalues = nullptr; }
    if (d_Acolind   != nullptr) { cudaFree(d_Acolind);     d_Acolind = nullptr; }
    if (d_Arowptr   != nullptr) { cudaFree(d_Arowptr);     d_Arowptr = nullptr; }
    if (d_rhs       != nullptr) { cudaFree(d_rhs);         d_rhs = nullptr; }
    if (d_sol       != nullptr) { cudaFree(d_sol);         d_sol = nullptr; }
    if (d_s         != nullptr) { cudaFree(d_s);           d_s = nullptr; }
    if (d_y         != nullptr) { cudaFree(d_y);           d_y = nullptr; }
    if (d_invA_y    != nullptr) { cudaFree(d_invA_y);      d_invA_y = nullptr; }
    if (d_ctx_point != nullptr) { cudaFree(d_ctx_point);   d_ctx_point = nullptr; }
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

    // Leave the source empty
    other.m_Hsize = 0;
    other.m_max_nnz = 0;
    other.m_nnz = 0;
    other.m_reg = 0.0;
    other.m_ys = 0.0;
    other.m_low_rank = false;
    other.m_fallback = false;
    other.m_valid = false;
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

        // Leave the source empty
        other.m_Hsize = 0;
        other.m_max_nnz = 0;
        other.m_nnz = 0;
        other.m_reg = 0.0;
        other.m_ys = 0.0;
        other.m_low_rank = false;
        other.m_fallback = false;
        other.m_valid = false;
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
    }
    return *this;
}

void SPLRBackwardContext::snapshot_sphess(
    const double* d_values, const int* d_colind, const int* d_rowptr,
    size_t nnz
)
{
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

    // Fresh factorization of the owned snapshot. This is the conservative
    // Stage-3B.1 correctness baseline; true forward-factorization ownership
    // transfer is intentionally left for a later stage.
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
