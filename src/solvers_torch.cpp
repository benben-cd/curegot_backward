// Only compile this file when PyTorch is available
#if defined(TORCH_BUILD)

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "solvers.h"
#include "backward_context.h"

namespace py = pybind11;

// PyTorch interface for sinkhorn_bcd function
py::dict torch_sinkhorn_bcd(
    torch::Tensor M,
    torch::Tensor a,
    torch::Tensor b,
    double reg,
    double tol,
    int max_iter,
    int verbose,
    const py::kwargs& kwargs
)
{
    // Check input dimensions
    if (M.dim() != 2)
    {
        throw std::runtime_error("M must be a 2D tensor");
    }
    if (a.dim() != 1 || b.dim() != 1)
    {
        throw std::runtime_error("a and b must be 1D tensors");
    }

    int n = M.size(0);
    int m = M.size(1);

    if (a.size(0) != n || b.size(0) != m)
    {
        throw std::runtime_error("Shape mismatch: M is [n x m], a should be length n, b should be length m");
    }

    // Ensure that inputs are on GPU with a double type
    // Zero-copy if already satisfying the conditions
    auto src_options = M.options();
    auto src_device = M.device();
    // If M is already on GPU, preserve its device ID
    // Otherwise use the default device
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    if (src_device.is_cuda())
    {
        options = options.device(torch::kCUDA, src_device.index());
    }
    M = M.to(options);
    a = a.to(options);
    b = b.to(options);

    // Set active device
    const at::cuda::CUDAGuard device_guard(options.device());

    // Ensure contiguous and get data pointers
    M = M.contiguous();
    a = a.contiguous();
    b = b.contiguous();
    // Pointing to device memory
    const double* d_M_ptr = M.data_ptr<double>();
    const double* d_a_ptr = a.data_ptr<double>();
    const double* d_b_ptr = b.data_ptr<double>();

    // Handle kwargs for initial values (x0)
    const double* d_x0_ptr = nullptr;
    torch::Tensor x0_storage;

    if (kwargs.contains("x0"))
    {
        torch::Tensor x0 = py::cast<torch::Tensor>(kwargs["x0"]);
        if (x0.dim() != 1 || x0.size(0) != n + m)
        {
            throw std::runtime_error("x0 must be a 1D tensor of length n + m");
        }

        // Shallow copy x0 to x0_storage if the device, type, and layout of x0 are compatible
        // Otherwise make a conversion
        x0_storage = x0.to(options).contiguous();
        d_x0_ptr = x0_storage.data_ptr<double>();
    }

    // Create output tensors that are double and on device
    torch::Tensor P = torch::empty({n, m}, options);
    torch::Tensor dual = torch::empty(n + m, options);
    // Pointing to device memory
    double* d_P_ptr = P.data_ptr<double>();
    double* d_dual_ptr = dual.data_ptr<double>();

    // Call CUDA function for BCD algorithm
    int niter = 0;
    cuda_sinkhorn_bcd(
        d_M_ptr, d_a_ptr, d_b_ptr, d_P_ptr,
        reg, max_iter, tol, n, m, &niter,
        d_x0_ptr, d_dual_ptr,
        true, true
    );

    // Convert P and dual when necessary using the input tensor options
    P = P.to(src_options);
    dual = dual.to(src_options);

    // Create result dictionary
    py::dict result;
    result["niter"] = niter;
    result["plan"] = P;
    result["dual"] = dual;

    if (verbose > 0)
    {
        std::cout << "torch_sinkhorn_bcd (CUDA implementation) completed" << std::endl;
        std::cout << "Input shape: [" << n << " x " << m << "]" << std::endl;
        std::cout << "Regularization parameter: " << reg << std::endl;
        std::cout << "Tolerance: " << tol << std::endl;
        std::cout << "Max iterations: " << max_iter << std::endl;
        std::cout << "Actual iterations: " << niter << std::endl;
        if (niter == max_iter)
        {
            std::cout << "Warning: Maximum iterations reached without convergence" << std::endl;
        }
    }

    return result;
}

// PyTorch interface for sinkhorn_splr function
py::dict torch_sinkhorn_splr(
    torch::Tensor M,
    torch::Tensor a,
    torch::Tensor b,
    double reg,
    double tol,
    int max_iter,
    int verbose,
    const py::kwargs& kwargs
)
{
    // Check input dimensions
    if (M.dim() != 2)
    {
        throw std::runtime_error("M must be a 2D tensor");
    }
    if (a.dim() != 1 || b.dim() != 1)
    {
        throw std::runtime_error("a and b must be 1D tensors");
    }

    int n = M.size(0);
    int m = M.size(1);

    if (a.size(0) != n || b.size(0) != m)
    {
        throw std::runtime_error("Shape mismatch: M is [n x m], a should be length n, b should be length m");
    }

    // Ensure that inputs are on GPU with a double type
    // Zero-copy if already satisfying the conditions
    auto src_options = M.options();
    auto src_device = M.device();
    // If M is already on GPU, preserve its device ID
    // Otherwise use the default device
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    if (src_device.is_cuda())
    {
        options = options.device(torch::kCUDA, src_device.index());
    }
    M = M.to(options);
    a = a.to(options);
    b = b.to(options);

    // Set active device
    const at::cuda::CUDAGuard device_guard(options.device());

    // Ensure contiguous and get data pointers
    M = M.contiguous();
    a = a.contiguous();
    b = b.contiguous();
    // Pointing to device memory
    const double* d_M_ptr = M.data_ptr<double>();
    const double* d_a_ptr = a.data_ptr<double>();
    const double* d_b_ptr = b.data_ptr<double>();

    // Handle kwargs for initial values (x0)
    const double* d_x0_ptr = nullptr;
    torch::Tensor x0_storage;

    if (kwargs.contains("x0"))
    {
        torch::Tensor x0 = py::cast<torch::Tensor>(kwargs["x0"]);
        if (x0.dim() != 1 || x0.size(0) != n + m)
        {
            throw std::runtime_error("x0 must be a 1D tensor of length n + m");
        }

        // Shallow copy x0 to x0_storage if the device, type, and layout of x0 are compatible
        // Otherwise make a conversion
        x0_storage = x0.to(options).contiguous();
        d_x0_ptr = x0_storage.data_ptr<double>();
    }

    // Get density_max from kwargs
    // Default to 10 / min(n, m)
    double density_max = 10.0 / std::min(n, m);
    density_max = std::min(density_max, 1.0);
    if (kwargs.contains("density"))
    {
        density_max = py::cast<double>(kwargs["density"]);
        density_max = std::min(density_max, 1.0);
        density_max = std::max(density_max, 0.0);
    }

    // Get shift_max from kwargs
    double shift_max = 0.001;
    if (kwargs.contains("shift"))
    {
        shift_max = py::cast<double>(kwargs["shift"]);
        shift_max = std::max(shift_max, 0.0);
    }

    // Get sparsity_pattern_cycle from kwargs
    int sparsity_pattern_cycle = 30;
    if (kwargs.contains("sparsity_pattern_cycle"))
    {
        sparsity_pattern_cycle = py::cast<int>(kwargs["sparsity_pattern_cycle"]);
    }

    // Get candidate_sinkhorn_iter from kwargs
    int candidate_sinkhorn_iter = 3;
    if (kwargs.contains("candidate_sinkhorn_iter"))
    {
        candidate_sinkhorn_iter = py::cast<int>(kwargs["candidate_sinkhorn_iter"]);
    }

    // Create output tensors that are double and on device
    torch::Tensor P = torch::empty({n, m}, options);
    torch::Tensor dual = torch::empty(n + m, options);
    // Pointing to device memory
    double* d_P_ptr = P.data_ptr<double>();
    double* d_dual_ptr = dual.data_ptr<double>();

    // Call CUDA function for SPLR algorithm
    int niter = 0;
    cuda_sinkhorn_splr(
        d_M_ptr, d_a_ptr, d_b_ptr, d_P_ptr,
        reg, max_iter, tol, n, m, &niter,
        density_max, shift_max,
        sparsity_pattern_cycle, candidate_sinkhorn_iter, verbose,
        d_x0_ptr, d_dual_ptr,
        true, true
    );

    // Convert P and dual when necessary using the input tensor options
    P = P.to(src_options);
    dual = dual.to(src_options);

    // Create result dictionary
    py::dict result;
    result["niter"] = niter;
    result["plan"] = P;
    result["dual"] = dual;

    if (verbose > 0)
    {
        std::cout << "torch_sinkhorn_splr (CUDA implementation) completed" << std::endl;
        std::cout << "Input shape: [" << n << " x " << m << "]" << std::endl;
        std::cout << "Regularization parameter: " << reg << std::endl;
        std::cout << "Tolerance: " << tol << std::endl;
        std::cout << "Max iterations: " << max_iter << std::endl;
        std::cout << "Actual iterations: " << niter << std::endl;
        if (niter == max_iter)
        {
            std::cout << "Warning: Maximum iterations reached without convergence" << std::endl;
        }
    }

    return result;
}
// ---------------------------------------------------------------------------
// Backward context: pybind glue
// ---------------------------------------------------------------------------

// Apply the preconditioner: z = B^{-1} r, where B is the approximate Hessian
// (sparsified Hessian factorization plus the rank-2 BFGS correction) reused
// from the last forward iteration
torch::Tensor ctx_apply_preconditioner(SPLRBackwardContext& ctx, torch::Tensor r)
{
    if (!ctx.is_valid())
    {
        throw std::runtime_error("SPLRBackwardContext: invalid context");
    }
    if (r.dim() != 1 || r.size(0) != ctx.problem_size())
    {
        throw std::runtime_error("SPLRBackwardContext: r must be a 1D tensor of length n + m - 1");
    }
    if (!r.is_cuda())
    {
        throw std::runtime_error("SPLRBackwardContext: r must be a CUDA tensor");
    }

    // Ensure double type, preserving the device
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(r.device());
    r = r.to(options).contiguous();

    const at::cuda::CUDAGuard device_guard(r.device());
    torch::Tensor z = torch::empty(ctx.problem_size(), options);
    ctx.apply_preconditioner(r.data_ptr<double>(), z.data_ptr<double>());
    return z;
}

// Debug/benchmark-only (Stage 3C-2A): profiled variant of
// apply_preconditioner. Same math and execution order; returns the output
// tensor plus a host/GPU timing breakdown. The production
// apply_preconditioner is unchanged.
py::dict ctx_profile_preconditioner(SPLRBackwardContext& ctx, torch::Tensor r)
{
    if (!ctx.is_valid())
    {
        throw std::runtime_error("SPLRBackwardContext: invalid context");
    }
    if (r.dim() != 1 || r.size(0) != ctx.problem_size())
    {
        throw std::runtime_error("SPLRBackwardContext: r must be a 1D tensor of length n + m - 1");
    }
    if (!r.is_cuda())
    {
        throw std::runtime_error("SPLRBackwardContext: r must be a CUDA tensor");
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(r.device());
    r = r.to(options).contiguous();

    const at::cuda::CUDAGuard device_guard(r.device());

    const auto t_w0 = std::chrono::steady_clock::now();
    torch::Tensor z = torch::empty(ctx.problem_size(), options);

    SPLRBackwardContext::PreconditionerProfile prof;
    ctx.apply_preconditioner_profiled(
        r.data_ptr<double>(), z.data_ptr<double>(), prof
    );
    const auto t_w1 = std::chrono::steady_clock::now();

    py::dict out;
    out["output"] = z;
    out["host_wrapper_us"] =
        std::chrono::duration<double, std::micro>(t_w1 - t_w0).count();
    out["host_memcpy_in_us"] = prof.host_memcpy_in_us;
    out["host_solve_launch_us"] = prof.host_solve_launch_us;
    out["host_lowrank_total_us"] = prof.host_lowrank_total_us;
    out["host_final_sync_wait_us"] = prof.host_final_sync_wait_us;
    out["host_memcpy_out_us"] = prof.host_memcpy_out_us;
    out["host_total_us"] = prof.host_total_us;
    out["gpu_cudss_solve_us"] = prof.gpu_cudss_solve_us;
    out["low_rank_active"] = prof.low_rank_active;
    out["lowrank_setup_launch_host_us"] = prof.lowrank.setup_launch_host_us;
    out["lowrank_sync_wait_host_us"] = prof.lowrank.sync_wait_host_us;
    out["lowrank_host_scalar_us"] = prof.lowrank.host_scalar_us;
    out["lowrank_correction_launch_host_us"] =
        prof.lowrank.correction_launch_host_us;
    out["lowrank_reduce_gpu_us"] = prof.lowrank.gpu_reduce_us;
    out["lowrank_correction_gpu_us"] = prof.lowrank.gpu_correction_us;
    return out;
}

// Debug/benchmark-only (Stage 3C-2A): C++ loop over the production
// apply_preconditioner on the same RHS. Compared against a Python loop over
// the public API, this isolates the Python/PyBind/tensor-wrapper per-call
// overhead. Returns the output of the last repetition for correctness
// comparison.
py::dict ctx_benchmark_preconditioner_repeated(
    SPLRBackwardContext& ctx, torch::Tensor r, int repeats
)
{
    if (!ctx.is_valid())
    {
        throw std::runtime_error("SPLRBackwardContext: invalid context");
    }
    if (r.dim() != 1 || r.size(0) != ctx.problem_size())
    {
        throw std::runtime_error("SPLRBackwardContext: r must be a 1D tensor of length n + m - 1");
    }
    if (!r.is_cuda())
    {
        throw std::runtime_error("SPLRBackwardContext: r must be a CUDA tensor");
    }
    if (repeats <= 0)
    {
        throw std::runtime_error("repeats must be positive");
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(r.device());
    r = r.to(options).contiguous();

    const at::cuda::CUDAGuard device_guard(r.device());
    torch::Tensor z = torch::empty(ctx.problem_size(), options);

    const double total_us = ctx.benchmark_preconditioner_repeated(
        r.data_ptr<double>(), z.data_ptr<double>(), repeats
    );

    py::dict out;
    out["output"] = z;
    out["repeats"] = repeats;
    out["cpp_repeated_total_us"] = total_us;
    out["cpp_per_call_us"] = total_us / (double)repeats;
    return out;
}

// Debug/benchmark-only (Stage 3C-2B): profiled pure base solve A^{-1} r
// (no low-rank Woodbury correction). Same buffers and execution order as
// the non-low-rank portion of apply_preconditioner; the production API is
// unchanged.
py::dict ctx_profile_base_solve(SPLRBackwardContext& ctx, torch::Tensor r)
{
    if (!ctx.is_valid())
    {
        throw std::runtime_error("SPLRBackwardContext: invalid context");
    }
    if (r.dim() != 1 || r.size(0) != ctx.problem_size())
    {
        throw std::runtime_error("SPLRBackwardContext: r must be a 1D tensor of length n + m - 1");
    }
    if (!r.is_cuda())
    {
        throw std::runtime_error("SPLRBackwardContext: r must be a CUDA tensor");
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(r.device());
    r = r.to(options).contiguous();

    const at::cuda::CUDAGuard device_guard(r.device());

    const auto t_w0 = std::chrono::steady_clock::now();
    torch::Tensor z = torch::empty(ctx.problem_size(), options);

    SPLRBackwardContext::BaseSolveProfile prof;
    ctx.apply_base_solve_profiled(
        r.data_ptr<double>(), z.data_ptr<double>(), prof
    );
    const auto t_w1 = std::chrono::steady_clock::now();

    py::dict out;
    out["output"] = z;
    out["host_wrapper_us"] =
        std::chrono::duration<double, std::micro>(t_w1 - t_w0).count();
    out["host_memcpy_in_us"] = prof.host_memcpy_in_us;
    out["host_cudss_execute_us"] = prof.host_cudss_execute_us;
    out["host_sync_wait_us"] = prof.host_sync_wait_us;
    out["host_memcpy_out_us"] = prof.host_memcpy_out_us;
    out["host_total_us"] = prof.host_total_us;
    out["gpu_solve_us"] = prof.gpu_solve_us;
    out["low_rank_enabled"] = ctx.low_rank();
    return out;
}

// Debug/benchmark-only (Stage 3C-2B): C++ loop over the uninstrumented
// base-solve sequence on the same RHS. Returns the output of the last
// repetition for correctness comparison.
py::dict ctx_benchmark_base_solve_repeated(
    SPLRBackwardContext& ctx, torch::Tensor r, int repeats
)
{
    if (!ctx.is_valid())
    {
        throw std::runtime_error("SPLRBackwardContext: invalid context");
    }
    if (r.dim() != 1 || r.size(0) != ctx.problem_size())
    {
        throw std::runtime_error("SPLRBackwardContext: r must be a 1D tensor of length n + m - 1");
    }
    if (!r.is_cuda())
    {
        throw std::runtime_error("SPLRBackwardContext: r must be a CUDA tensor");
    }
    if (repeats <= 0)
    {
        throw std::runtime_error("repeats must be positive");
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(r.device());
    r = r.to(options).contiguous();

    const at::cuda::CUDAGuard device_guard(r.device());
    torch::Tensor z = torch::empty(ctx.problem_size(), options);

    const double total_us = ctx.benchmark_base_solve_repeated(
        r.data_ptr<double>(), z.data_ptr<double>(), repeats
    );

    py::dict out;
    out["output"] = z;
    out["repeats"] = repeats;
    out["cpp_repeated_total_us"] = total_us;
    out["cpp_per_call_us"] = total_us / (double)repeats;
    return out;
}

// Validation-only accessors. All of them return copies, so the returned
// tensors are independent of the context lifetime.
torch::Tensor ctx_csr_values(SPLRBackwardContext& ctx)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    torch::Tensor out = torch::empty({(long)ctx.csr_nnz()}, options);
    cudaMemcpy(out.data_ptr<double>(), ctx.csr_values(),
               ctx.csr_nnz() * sizeof(double), cudaMemcpyDeviceToDevice);
    return out;
}

torch::Tensor ctx_csr_colind(SPLRBackwardContext& ctx)
{
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    torch::Tensor out = torch::empty({(long)ctx.csr_nnz()}, options);
    cudaMemcpy(out.data_ptr<int>(), ctx.csr_colind(),
               ctx.csr_nnz() * sizeof(int), cudaMemcpyDeviceToDevice);
    return out;
}

torch::Tensor ctx_csr_rowptr(SPLRBackwardContext& ctx)
{
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    torch::Tensor out = torch::empty({(long)ctx.problem_size() + 1}, options);
    cudaMemcpy(out.data_ptr<int>(), ctx.csr_rowptr(),
               (ctx.problem_size() + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
    return out;
}

torch::Tensor ctx_secant_s(SPLRBackwardContext& ctx)
{
    if (!ctx.low_rank())
    {
        throw std::runtime_error("SPLRBackwardContext: low-rank correction is disabled, s is unavailable");
    }
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    torch::Tensor out = torch::empty({ctx.problem_size()}, options);
    cudaMemcpy(out.data_ptr<double>(), ctx.secant_s(),
               ctx.problem_size() * sizeof(double), cudaMemcpyDeviceToDevice);
    return out;
}

torch::Tensor ctx_secant_y(SPLRBackwardContext& ctx)
{
    if (!ctx.low_rank())
    {
        throw std::runtime_error("SPLRBackwardContext: low-rank correction is disabled, y is unavailable");
    }
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    torch::Tensor out = torch::empty({ctx.problem_size()}, options);
    cudaMemcpy(out.data_ptr<double>(), ctx.secant_y(),
               ctx.problem_size() * sizeof(double), cudaMemcpyDeviceToDevice);
    return out;
}

torch::Tensor ctx_point(SPLRBackwardContext& ctx)
{
    if (!ctx.has_ctx_point())
    {
        throw std::runtime_error("SPLRBackwardContext: context point unavailable (call with return_validation=True)");
    }
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    torch::Tensor out = torch::empty({ctx.problem_size()}, options);
    cudaMemcpy(out.data_ptr<double>(), ctx.ctx_point(),
               ctx.problem_size() * sizeof(double), cudaMemcpyDeviceToDevice);
    return out;
}

// Register the SPLRBackwardContext class. Called from PYBIND11_MODULE.
void register_backward_context(py::module_& m)
{
    py::class_<SPLRBackwardContext>(m, "SPLRBackwardContext",
        "Opaque backward context of the SPLR solver. Owns the cuDSS "
        "factorization of the last forward iteration's sparsified Hessian "
        "and the rank-2 BFGS materials, and applies B^{-1} as a "
        "preconditioner for the backward PCG.")
        .def("apply_preconditioner", &ctx_apply_preconditioner, py::arg("r"),
            "Compute z = B^{-1} r. r must be a 1D float64 CUDA tensor of length n + m - 1.")
        // Debug/benchmark-only profiling APIs (Stage 3C-2A)
        .def("profile_preconditioner", &ctx_profile_preconditioner, py::arg("r"),
            "Debug/benchmark-only: profiled variant of apply_preconditioner. "
            "Returns a dict with the output tensor and a host/GPU timing breakdown.")
        .def("benchmark_preconditioner_repeated", &ctx_benchmark_preconditioner_repeated,
            py::arg("r"), py::arg("repeats"),
            "Debug/benchmark-only: C++ loop over apply_preconditioner on the "
            "same RHS; returns per-call timing and the last output.")
        // Debug/benchmark-only base-solve profiling APIs (Stage 3C-2B)
        .def("profile_base_solve", &ctx_profile_base_solve, py::arg("r"),
            "Debug/benchmark-only: profiled pure base solve A^{-1} r (no "
            "low-rank correction). Returns a dict with the output tensor "
            "and a host/GPU timing breakdown.")
        .def("benchmark_base_solve_repeated", &ctx_benchmark_base_solve_repeated,
            py::arg("r"), py::arg("repeats"),
            "Debug/benchmark-only: C++ loop over the uninstrumented base-solve "
            "sequence on the same RHS; returns per-call timing and the last "
            "output.")
        .def_property_readonly("reg", &SPLRBackwardContext::reg)
        .def_property_readonly("dim", &SPLRBackwardContext::problem_size)
        .def_property_readonly("low_rank_enabled", &SPLRBackwardContext::low_rank)
        .def_property_readonly("is_fallback", &SPLRBackwardContext::is_fallback)
        .def_property_readonly("adopted_forward_state", &SPLRBackwardContext::adopted_forward_state)
        .def_property_readonly("ys", &SPLRBackwardContext::ys)
        // Validation-only accessors (copies)
        .def("csr_values", &ctx_csr_values)
        .def("csr_colind", &ctx_csr_colind)
        .def("csr_rowptr", &ctx_csr_rowptr)
        .def("secant_s", &ctx_secant_s)
        .def("secant_y", &ctx_secant_y)
        .def("ctx_point", &ctx_point);
}

// PyTorch interface for the backward-capable SPLR solver
py::dict torch_sinkhorn_splr_bwd(
    torch::Tensor M,
    torch::Tensor a,
    torch::Tensor b,
    double reg,
    double tol,
    int max_iter,
    int verbose,
    const py::kwargs& kwargs
)
{
    // Check input dimensions
    if (M.dim() != 2)
    {
        throw std::runtime_error("M must be a 2D tensor");
    }
    if (a.dim() != 1 || b.dim() != 1)
    {
        throw std::runtime_error("a and b must be 1D tensors");
    }

    int n = M.size(0);
    int m = M.size(1);

    if (a.size(0) != n || b.size(0) != m)
    {
        throw std::runtime_error("Shape mismatch: M is [n x m], a should be length n, b should be length m");
    }

    // Ensure that inputs are on GPU with a double type
    // Zero-copy if already satisfying the conditions
    auto src_options = M.options();
    auto src_device = M.device();
    // If M is already on GPU, preserve its device ID
    // Otherwise use the default device
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    if (src_device.is_cuda())
    {
        options = options.device(torch::kCUDA, src_device.index());
    }
    M = M.to(options);
    a = a.to(options);
    b = b.to(options);

    // Set active device
    const at::cuda::CUDAGuard device_guard(options.device());

    // Ensure contiguous and get data pointers
    M = M.contiguous();
    a = a.contiguous();
    b = b.contiguous();
    // Pointing to device memory
    const double* d_M_ptr = M.data_ptr<double>();
    const double* d_a_ptr = a.data_ptr<double>();
    const double* d_b_ptr = b.data_ptr<double>();

    // Handle kwargs for initial values (x0)
    const double* d_x0_ptr = nullptr;
    torch::Tensor x0_storage;

    if (kwargs.contains("x0"))
    {
        torch::Tensor x0 = py::cast<torch::Tensor>(kwargs["x0"]);
        if (x0.dim() != 1 || x0.size(0) != n + m)
        {
            throw std::runtime_error("x0 must be a 1D tensor of length n + m");
        }

        // Shallow copy x0 to x0_storage if the device, type, and layout of x0 are compatible
        // Otherwise make a conversion
        x0_storage = x0.to(options).contiguous();
        d_x0_ptr = x0_storage.data_ptr<double>();
    }

    // Get density_max from kwargs
    // Default to 10 / min(n, m)
    double density_max = 10.0 / std::min(n, m);
    density_max = std::min(density_max, 1.0);
    if (kwargs.contains("density"))
    {
        density_max = py::cast<double>(kwargs["density"]);
        density_max = std::min(density_max, 1.0);
        density_max = std::max(density_max, 0.0);
    }

    // Get shift_max from kwargs
    double shift_max = 0.001;
    if (kwargs.contains("shift"))
    {
        shift_max = py::cast<double>(kwargs["shift"]);
        shift_max = std::max(shift_max, 0.0);
    }

    // Get sparsity_pattern_cycle from kwargs
    int sparsity_pattern_cycle = 30;
    if (kwargs.contains("sparsity_pattern_cycle"))
    {
        sparsity_pattern_cycle = py::cast<int>(kwargs["sparsity_pattern_cycle"]);
    }

    // Get candidate_sinkhorn_iter from kwargs
    int candidate_sinkhorn_iter = 3;
    if (kwargs.contains("candidate_sinkhorn_iter"))
    {
        candidate_sinkhorn_iter = py::cast<int>(kwargs["candidate_sinkhorn_iter"]);
    }

    // Get return_validation from kwargs
    bool return_validation = false;
    if (kwargs.contains("return_validation"))
    {
        return_validation = py::cast<bool>(kwargs["return_validation"]);
    }

    // Get context_mode from kwargs (Stage-3C-1 benchmark/debug only).
    // "adopt" (default): transfer the forward factorization into the context
    // (zero extra factorization). "snapshot": Stage-3B.1 baseline semantics
    // (CSR snapshot + fresh context analyze/factorize) for ablation
    // benchmarking. This only affects the post-loop context export, never
    // the forward iteration sequence.
    std::string context_mode = "adopt";
    if (kwargs.contains("context_mode"))
    {
        context_mode = py::cast<std::string>(kwargs["context_mode"]);
    }
    if (context_mode != "adopt" && context_mode != "snapshot")
    {
        throw std::runtime_error(
            "context_mode must be \"adopt\" or \"snapshot\""
        );
    }
    const bool snapshot_baseline = (context_mode == "snapshot");

    // Create output tensors that are double and on device
    torch::Tensor P = torch::empty({n, m}, options);
    torch::Tensor dual = torch::empty(n + m, options);
    // Pointing to device memory
    double* d_P_ptr = P.data_ptr<double>();
    double* d_dual_ptr = dual.data_ptr<double>();

    // Create the backward context with the same capacity as the forward
    // solver's CSR buffers (Kmax + Hsize)
    const int Hsize = n + m - 1;
    size_t Kmax = static_cast<size_t>(density_max * n * (m - 1));
    Kmax = std::max(Kmax, size_t(1));
    SPLRBackwardContext ctx(Hsize, (int)(Kmax + Hsize));

    // Call CUDA function for the backward-capable SPLR algorithm
    int niter = 0;
    bool converged = false;
    cuda_sinkhorn_splr_bwd(
        d_M_ptr, d_a_ptr, d_b_ptr, d_P_ptr,
        reg, max_iter, tol, n, m, &niter,
        density_max, shift_max,
        sparsity_pattern_cycle, candidate_sinkhorn_iter, verbose,
        d_x0_ptr, d_dual_ptr,
        true, true,
        &ctx, &converged, return_validation, snapshot_baseline
    );

    if (!ctx.is_valid())
    {
        throw std::runtime_error(
            "torch_sinkhorn_splr_bwd: forward pass failed (non-finite "
            "objective value); no backward context available"
        );
    }

    // Convert P and dual when necessary using the input tensor options
    P = P.to(src_options);
    dual = dual.to(src_options);

    // Create result dictionary
    py::dict result;
    result["niter"] = niter;
    result["converged"] = converged;
    result["plan"] = P;
    result["dual"] = dual;
    result["context"] = py::cast(std::move(ctx));

    if (verbose > 0)
    {
        std::cout << "torch_sinkhorn_splr_bwd (CUDA implementation) completed" << std::endl;
        std::cout << "Input shape: [" << n << " x " << m << "]" << std::endl;
        std::cout << "Regularization parameter: " << reg << std::endl;
        std::cout << "Tolerance: " << tol << std::endl;
        std::cout << "Max iterations: " << max_iter << std::endl;
        std::cout << "Actual iterations: " << niter << std::endl;
        if (niter == max_iter)
        {
            std::cout << "Warning: Maximum iterations reached without convergence" << std::endl;
        }
    }

    return result;
}


#endif  // #if defined(TORCH_BUILD)
