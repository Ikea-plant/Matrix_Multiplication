#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <immintrin.h> // Required for AVX2 intrinsics

// 1. Naive Matrix Multiplication (O(N^3))
void matrixMultiplyNaive(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// 2. Optimized Matrix Multiplication (Cache Blocking + AVX2 SIMD + FMA)
void matrixMultiplyOptimized(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int N) {
    // Initialize result matrix to zero
    std::fill(C.begin(), C.end(), 0.0f);

    const int BLOCK_SIZE = 64; // Tuned block size for L1/L2 cache efficiency

    // Outer cache-blocking loops
    for (int sj = 0; sj < N; sj += BLOCK_SIZE) {
        for (int sk = 0; sk < N; sk += BLOCK_SIZE) {
            for (int si = 0; si < N; si += BLOCK_SIZE) {

                int imax = std::min(si + BLOCK_SIZE, N);
                int kmax = std::min(sk + BLOCK_SIZE, N);
                int jmax = std::min(sj + BLOCK_SIZE, N);

                // Inner computational kernel
                for (int i = si; i < imax; ++i) {
                    for (int k = sk; k < kmax; ++k) {
                        // Broadcast scalar A[i, k] across an 8-float AVX register
                        __m256 a_vec = _mm256_set1_ps(A[i * N + k]);

                        int j = sj;
                        // Vectorized loop processing 8 floats per instruction using FMA
                        for (; j <= jmax - 8; j += 8) {
                            __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                            __m256 c_vec = _mm256_loadu_ps(&C[i * N + j]);

                            // C = (A * B) + C fused into a single instruction
                            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);

                            _mm256_storeu_ps(&C[i * N + j], c_vec);
                        }

                        // Handle remaining elements if block size isn't a multiple of 8
                        for (; j < jmax; ++j) {
                            C[i * N + j] += A[i * N + k] * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

// Verification and Benchmarking Harness
int main() {
    const int N = 512; // Matrix dimensions (N x N)
    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> C_naive(N * N, 0.0f);
    std::vector<float> C_optimized(N * N, 0.0f);

    // Initialize matrices with random values
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N * N; ++i) {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }

    std::cout << "Running Benchmark with Matrix Size: " << N << "x" << N << "\n\n";

    // Benchmark Naive Implementation
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplyNaive(A, B, C_naive, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> naive_duration = end - start;
    std::cout << "Naive Implementation Time:    " << naive_duration.count() << " ms\n";

    // Benchmark Optimized Implementation
    start = std::chrono::high_resolution_clock::now();
    matrixMultiplyOptimized(A, B, C_optimized, N);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> opt_duration = end - start;
    std::cout << "Optimized Implementation Time: " << opt_duration.count() << " ms\n";

    // Calculate Speedup
    double speedup = naive_duration.count() / opt_duration.count();
    std::cout << "\nSpeedup: " << speedup << "x faster\n";

    // Validate correctness
    bool correct = true;
    for (int i = 0; i < N * N; ++i) {
        if (std::abs(C_naive[i] - C_optimized[i]) > 1e-3) {
            correct = false;
            std::cout << "Mismatch at index " << i << ": Naive = " << C_naive[i] << ", Opt = " << C_optimized[i] << "\n";
            break;
        }
    }
    std::cout << "Verification: " << (correct ? "PASSED" : "FAILED") << "\n";

    return 0;
}