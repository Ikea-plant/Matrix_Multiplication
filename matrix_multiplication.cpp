#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <arm_neon.h> // Required for ARM NEON intrinsics on Apple Silicon

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

// 2. Optimized Matrix Multiplication for Apple Silicon (Cache Blocking + ARM NEON SIMD)
void matrixMultiplyOptimized(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int N) {
    std::fill(C.begin(), C.end(), 0.0f);

    const int BLOCK_SIZE = 64; // Tuned block size for L1/L2 cache efficiency

    for (int sj = 0; sj < N; sj += BLOCK_SIZE) {
        for (int sk = 0; sk < N; sk += BLOCK_SIZE) {
            for (int si = 0; si < N; si += BLOCK_SIZE) {

                int imax = std::min(si + BLOCK_SIZE, N);
                int kmax = std::min(sk + BLOCK_SIZE, N);
                int jmax = std::min(sj + BLOCK_SIZE, N);

                for (int i = si; i < imax; ++i) {
                    for (int k = sk; k < kmax; ++k) {
                        // Broadcast scalar A[i, k] across a 4-float NEON register
                        float32x4_t a_vec = vdupq_n_f32(A[i * N + k]);

                        int j = sj;
                        // Vectorized loop processing 4 floats per instruction using NEON FMA
                        for (; j <= jmax - 4; j += 4) {
                            float32x4_t b_vec = vld1q_f32(&B[k * N + j]);
                            float32x4_t c_vec = vld1q_f32(&C[i * N + j]);

                            // C = (A * B) + C fused into a single instruction
                            c_vec = vmlaq_f32(c_vec, a_vec, b_vec);

                            vst1q_f32(&C[i * N + j], c_vec);
                        }

                        // Handle remaining elements
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
    const int N = 512;
    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> C_naive(N * N, 0.0f);
    std::vector<float> C_optimized(N * N, 0.0f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N * N; ++i) {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }

    std::cout << "Running Benchmark on Apple Silicon with Matrix Size: " << N << "x" << N << "\n\n";

    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplyNaive(A, B, C_naive, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> naive_duration = end - start;
    std::cout << "Naive Implementation Time:    " << naive_duration.count() << " ms\n";

    start = std::chrono::high_resolution_clock::now();
    matrixMultiplyOptimized(A, B, C_optimized, N);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> opt_duration = end - start;
    std::cout << "Optimized Implementation Time: " << opt_duration.count() << " ms\n";

    double speedup = naive_duration.count() / opt_duration.count();
    std::cout << "\nSpeedup: " << speedup << "x faster\n";

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