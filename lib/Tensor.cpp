#include "Tensor.h"

void gemm(const Tensor &A, bool tA, const Tensor &B, bool tB, Tensor &C,
          bool accumulate) {
  const size_t M = tA ? A.s.c : A.s.r;
  const size_t K = tA ? A.s.r : A.s.c;
  const size_t Kb = tB ? B.s.c : B.s.r;
  const size_t N = tB ? B.s.r : B.s.c;
  if (K != Kb || C.s.r != M || C.s.c != N)
    throw std::invalid_argument("gemm: shape mismatch");
  if (!accumulate) C.zero();
  // i-k-j order: the inner loop walks C and (non-transposed) B contiguously.
  for (size_t i = 0; i < M; i++) {
    for (size_t k = 0; k < K; k++) {
      const Scalar a = tA ? A(k, i) : A(i, k);
      if (!tB) {
        const Scalar *brow = &B.v[k * B.s.c];
        Scalar *crow = &C.v[i * C.s.c];
        for (size_t j = 0; j < N; j++) crow[j] += a * brow[j];
      } else {
        for (size_t j = 0; j < N; j++) C(i, j) += a * B(j, k);
      }
    }
  }
}