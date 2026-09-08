use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
pub unsafe fn horizontal_sum_avx2(v: __m256) -> f32 {
    // Extract the "last" 128 bits
    let hi: __m128 = _mm256_extractf128_ps(v, 1);

    // "Re-cast" the remaining "vector" ("first" 128 bits) into an 128-bit "vector"
    let lo: __m128 = _mm256_castps256_ps128(v);

    // Add the two "vectors" element-wise
    let sum128: __m128 = _mm_add_ps(hi, lo);

    // Instead of splitting again into "vectors", just add to self
    // We have 4 values (4*32 = 128), a, b, c, d
    // This produces [a + b, c + d, a + b, c + d]
    let sum64: __m128 = _mm_hadd_ps(sum128, sum128);

    // Again
    // Given [a + b, c + d, a + b, c + d] this produces
    // [a + b + c + d, a + b + c + d, a + b + c + d, a + b + c + d]
    let sum32: __m128 = _mm_hadd_ps(sum64, sum64);

    // Get first 32 bits (= a single f32) - essentially the sum
    _mm_cvtss_f32(sum32)
}


#[target_feature(enable = "avx2")]
pub unsafe fn l2_squared_avx2(a: &[f32], b: &[f32]) -> f32{
    // For AVX2, we have 8 lanes
    let lanes: usize = 8;

    // So split in lanes, then add remainder as scalar
    let chunks = a.len() / lanes;

    let mut acc = _mm256_setzero_ps();

    for i in 0..chunks {
        let offset: usize = i * lanes;

        let a_buf = _mm256_loadu_ps(a.as_ptr().add(offset));
        let b_buf = _mm256_loadu_ps(b.as_ptr().add(offset));

        let diff = _mm256_sub_ps(a_buf, b_buf);

        acc = _mm256_fmadd_ps(diff, diff, acc);

    }
    let mut sum = horizontal_sum_avx2(acc);

    for i in (chunks * lanes)..lanes {
        sum += (a[i] - b[i]).powi(2);
    }

    sum
}

