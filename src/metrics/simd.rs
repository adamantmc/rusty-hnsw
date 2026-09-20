use std::arch::x86_64::*;

#[target_feature(enable = "avx2,fma")]
pub unsafe fn horizontal_sum_avx2(v: __m256) -> f32 {
    // Extract the "last" 128 bits
    let hi: __m128 = _mm256_extractf128_ps(v, 1);

    // "Re-cast" the remaining "array" ("first" 128 bits) into an 128-bit "array"
    let lo: __m128 = _mm256_castps256_ps128(v);

    // Add the two "array" element-wise
    let sum128: __m128 = _mm_add_ps(hi, lo);

    // Instead of splitting again into "arrays", just add to self
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


#[target_feature(enable = "avx2,fma")]
pub unsafe fn l2_distance_avx2<const ACCUMULATORS: usize>(a: &[f32], b: &[f32]) -> f32{
    // For AVX2, we have 8 lanes
    let lanes: usize = 8;

    // So split in lanes, then add remainder as scalar
    let chunks = a.len() / lanes;

    // Initialize `ACCUMULATORS` arrays of 8 floats, all set to 0 - these will be the accumulators
    let mut accumulator_arrays = [_mm256_setzero_ps(); ACCUMULATORS];

    let chunks_end = chunks - (chunks % ACCUMULATORS);

    for i in (0..chunks_end).step_by(accumulator_arrays.len()) {
        for arr_idx in 0..ACCUMULATORS {
            // For the first accumulator, take the first 8 floats (offset = 0);
            // for the second one, the next 8 (offset = 8), and so on
            let offset: usize = (i + arr_idx) * lanes;

            // Create array out of 8 floats of vector a
            let a_buf = _mm256_loadu_ps(a.as_ptr().add(offset));
            // Create array out of 8 floats of vector b
            let b_buf = _mm256_loadu_ps(b.as_ptr().add(offset));

            // Subtract the two arrays
            // a_buf is [fa1, fa2, fa3, fa4, fa5, fa6, fa7, fa8]
            // b_buf is [fb1, fb2, fb3, fb4, fb5, fb6, fb7, fb8]
            // diff is [fa1-fb1, fa2-fb2, ..., fa8-fb8]
            let diff = _mm256_sub_ps(a_buf, b_buf);

            // Multiply the differences with themselves (= square) and add into acc
            accumulator_arrays[arr_idx] = _mm256_fmadd_ps(diff, diff, accumulator_arrays[arr_idx]);
        }
    }
    let mut sum: f32 = 0.0;

    for i in 0..ACCUMULATORS {
        sum += horizontal_sum_avx2(accumulator_arrays[i]);
    }

    // Manually add in any remainder
    for i in (chunks_end * lanes)..a.len() {
        sum += (a[i] - b[i]).powi(2);
    }

    sum.sqrt()
}

#[inline(always)]
fn l2_distance_chunked<const LANES: usize>(v1: &[f32], v2: &[f32]) -> f32 {
    // Chunk into N chunks of size `LANES`
    let v1_chunks = v1.chunks_exact(LANES);
    let v2_chunks = v2.chunks_exact(LANES);

    // Grab iterators to the remainders
    let (v1_remainder, v2_remainder) = (v1_chunks.remainder(), v2_chunks.remainder());

    // Initialize an array of accumulators
    let mut accumulators = [0.0f32; LANES];

    // Iterate over the chunks in parallel
    for (v1_chunk, v2_chunk) in v1_chunks.zip(v2_chunks) {
        for lane in 0..LANES {
            // Eeach chunk has `LANES` elements, so simply accumulate
            accumulators[lane] += (v1_chunk[lane] - v2_chunk[lane]).powi(2);
        }
    }

    let mut sum: f32 = 0.0;

    // Add all the partial sums
    for v in accumulators {
        sum += v;
    }

    // Add in any remainder
    for (v1_f, v2_f) in v1_remainder.iter().zip(v2_remainder.iter()) {
        sum += (v1_f - v2_f).powi(2);
    }

    // Return
    sum.sqrt()
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn l2_distance_chunked_16_lanes_avx2(a: &[f32], b: &[f32]) -> f32{
    l2_distance_chunked::<16>(a, b)
}

pub fn l2_distance_chunked_16_lanes(a: &[f32], b: &[f32]) -> f32{
    l2_distance_chunked::<16>(a, b)
}

