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

