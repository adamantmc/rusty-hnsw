fn magnitude(v: &[f32]) -> f32 {
    let mut m: f32 = 0.0;

    for v in v.iter() {
        m += v.powi(2);
    }

    m.sqrt()
}

fn dot_product(v1: &[f32], v2: &[f32]) -> f32 {
    let mut dp: f32 = 0.0;

    for (a, b) in v1.iter().zip(v2.iter()) {
        dp += a * b
    }

    dp
}

pub fn cosine_similarity(v1: &[f32], v2: &[f32], unit_vectors: bool) -> f32 {
    let mut v1_magnitude = 1.0;
    let mut v2_magnitude = 1.0;

    if !unit_vectors {
        v1_magnitude = magnitude(&v1);
        v2_magnitude = magnitude(&v2);
    }

    let cos_sim = dot_product(&v1, &v2) / (v1_magnitude * v2_magnitude);

    (cos_sim + 1.0) / 2.0
}

pub fn l2_distance(v1: &[f32], v2: &[f32]) -> f32 {
    let mut sum: f32 = 0.0;

    for (a,b) in v1.iter().zip(v2.iter()) {
        sum += (a - b).powi(2)
    }

    sum.sqrt()
}