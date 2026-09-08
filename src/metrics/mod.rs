use crate::metrics::calc::magnitude;

mod calc;
mod simd;

#[derive(Debug, Copy, Clone)]
pub struct L2Distance;
#[derive(Debug, Copy, Clone)]
pub struct CosineDistance {pub unit_vectors: bool}

pub trait Metric {
    fn distance(&self, v1: &[f32], v2: &[f32]) -> f32;
}

impl Metric for L2Distance {
    fn distance(&self, v1: &[f32], v2: &[f32]) -> f32 {
        calc::l2_distance(v1, v2)
    }
}

impl Metric for CosineDistance {
    fn distance(&self, v1: &[f32], v2: &[f32]) -> f32 {
        1.0 - calc::cosine_similarity(&v1, &v2, self.unit_vectors)
    }
}

pub fn normalize_vector(v: &[f32]) -> Vec<f32> {
    let mag = magnitude(v);

    v.iter().map(|x| *x / mag).collect()
}