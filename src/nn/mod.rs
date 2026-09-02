pub(crate) mod knn;
pub(crate) mod dist;
pub(crate) mod hnsw;

pub trait NearestNeighbours {
    type SearchParams: Default + Copy + Sync + Send;

    fn insert(&mut self, id: String, vector: Vec<f32>);
    fn search(&self, vector: &[f32], k: usize, params: Self::SearchParams) -> Vec<(&String, f32)>;
}