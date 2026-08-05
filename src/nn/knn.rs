use std::cmp::min;
use std::collections::HashMap;
use crate::metrics::Metric;

pub trait KNN {
    fn insert(&mut self, id: String, vector: Vec<f32>);
    fn search(&self, vector: &[f32], k: usize) -> Vec<(&String, f32)>;
}

pub struct BruteForceKNN<M: Metric> {metric: M, id_map: HashMap<String, usize>, data: Vec<Vec<f32>>}

impl<M: Metric> BruteForceKNN<M> {
    pub fn new(metric: M) -> Self {
        Self {metric, id_map: HashMap::new(), data: Vec::new()}
    }
}

impl<M: Metric> KNN for BruteForceKNN<M> {
    fn insert(&mut self, id: String, vector: Vec<f32>) {
        let entry = self.id_map.get(&id);
        let vec_id: usize = self.data.len();

        match entry {
            None => {
                self.id_map.insert(id, vec_id);
                self.data.push(vector);
            }
            Some(e) => {
                self.data[*e] = vector;
            }
        }

    }

    fn search(&self, vector: &[f32], k: usize) -> Vec<(&String, f32)> {
        let mut reverse_id_map: HashMap<usize, &String> = HashMap::new();

        for pair in self.id_map.iter() {
            reverse_id_map.insert(*pair.1, pair.0);
        }

        let mut distances: Vec<(&String, f32)> = self.data.iter().enumerate()
            .map(
                |item| (*reverse_id_map.get(&item.0).unwrap(), self.metric.distance(&vector, item.1))
            ).collect();

        distances.sort_by(|a, b| a.1.total_cmp(&b.1));

        distances[0..min(k, distances.len())].to_vec()
    }
}