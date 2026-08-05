use std::cmp::{min, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};
use log::debug;
use crate::metrics::Metric;
use crate::nn::dist::Dist;

fn non_zero_rand() -> f64 {
    let mut v: f64 = rand::random();

    while v == 0.0 {
        v = rand::random();
    }

    v
}

#[derive(Debug)]
pub struct Node {layer: u8, edges_per_layer: Vec<Edges>}

impl Node {
    pub fn new(layer: u8) -> Self {
        let mut edges_per_layer: Vec<Edges> = Vec::new();
        for _ in 0..=layer {
            edges_per_layer.push(Edges::new())
        }
        Self {layer, edges_per_layer }
    }

    pub fn neighbours(&self, layer: u8) -> Option<&Vec<usize>> {
        if layer > self.layer {
            return None;
        }

        Some(&self.edges_per_layer[layer as usize].ids)
    }

    pub fn distances(&self, layer: u8) -> Option<&Vec<f32>> {
        if layer > self.layer {
            return None;
        }

        Some(&self.edges_per_layer[layer as usize].distances)
    }

    pub fn add_edge(&mut self, id: usize, distance: f32, layer: u8) {
        self.edges_per_layer.get_mut(layer as usize).unwrap().add(id, distance);
    }

    pub fn remove_edge(&mut self, id: usize, layer: u8) {
        self.edges_per_layer.get_mut(layer as usize).unwrap().remove(id);
    }

}

#[derive(Debug)]
struct Edges {ids: Vec<usize>, distances: Vec<f32>}

impl Edges {
    pub fn new() -> Self {
        Self {ids: Vec::new(), distances: Vec::new()}
    }

    pub fn add(&mut self, id: usize, distance: f32) {
        if !self.ids.contains(&id) {
            self.ids.push(id);
            self.distances.push(distance);
        }
    }

    pub fn remove(&mut self, id: usize) {
        let pos = self.ids.iter().position(|v| *v == id);

        match pos {
            None => {}
            Some(idx) => {
                self.ids.remove(idx);
                self.distances.remove(idx);
            }
        }

    }
}

#[derive(Debug)]
pub(crate) struct HNSWGraph<M: Metric> {
    metric: M,
    id_map: HashMap<String, usize>,
    reverse_id_map: HashMap<usize, String>,
    vectors: Vec<Vec<f32>>,
    nodes: Vec<Node>,
    layers: u8,
    entry_point: Option<usize>,
    m: usize,
    ef_construction: usize,
    m_max_0: usize,
    m_max: usize,
    ml: f64
}

impl<M: Metric> HNSWGraph<M> {
    pub fn new(metric: M, ef_construction: usize, m: usize) -> Self {
        HNSWGraph {
            metric,
            id_map: HashMap::new(),
            reverse_id_map: HashMap::new(),
            vectors: Vec::new(),
            nodes: Vec::new(),
            layers: 0,
            entry_point: None,
            m: m,
            ef_construction: ef_construction,
            m_max_0: 2*m,
            m_max: m,
            ml: 1.0/(m as f64).ln(),
        }
    }

    pub fn insert(&mut self, id: String, vector: Vec<f32>, landing_layer: Option<u8>) {
        let l = match landing_layer {
            Some(v) => {v},
            None => {(-non_zero_rand().ln()*self.ml).floor() as u8}
        };

        let node_id = self.nodes.len();

        debug!("Inserting node {} with assigned id {}, landing at layer {}", id, node_id, l);

        let top_layer = if self.layers != 0 {self.layers - 1} else {0};

        let mut entry_points: Vec<usize> = match self.entry_point {
            None => {
                Vec::new()
            }
            Some(v) => {
                Vec::from([v])
            }
        };
        let id_clone = id.clone();
        self.id_map.insert(id, node_id);
        self.reverse_id_map.insert(node_id, id_clone);

        let node = Node::new(l);

        self.nodes.push(node);
        self.vectors.push(vector);


        // Descend from top layer to just above the one we landed at, getting a single result
        // ("greedy search") and keeping it as the entrypoint for each next layer
        for layer in (l + 1..=top_layer).rev() {

            let search_results = self.search_layer(
                &self.vectors.last().unwrap(), &entry_points, 1, layer
            );


            entry_points = Vec::from([(*search_results.first().unwrap()).1]);

        }

        // Descend from landing layer to 0, starting at the entrypoint of the previous layer
        for layer in (0..=min(l, top_layer)).rev() {
            if entry_points.len() == 0 {
                debug!("No entrypoints to search for node {} at layer {} - breaking", node_id, layer);
                break
            }

            // Search for `ef_construction` closest vectors, using the entrypoints found previously
            let search_results = self.search_layer(
                &self.vectors.last().unwrap(), &entry_points, self.ef_construction, layer
            );

            let neighbours = self.select_neighbours_simple(&search_results, self.m);

            // Add bi-directional edges for closest vectors
            for tuple in &neighbours {
                self.nodes.get_mut(node_id).unwrap().add_edge(tuple.1, tuple.0, layer);
                self.nodes.get_mut(tuple.1).unwrap().add_edge(node_id, tuple.0, layer);
            }

            for tuple in &neighbours {
                // Check if any neighbour has exceeded m_max - if so, trim down to m_max by
                // calling select_neighbours and removing edges
                // TODO: this seems slow
                let e_conn_option = self.nodes[tuple.1].neighbours(layer);
                let e_conn_distances_option = self.nodes[tuple.1].distances(layer);

                let m_max = if layer == 0 {self.m_max_0} else {self.m_max};

                match e_conn_option {
                    None => {}
                    Some(e_conn) => {
                        let e_conn_distances = e_conn_distances_option.unwrap();
                        if e_conn.len() > m_max {
                            debug!("Pruning the neighbours of node {}", tuple.1);

                            let mut e_conn_tuples: Vec<(f32, usize)> = e_conn.iter()
                                .zip(e_conn_distances.iter())
                                .map(|(a, b)| (*b, *a)).collect();

                            e_conn_tuples.sort_by(|a, b| a.0.total_cmp(&b.0));

                            let new_e_conn = self.select_neighbours_simple(&e_conn_tuples, m_max);

                            let s_old: HashSet<usize> = e_conn.into_iter().map(|v| *v).collect();
                            let s_new: HashSet<usize> = HashSet::from_iter(new_e_conn.iter().map(|tup| tup.1));

                            let diff = s_old.difference(&s_new);

                            for v in diff {
                                self.nodes[tuple.1].remove_edge(*v, layer);
                                self.nodes[*v].remove_edge(tuple.1, layer);
                            }
                        }
                    }
                }

            }
            entry_points = search_results.iter().map(|(_, b)| *b).collect();
        }
        if l > top_layer || self.layers == 0 {
            self.layers = l + 1;
            self.entry_point = Some(node_id);
            debug!("Set self.layers to {} and self.entry_point to {}", self.layers, node_id)
        }
    }

    fn search_layer(&self, q: &[f32], entry_points: &[usize], ef_search: usize, layer: u8) -> Vec<(f32, usize)> {
        let mut visited: HashSet<usize> = HashSet::from_iter(entry_points.iter().map(|v| *v));

        let ep_distances: Vec<(Dist, usize)> = visited.iter().map(
            |a| (Dist(self.metric.distance(q, &self.vectors[*a])), *a)
        ).collect();

        // BinaryHeap is a max heap. We want to find the nearest element to Q from the candidate
        // list, so we reverse this one
        let mut candidates: BinaryHeap<Reverse<(Dist, usize)>> = BinaryHeap::from_iter(
            ep_distances.iter().map(|a| Reverse(*a))
        );

        // We want to remove the farthest, so we keep it as a max heap, to get the element with the
        // maximum distance to q
        let mut results: BinaryHeap<(Dist, usize)> = BinaryHeap::from(ep_distances.clone());

        while candidates.len() > 0 {
            let candidate = candidates.pop().unwrap().0;
            let mut farthest = results.peek().unwrap();

            // Stopping condition - if the closest candidate is farther than the farthest result, stop
            if candidate.0 > farthest.0 {
                break
            }
            let candidate_neighbours_ref = self.nodes[candidate.1].neighbours(layer);
            match candidate_neighbours_ref {
                None => {}
                Some(candidate_neighbours) => {
                    debug!("Candidate {} has {} neighbours in layer {}", candidate.1, candidate_neighbours.len(), layer);
                    for neighbour in candidate_neighbours {
                        if visited.contains(&neighbour) {
                            continue
                        }

                        visited.insert(*neighbour);

                        // TODO: why twice?
                        farthest = results.peek().unwrap();

                        let distance = self.metric.distance(q, &self.vectors[*neighbour]);

                        // If the neighbour is closer than the farthest result, add it to the candidates and results
                        // Ignore distance if we have less than `ef_search` results
                        if farthest.0.0 > distance || results.len() < ef_search {
                            candidates.push(Reverse((Dist(distance), *neighbour)));
                            results.push((Dist(distance), *neighbour));

                            // If we have more than `ef_search` results, pop the farthest one
                            if results.len() > ef_search {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        let mut out = Vec::from_iter(results.iter().cloned().map(|a| (a.0, a.1)));

        out.sort();

        out.iter().map(|a| (a.0.0, a.1)).collect()

    }

    pub fn search(&self, q: &[f32], ef_search: usize, k: usize) -> Vec<(&String, f32)> {
        let top_layer = self.layers - 1;
        let mut entry_point: usize;

        match self.entry_point {
            None => {return Vec::new()}
            Some(v) => {entry_point = v;}
        }
        for layer in (1..=top_layer).rev() {
            debug!("Greedy search (ef=1) on layer {} with entry point {}", layer, entry_point);
            let search_results = self.search_layer(q,&[entry_point], 1, layer);
            entry_point = search_results.first().unwrap().1;
        }

        debug!("Search on layer 0 with entry point {}", entry_point);

        let out = self.search_layer(&q, &[entry_point], ef_search, 0);

        out.iter().take(k).map(|item| (self.reverse_id_map.get(&item.1).unwrap(), item.0)).collect()
    }

    fn select_neighbours_simple(&self, candidates: &[(f32, usize)], m: usize) -> Vec<(f32, usize)>{
        candidates[0..min(m, candidates.len())].to_vec()
    }
}
