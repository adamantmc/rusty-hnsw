use std::cmp::{min, Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fmt::{Display, Formatter};
use std::hash::Hash;
use std::time::{Instant};
use chrono::prelude::*;
use clap::builder::Str;
use log::{Record, Metadata, SetLoggerError, debug, LevelFilter, info};
use rand::seq::SliceRandom;
use clap::Parser;
use kdam::{tqdm, BarExt};

struct SimpleLogger;

impl log::Log for SimpleLogger {
    fn enabled(&self, _: &Metadata) -> bool {
        true
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            let local: DateTime<Local> = Local::now();

            println!("{} - {} - {}", record.level(), local.format("%Y-%m-%d %H:%M:%S").to_string(), record.args());
        }
    }

    fn flush(&self) {}
}

static LOGGER: SimpleLogger = SimpleLogger;

pub fn init_logging(level: LevelFilter) -> Result<(), SetLoggerError> {
    log::set_logger(&LOGGER).map(|()| log::set_max_level(level))
}

pub trait Metric {
    fn distance(&self, v1: &[f32], v2: &[f32]) -> f32;
}

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


fn cosine_similarity(v1: &[f32], v2: &[f32], unit_vectors: bool) -> f32 {
    let mut v1_magnitude = 1.0;
    let mut v2_magnitude = 1.0;

    if !unit_vectors {
        v1_magnitude = magnitude(&v1);
        v2_magnitude = magnitude(&v2);
    }

    let cos_sim = dot_product(&v1, &v2) / (v1_magnitude * v2_magnitude);

    (cos_sim + 1.0) / 2.0
}


#[derive(Debug)]
struct L2Distance;
#[derive(Debug)]
struct CosineDistance {unit_vectors: bool}

impl Metric for L2Distance {
    fn distance(&self, v1: &[f32], v2: &[f32]) -> f32 {
        let mut sum: f32 = 0.0;

        for (a,b) in v1.iter().zip(v2.iter()) {
            sum += (a - b).powi(2)
        }

        sum.sqrt()
    }
}

impl Metric for CosineDistance {
    fn distance(&self, v1: &[f32], v2: &[f32]) -> f32 {

        1.0 - cosine_similarity(&v1, &v2, self.unit_vectors)
    }
}

trait KNN {
    fn insert(&mut self, id: String, vector: Vec<f32>);
    fn search(&self, vector: &[f32], k: usize) -> Vec<(&String, f32)>;
}

struct BruteForceKNN<M: Metric> {metric: M, id_map: HashMap<String, usize>, data: Vec<Vec<f32>>}

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


#[derive(Debug)]
struct Node {layer: u8, edges_per_layer: Vec<Edges>}

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

#[derive(Copy, Clone, PartialEq)]
struct Dist(f32);

impl Display for Dist {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Eq for Dist {}

impl Ord for Dist {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl PartialOrd<Self> for Dist {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
#[derive(Debug)]
struct HNSWGraph<M: Metric> {
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

fn non_zero_rand() -> f64 {
    let mut v: f64 = rand::random();

    while v == 0.0 {
        v = rand::random();
    }

    v
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

    fn search(&self, q: &[f32], ef_search: usize, k: usize) -> Vec<(&String, f32)> {
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


fn random_vectors(dims: usize, length: usize) -> Vec<Vec<f32>> {
    let mut vecs: Vec<Vec<f32>> = Vec::with_capacity(length);

    for _ in 0..length {
        let mut random_vec: Vec<f32> = Vec::with_capacity(dims);
        for _ in 0..dims {
            random_vec.push(rand::random());
        }

        vecs.push(random_vec);
    }

    vecs
}

fn recall<T: Hash + Eq>(retrieved: &HashSet<T>, gold: &HashSet<T>) -> f64 {
    retrieved.intersection(gold).count() as f64 / gold.len() as f64
}

#[derive(Debug)]
struct Stats {mean: f64, standard_deviation: f64, ci_95_low: f64, ci_95_high: f64, p95: f64, p99: f64}


fn percentile<T>(numbers: &[T], percentile: f64) -> &T {
    let idx: usize = (numbers.len() as f64 * percentile).ceil() as usize;

    &numbers[idx - 1]
}

fn stats(numbers: &[f64], ascending: bool) -> Stats {
    let sum: f64 = numbers.iter().sum();
    let mean: f64 = sum / numbers.len() as f64;
    let standard_deviation: f64 = (numbers.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / numbers.len() as f64).sqrt();
    let standard_error = standard_deviation / (numbers.len() as f64).sqrt();
    let error_margin: f64 = 1.96 * standard_error;

    let mut sorted_values = Vec::from(numbers);
    sorted_values.sort_by(|a, b| if ascending {a.total_cmp(b)} else {a.total_cmp(b).reverse()});

    let p95 = percentile(&sorted_values, 0.95);
    let p99 = percentile(&sorted_values, 0.99);

    Stats{mean, standard_deviation, ci_95_low: mean - error_margin, ci_95_high: mean + error_margin, p95: *p95, p99: *p99}
}


fn calculate_recall(retrieved: &[&String], golden: &[&String], recall_levels: &Vec<usize>) -> Vec<f64> {
    let mut recall_values: Vec<f64> = Vec::new();

    for level in recall_levels {
        let s1: HashSet<&String> = HashSet::from_iter(retrieved[0..*level].iter().map(|x| *x));
        let s2: HashSet<&String> = HashSet::from_iter(golden[0..*level].iter().map(|x| *x));

        recall_values.push(recall(&s1, &s2));
    }

    recall_values

}


fn benchmark<T: Metric>(vectors: Vec<Vec<f32>>, mut hnsw_graph: HNSWGraph<T>, mut brute_force_knn: BruteForceKNN<T>, ef_search: usize, recall_levels: &[usize]) -> (HashMap<usize, Vec<f64>>, Vec<f64>, Vec<f64>){
    // HNSW insert
    let mut pb_hnsw_insert = tqdm!(total=vectors.len());
    for i in 0..vectors.len() {
        pb_hnsw_insert.set_description("HNSW insertion");
        let _ =pb_hnsw_insert.update(1);
        hnsw_graph.insert(String::from(i.to_string()), vectors[i].clone(), None);
    }
    eprintln!();

    // KNN insert
    let mut pb_knn_insert = tqdm!(total=vectors.len());
    for i in 0..vectors.len() {
        pb_knn_insert.set_description("KNN insertion");
        let _ =pb_knn_insert.update(1);        brute_force_knn.insert(String::from(i.to_string()), vectors[i].clone());
    }
    eprintln!();


    let mut cloned_vectors = vectors.clone();
    cloned_vectors.shuffle(&mut rand::rng());

    let mut sorted_recall_levels = Vec::from(recall_levels);
    sorted_recall_levels.sort();

    let max_recall_level: usize = sorted_recall_levels[sorted_recall_levels.len() - 1] as usize;

    let mut recall_per_query_per_k: HashMap<usize, Vec<f64>> = HashMap::new();

    for val in &sorted_recall_levels {
        recall_per_query_per_k.insert(*val, Vec::new());
    }

    // HNSW search
    let mut pb_hnsw_search = tqdm!(total=cloned_vectors.len());
    let mut hnsw_results: Vec<Vec<(&String, f32)>> = Vec::new();
    let mut hnsw_durations_per_query: Vec<f64> = Vec::new();

    for x in cloned_vectors.iter().enumerate() {
        pb_hnsw_search.set_description("HNSW search");
        let _ = pb_hnsw_search.update(1);

        let start = Instant::now();
        let query_hnsw_results = hnsw_graph.search(&x.1, ef_search, max_recall_level);
        hnsw_durations_per_query.push(start.elapsed().as_secs_f64());
        hnsw_results.push(query_hnsw_results);
    }

    eprintln!();

    // KNN Search
    let mut pb_knn_search = tqdm!(total=cloned_vectors.len());
    let mut knn_results: Vec<Vec<(&String, f32)>> = Vec::new();
    let mut knn_durations_per_query: Vec<f64> = Vec::new();

    for x in cloned_vectors.iter().enumerate() {
        pb_knn_search.set_description("KNN search");
        let _ = pb_knn_search.update(1);

        let start = Instant::now();
        let query_knn_results = brute_force_knn.search(&x.1, max_recall_level);
        knn_durations_per_query.push(start.elapsed().as_secs_f64());
        knn_results.push(query_knn_results);
    }
    
    eprintln!();

    for (hnsw_results, knn_results) in hnsw_results.iter().zip(knn_results) {
        let recall = calculate_recall(
            &hnsw_results.iter().map(|x| x.0).collect::<Vec<_>>(),
            &knn_results.iter().map(|x| x.0).collect::<Vec<_>>(),
            &sorted_recall_levels
        );

        for recall_value in recall.iter().enumerate() {
            recall_per_query_per_k.get_mut(&recall_levels[recall_value.0]).unwrap().push(*recall_value.1);
        }
    }

    (recall_per_query_per_k, hnsw_durations_per_query, knn_durations_per_query)
}


#[derive(Parser, Debug)]
#[command(version, about, long_about=None)]
struct Args {
    #[arg(short, long, default_value_t = 10000)]
    no_vectors: usize,

    #[arg(short, long, default_value_t = 1024)]
    dimensions: usize,

    #[arg(short, long, default_value_t = 512)]
    ef_search: usize,

    #[arg(short, long, default_value_t = 512)]
    ef_construction: usize,

    #[arg(short, long, default_value_t = 16)]
    m: usize,

    #[arg(short, long, default_values_t = Vec::from([1, 5, 10, 20, 32]))]
    recall: Vec<usize>,
}


fn main() {
    let args = Args::parse();

    println!("{:?}", args);

    let _ = init_logging(LevelFilter::Info);

    let knn: BruteForceKNN<CosineDistance>= BruteForceKNN::new(CosineDistance{unit_vectors: false });
    let hnsw: HNSWGraph<CosineDistance> = HNSWGraph::new(CosineDistance{unit_vectors: false}, args.ef_construction, args.m);

    let vectors = random_vectors(args.dimensions, args.no_vectors);
    let results = benchmark(vectors, hnsw, knn, 128, &args.recall);

    for recall_level in args.recall {
        let recall_per_query = results.0.get(&recall_level).unwrap();

        println!("Recall@{}: {:?}", recall_level, stats(recall_per_query, false));
    }

    println!("HNSW runtime {:?}", stats(&results.1, true));
    println!("KNN runtime {:?}", stats(&results.2, true));
}
