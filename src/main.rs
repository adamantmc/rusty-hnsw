mod metrics;
mod nn;

use nn::knn::{BruteForceKNN, KNN};
use nn::hnsw::graph::HNSWGraph;
use metrics::{Metric, CosineDistance, L2Distance};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::hash::Hash;
use std::io::{self, BufReader, Read};
use std::time::{Instant};
use chrono::prelude::*;
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

fn read_fvecs(path: &str) -> io::Result<Vec<Vec<f32>>> {
    let mut r = BufReader::new(File::open(path)?);
    let mut vecs = Vec::new();
    let mut dim_buf = [0u8; 4];

    loop {
        // Read the count. A clean EOF here means we're done.
        match r.read_exact(&mut dim_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
        let dim = u32::from_le_bytes(dim_buf) as usize;

        // Read dim * 4 bytes, then reinterpret as f32s.
        let mut bytes = vec![0u8; dim * 4];
        r.read_exact(&mut bytes)?;  // a short read *here* is a truncated file, so it stays an error

        let v: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        vecs.push(v);
    }
    Ok(vecs)
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


fn benchmark<T: Metric>(vectors: Vec<Vec<f32>>, query_vectors: Option<Vec<Vec<f32>>>, mut hnsw_graph: HNSWGraph<T>, mut brute_force_knn: BruteForceKNN<T>, ef_search: usize, recall_levels: &[usize]) -> (HashMap<usize, Vec<f64>>, Vec<f64>, Vec<f64>){
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
        let _ =pb_knn_insert.update(1);
        brute_force_knn.insert(String::from(i.to_string()), vectors[i].clone());
    }
    eprintln!();

    let mut vectors_to_query = if query_vectors.is_none() {vectors.clone()} else {query_vectors.unwrap().clone()};
    vectors_to_query.shuffle(&mut rand::rng());

    let mut sorted_recall_levels = Vec::from(recall_levels);
    sorted_recall_levels.sort();

    let max_recall_level: usize = sorted_recall_levels[sorted_recall_levels.len() - 1] as usize;

    let mut recall_per_query_per_k: HashMap<usize, Vec<f64>> = HashMap::new();

    for val in &sorted_recall_levels {
        recall_per_query_per_k.insert(*val, Vec::new());
    }

    // HNSW search
    let mut pb_hnsw_search = tqdm!(total=vectors_to_query.len());
    let mut hnsw_results: Vec<Vec<(&String, f32)>> = Vec::new();
    let mut hnsw_durations_per_query: Vec<f64> = Vec::new();

    for x in vectors_to_query.iter().enumerate() {
        pb_hnsw_search.set_description("HNSW search");
        let _ = pb_hnsw_search.update(1);

        let start = Instant::now();
        let query_hnsw_results = hnsw_graph.search(&x.1, ef_search, max_recall_level);
        hnsw_durations_per_query.push(start.elapsed().as_secs_f64());
        hnsw_results.push(query_hnsw_results);
    }

    eprintln!();

    // KNN Search
    let mut pb_knn_search = tqdm!(total=vectors_to_query.len());
    let mut knn_results: Vec<Vec<(&String, f32)>> = Vec::new();
    let mut knn_durations_per_query: Vec<f64> = Vec::new();

    for x in vectors_to_query.iter().enumerate() {
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
    #[arg(long, default_value_t = 10000)]
    no_vectors: usize,

    #[arg(long, default_value = None)]
    index_vectors_path: Option<String>,

    #[arg(long, default_value = None)]
    query_vectors_path: Option<String>,

    #[arg(long, default_value_t = 1024)]
    dimensions: usize,

    #[arg(long, default_value_t = 512)]
    ef_search: usize,

    #[arg(long, default_value_t = 512)]
    ef_construction: usize,

    #[arg(long, default_value_t = 16)]
    m: usize,

    #[arg(long, default_values_t = Vec::from([1, 5, 10, 20, 32]))]
    recall: Vec<usize>,

    #[arg(long, default_value = "cosine", value_parser = ["cosine", "euclidean"])]
    distance: String,
}


fn run_benchmark<M: Metric+Copy>(metric: M, args: &Args) {
    let knn: BruteForceKNN<M> = BruteForceKNN::new(metric);
    let hnsw: HNSWGraph<M> = HNSWGraph::new(metric, args.ef_construction, args.m);

    let mut vectors: Vec<Vec<f32>>;
    let mut query_vectors: Option<Vec<Vec<f32>>> = Option::None;

    if args.index_vectors_path.is_some() {
        let path = args.index_vectors_path.clone().unwrap();
        println!("Loading index vectors from {}", path);
        vectors = read_fvecs(&path).unwrap();
        println!("Loaded {} index vectors", vectors.len());
    }
    else {
        vectors = random_vectors(args.dimensions, args.no_vectors);
    }

    if args.query_vectors_path.is_some() {
        let path = args.query_vectors_path.clone().unwrap();
        println!("Loading query vectors from {}", path);
        let out = read_fvecs(&path).unwrap();
        println!("Loaded {} query vectors", out.len());
        query_vectors = Some(out);
    }

    let results = benchmark(vectors, query_vectors, hnsw, knn, 128, &args.recall);

    for recall_level in &args.recall {
        let recall_per_query = results.0.get(&recall_level).unwrap();

        println!("Recall@{}: {:?}", recall_level, stats(recall_per_query, false));
    }

    println!("HNSW runtime {:?}", stats(&results.1, true));
    println!("KNN runtime {:?}", stats(&results.2, true));
}

fn main() {
    let vecs = read_fvecs("/home/swagdam/Downloads/sift/sift_base.fvecs");
    let args = Args::parse();

    println!("{:?}", args);

    let _ = init_logging(LevelFilter::Info);

    if args.distance == "cosine" {
        run_benchmark(CosineDistance {unit_vectors: false}, &args);
    }
    else if args.distance == "euclidean" {
        run_benchmark(L2Distance {}, &args);
    }
}
