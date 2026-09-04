mod metrics;
mod nn;

use std::cmp::min;
use nn::knn::{KNN};
use nn::NearestNeighbours;
use nn::hnsw::graph::HNSWGraph;
use metrics::{Metric, CosineDistance, L2Distance};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::hash::Hash;
use std::io::{self, BufReader, Read};
use std::sync::mpsc;
use std::thread;
use std::time::{Instant};
use chrono::prelude::*;
use log::{Record, Metadata, SetLoggerError, LevelFilter};
use rand::seq::SliceRandom;
use clap::Parser;
use kdam::{tqdm, BarExt};
use crate::nn::hnsw::graph::{HNSWSearchParams};

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
        let s1: HashSet<&String> = HashSet::from_iter(retrieved[0..min(*level, retrieved.len())].iter().map(|x| *x));
        let s2: HashSet<&String> = HashSet::from_iter(golden[0..*level].iter().map(|x| *x));

        recall_values.push(recall(&s1, &s2));
    }

    recall_values

}


fn threaded_nn_search<'a, T: NearestNeighbours + Send + Sync>(
    nn: &'a T,
    vectors_to_query: &[(usize, &Vec<f32>)],
    p: T::SearchParams,
    k: usize,
    no_search_threads: i32,
    pb_prefix: &str
) -> (Vec<(usize, Vec<(&'a String, f32)>)>, Vec<f64>) {
    let thread_chunk_size = (vectors_to_query.len() as f32 / no_search_threads as f32).ceil() as usize;
    let mut pb = tqdm!(total=vectors_to_query.len());
    let mut results: Vec<(usize, Vec<(&String, f32)>)> = Vec::new();
    let mut durations_per_query: Vec<f64> = Vec::new();

    let (tx, rx) = mpsc::channel::<(usize, Vec<(&String, f32)>)/* Type */>();
    let (dur_tx, dur_rx) = mpsc::channel::<f64>();

    thread::scope(|sc| {
        let mut vector_chunks = vectors_to_query.chunks(thread_chunk_size);

        for _ in 0..no_search_threads {
            let tx_clone = tx.clone();
            let dur_tx_clone = dur_tx.clone();

            let chunk = vector_chunks.next().unwrap();
            sc.spawn(move || {
                for x in chunk {
                    let start = Instant::now();
                    let query_results = nn.search(&x.1, k, p);
                    tx_clone.send((x.0, query_results)).unwrap();
                    dur_tx_clone.send(start.elapsed().as_secs_f64()).unwrap();
                }
                drop(tx_clone);
                drop(dur_tx_clone);
            });
        }

        drop(tx);
        drop(dur_tx);

        for (result, duration) in rx.iter().zip(dur_rx.iter()) {
            pb.set_description(pb_prefix);
            let _ = pb.update(1);
            results.push(result);
            durations_per_query.push(duration);
        }
    });

    (results, durations_per_query)
}


fn benchmark<T: Metric + Send + Sync >(vectors: &[Vec<f32>], query_vectors: &[Vec<f32>], mut hnsw_graph: HNSWGraph<T>, mut brute_force_knn: KNN<T>, ef_search: usize, recall_levels: &[usize], no_search_threads: i32) -> (HashMap<usize, Vec<f64>>, Vec<f64>, Vec<f64>){
    // HNSW insert
    let mut pb_hnsw_insert = tqdm!(total=vectors.len());
    for i in 0..vectors.len() {
        pb_hnsw_insert.set_description("HNSW insertion");
        let _ =pb_hnsw_insert.update(1);
        hnsw_graph.insert(String::from(i.to_string()), vectors[i].clone());
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

    let vectors_to_query: Vec<(usize, &Vec<f32>)> = query_vectors.iter().enumerate().collect();

    let mut sorted_recall_levels = Vec::from(recall_levels);
    sorted_recall_levels.sort();

    let max_recall_level: usize = sorted_recall_levels[sorted_recall_levels.len() - 1] as usize;

    let mut recall_per_query_per_k: HashMap<usize, Vec<f64>> = HashMap::new();

    for val in &sorted_recall_levels {
        recall_per_query_per_k.insert(*val, Vec::new());
    }

    // HNSW search
    let hnsw_out = threaded_nn_search(
        &hnsw_graph,
        &vectors_to_query,
        HNSWSearchParams{ef_search: ef_search},
        max_recall_level,
        no_search_threads,
        "HNSW Search"
    );
    let mut hnsw_results = hnsw_out.0;
    let hnsw_durations_per_query: Vec<f64> = hnsw_out.1;

    eprintln!();

    // KNN Search
    let knn_out = threaded_nn_search(
        &brute_force_knn,
        &vectors_to_query,
        (),
        max_recall_level,
        no_search_threads,
        "KNN Search"
    );
    let mut knn_results = knn_out.0;
    let knn_durations_per_query: Vec<f64> = knn_out.1;
    eprintln!();

    hnsw_results.sort_by(|a, b| a.0.cmp(&b.0));
    knn_results.sort_by(|a, b| a.0.cmp(&b.0));

    for (hnsw_result, knn_result) in hnsw_results.iter().zip(knn_results) {
        let recall = calculate_recall(
            &hnsw_result.1.iter().map(|x| x.0).collect::<Vec<_>>(),
            &knn_result.1.iter().map(|x| x.0).collect::<Vec<_>>(),
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
    #[arg(long, default_value_t = 8)]
    no_search_threads: i32,

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

    #[arg(long, default_value = "heuristic", value_parser = ["heuristic", "simple"])]
    hnsw_neighbours_algorithm: String,

    #[arg(action = clap::ArgAction::SetFalse, default_value_t = true)]
    hnsw_neighbours_heuristic_extend_candidates: bool
}


fn run_benchmark<M: Metric + Copy + Send + Sync>(metric: M, args: &Args) {
    let vectors: Vec<Vec<f32>>;
    let mut query_vectors: Vec<Vec<f32>>;

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
        query_vectors = out;
    }
    else {
        query_vectors = vectors.clone();
        query_vectors.shuffle(&mut rand::rng());
    }

    let knn: KNN<M> = KNN::new(metric);
    let hnsw: HNSWGraph<M> = HNSWGraph::new(
        metric,
        vectors[0].len(),
        args.ef_construction,
        args.m,
        args.hnsw_neighbours_algorithm == "heuristic",
        args.hnsw_neighbours_heuristic_extend_candidates
    );

    let results = benchmark(&vectors, &query_vectors, hnsw, knn, args.ef_search, &args.recall, args.no_search_threads);

    for recall_level in &args.recall {
        let recall_per_query = results.0.get(&recall_level).unwrap();

        println!("Recall@{}: {:?}", recall_level, stats(recall_per_query, false));
    }

    println!("HNSW runtime {:?}", stats(&results.1, true));
    println!("KNN runtime {:?}", stats(&results.2, true));
}

fn main() {
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
