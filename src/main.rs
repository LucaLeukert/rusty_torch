extern crate rust_stemmers;

use std::{env, fs};
use std::io::Result;
use edit_distance::edit_distance;
use rust_stemmers::{Algorithm, Stemmer};

use crate::classify::{classify, ImageClassification};
use crate::tokenizer::tokenize;

mod tokenizer;
mod classify;

fn save_json(location: &str, classifications: Vec<ImageClassification>) -> Result<()> {
    if fs::metadata(location).is_ok() {
        fs::remove_file(location)?;
    }

    let file = fs::File::create(location)?;

    serde_json::to_writer_pretty(file, &classifications)?;
    Ok(())
}

fn classify_path(path: &str, count: i64) -> Result<Vec<ImageClassification>> {
    let entries = fs::read_dir(path)?
        .map(|res| res.map(|e| e.path()))
        .collect::<Result<Vec<_>>>()?;

    let mut classifications: Vec<ImageClassification> = Vec::new();

    for entry in entries {
        let image_path = entry.as_path().to_str().unwrap_or_else(|| {
            panic!("Error: Could not read image path: {:?}", entry);
        });

        let classification = classify(image_path, count)?;
        classifications.push(classification);
    }

    Ok(classifications)
}

fn search(image_classifications: Vec<ImageClassification>, query: String) -> Result<()> {
    let mut unsorted: Vec<(&ImageClassification, usize)> = Vec::new();

    for image_classification in &image_classifications {
        let classification = &image_classification.classifications.get(0).unwrap();
        let mut distance = usize::MAX;

        classification.stem.iter().for_each(|class| {
            let current_distance = edit_distance(query.as_str(), class);

            if current_distance < distance {
                distance = current_distance;
            }
        });

        unsorted.push((image_classification, distance));
    }

    unsorted.sort_by(|a, b| a.1.cmp(&b.1));

    for i in 0..(unsorted.len().min(10)) {
        let (image_classification, _) = &unsorted[i];
        let classification = &image_classification.classifications.get(0).unwrap();

        println!("{}: {}", i + 1, image_classification.absolute_path);
        for class in &classification.class {
            println!("    - {}", class);
        }
    }
    Ok(())
}

fn read_json(path: &str) -> Result<Vec<ImageClassification>> {
    let file = fs::File::open(path)?;
    let classifications: Vec<ImageClassification> = serde_json::from_reader(file).map_err(|err| {
        eprintln!("Error: Could not parse JSON: {}", err);
    }).unwrap();

    Ok(classifications)
}

fn main() -> Result<()> {
    let mut args = env::args();
    args.next().expect("missing program name");

    let subcommand = args.next().ok_or_else(|| {
        eprintln!("Error: Missing subcommand");
    }).map_err(|_| {
        std::process::exit(1)
    }).unwrap();

    match subcommand.as_str() {
        "classify" => {
            let path = args.next().ok_or_else(|| {
                eprintln!("Error: Missing path");
            }).unwrap();
            let output = args.next().unwrap_or("output.json".to_string());
            let count = args.next().unwrap_or("1".to_string()).parse::<i64>().unwrap_or(1);

            let classifications = classify_path(&path, count)?;
            save_json(&output, classifications)?;
        }

        "search" => {
            let en_stemmer = Stemmer::create(Algorithm::English);

            let json = args.next().ok_or_else(|| {
                eprintln!("Error: Missing path");
            }).unwrap();
            let query = args.collect::<Vec<_>>().join(" ").trim().to_string();
            let query_tokens = tokenize(&query).collect::<Vec<_>>();
            let normalized_query = query_tokens.iter().enumerate().map(|(index, token)| {
                if index < query_tokens.len() - 1 {
                    format!("{} ", token.s)
                } else {
                    token.s.to_string()
                }
            }).collect::<Vec<_>>();

            if query_tokens.len() == 0 {
                eprintln!("Error: Missing query");
                return Ok(());
            }

            let classifications = read_json(&json)?;
            search(classifications, en_stemmer.stem(&normalized_query.join("")).to_string().to_lowercase())?;
        }

        _ => {
            eprintln!("Error: Unknown subcommand");
        }
    }

    Ok(())
}
