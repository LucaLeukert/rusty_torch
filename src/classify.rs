use std::fs;
use std::io::{Result, Error};
use std::path::PathBuf;
use image::ImageFormat;
use image::io::Reader;
use rust_stemmers::{Algorithm, Stemmer};
use serde::{Deserialize, Serialize};
use tch::Device;
use tch::nn::{ModuleT, VarStore};
use tch::vision::{imagenet, resnet};
use crate::tokenizer::tokenize;

const WEIGHTS: &str = "models/resnet34.ot";

#[derive(Serialize, Deserialize, Debug)]
pub struct Classification {
    probability: f64,
    pub(crate) stem: Vec<String>,
    pub(crate) class: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ImageClassification {
    image_path: String,
    pub(crate) absolute_path: String,
    pub(crate) classifications: Vec<Classification>,
}

fn parse_classifications(classifications: Vec<(f64, String)>, stemmer: Stemmer) -> Vec<Classification> {
    let mut normalized_classifications: Vec<Classification> = Vec::new();

    for (probability, class) in classifications {
        let classes = class.split(", ").collect::<Vec<_>>();
        let mut normalized_classes: Vec<String> = Vec::new();
        let mut parsed_classes: Vec<String> = Vec::new();

        for class in &classes {
            let tokens = tokenize(class);
            let count = tokenize(class).count();

            let mut normalized_token: Vec<String> = Vec::new();
            for (index, token) in tokens.enumerate() {
                normalized_token.push(token.s.to_lowercase());
                if index < count - 1 {
                    normalized_token.push(" ".to_string());
                }
            }

            let stemmed_class = stemmer.stem(&*normalized_token.join("")).to_string().to_lowercase();
            normalized_classes.push(stemmed_class);
            parsed_classes.push(normalized_token.join(""));
        }

        normalized_classifications.push(Classification {
            probability,
            stem: normalized_classes,
            class: parsed_classes,
        });
    }

    normalized_classifications
}

pub fn classify(image_path: &str, count: i64) -> Result<ImageClassification> {
    let en_stemmer = Stemmer::create(Algorithm::English);
    let absolute_path = validate_image(image_path)?;

    let mut var_store = VarStore::new(Device::Cpu);
    let image = imagenet::load_image_and_resize224(image_path).map_err(|err| {
        eprintln!("Error: Could not read image into ImageNet: {} because: {}", image_path, err);
    }).unwrap();

    /*let alexnet_model = alexnet::alexnet(&var_store.root(), imagenet::CLASS_COUNT);*/

    let resnet34 = resnet::resnet34(&var_store.root(), imagenet::CLASS_COUNT);
    var_store.load(WEIGHTS).map_err(|err| {
        panic!("Error: Could not read Image {:?}", err);
    }).expect("Error: Could not read Image");

    let output = resnet34.forward_t(&image.unsqueeze(0), false).softmax(-1, tch::Kind::Float);
    let classifications = parse_classifications(imagenet::top(&output, count), en_stemmer);

    print!("{}: ", absolute_path);
    for classification in &classifications {
        print!("{:?}: {:.2}%, ", classification.class, classification.probability * 100.0);
    }
    println!();

    Ok(ImageClassification {
        image_path: image_path.to_string(),
        absolute_path: absolute_path.to_string(),
        classifications,
    })
}

fn validate_image(image_path: &str) -> Result<String> {
    let image = Reader::open(image_path).map_err(|err| {
        eprintln!("Error: Could not read image: {} because: {}", image_path, err);
    }).unwrap();
    let image_format = image.format().unwrap_or(ImageFormat::Tiff);

    match image_format {
        ImageFormat::Png | ImageFormat::Jpeg | ImageFormat::WebP => {}

        _ => {
            eprintln!("Error: Image format is not supported");
            return Err(Error::new(std::io::ErrorKind::Other, "Image format is not supported"));
        }
    }

    let path_buf = PathBuf::from(image_path);
    let path = fs::canonicalize(&path_buf).map_err(|err| {
        eprintln!("Error: Could not read image: {} because: {}", image_path, err);
    }).unwrap().as_path().to_str().unwrap_or(image_path).to_string().replace(r"\\?\", "");
    Ok(path)
}