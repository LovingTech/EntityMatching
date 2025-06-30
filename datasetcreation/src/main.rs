use tokio_postgres::NoTls;
use std::env;
use rand::prelude::IndexedRandom;
use rand::rng;
use rayon::prelude::*;
use strsim::levenshtein;
use crossbeam::channel;
use std::sync::Arc;
use std::thread;
use std::io;
use csv::WriterBuilder;

#[derive(Clone)]
struct NameEntry {
    name: String,
    lei: String,
}

const RANDOM_SAMPLE_SIZE: usize = 10000;
const NEGATIVES: usize = 300000;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv::dotenv().ok();
    let db_password = env::var("DB_PASSWORD").expect("DB_PASSWORD must be set in .env");
    let db_url = format!("postgres://postgres:{}@127.0.0.1:5432/lei", db_password);

    let (client, connection) = tokio_postgres::connect(&db_url, NoTls).await?;
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("DB connection error: {}", e);
        }
    });

    // Load all names into memory and wrap in Arc<[T]>
    let name_rows = client.query("SELECT name, lei FROM names", &[]).await?;
    let all_names: Arc<[NameEntry]> = Arc::from(
        name_rows
            .into_iter()
            .map(|r| NameEntry {
                name: r.get("name"),
                lei: r.get("lei"),
            })
            .collect::<Vec<_>>(),
    );

    // Buffered channel for parallel workers
    let (sender, receiver): (channel::Sender<Vec<String>>, channel::Receiver<Vec<String>>) =
        channel::bounded(10000);

    // Printer thread using csv::Writer for safe escaping
    let printer_handle = thread::spawn(move || {
        let stdout = io::stdout();
        let handle = stdout.lock();
        let mut wtr = WriterBuilder::new().from_writer(handle);

        wtr.write_record(&["texta", "textb", "label"]).ok();

        for record in receiver {
            wtr.write_record(&record).ok();
        }

        wtr.flush().ok();
    });

    // Load all anchor-positive pairs
    let positives = client
        .query(
            "
            WITH possible_name AS (
              SELECT * FROM names
              WHERE LENGTH(name) < 255
            )
             SELECT 
                n1.name AS texta, 
                n2.name AS textb 
              FROM 
                possible_name n1
              JOIN 
                possible_name n2 
                ON n1.lei = n2.lei
                AND n1.name < n2.name
                AND levenshtein(n1.name, n2.name) < FLOOR(GREATEST(6, LENGTH(n1.name) * 0.3))
            ",
            &[],
        )
        .await?;

    for record in positives {
        let texta: String = record.get("texta");
        let textb: String = record.get("textb");

        let result = vec![texta,textb,true.to_string()];
        sender.send(result).ok();
    }

    let mut global_rng = rng();

    // Parallel processing using for_each_with + your rng()
    all_names.choose_multiple(&mut global_rng, NEGATIVES)
        .cloned()
        .collect::<Vec<_>>()
        .into_par_iter()
        .for_each_with(sender.clone(), move |s, row| {
        let texta: String = row.name;
        let lei: String = row.lei;
        let texta_lc = texta.to_lowercase();

        let mut local_rng = rng(); 

        let textb = all_names
            .choose_multiple(&mut local_rng, RANDOM_SAMPLE_SIZE)
            .cloned()
            .collect::<Vec<_>>()
            .into_par_iter()
            .filter(|r| r.lei != lei)
            .map(|r| {
                let dist = levenshtein(&texta_lc, &r.name.to_lowercase());
                (r.name.clone(), dist)
            })
            .filter(|(_, dist)| *dist < ((texta.len() as f32 * 0.3).floor() as usize).max(6) )
            .min_by_key(|(_, dist)| *dist)
            .map(|(name, _)| name);

        if let Some(hard_neg) = textb {
            let record = vec![texta, hard_neg, false.to_string()];
            s.send(record).ok(); 
        }
    });

    drop(sender);
    printer_handle.join().unwrap();

    Ok(())
}

