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

    // Load all anchor-positive pairs
    let anchor_rows = client
        .query(
            "
            SELECT n1.lei, n1.name AS anchor, n2.name AS positive
            FROM names n1
            JOIN names n2 ON n1.lei = n2.lei AND n1.name != n2.name
            ",
            &[],
        )
        .await?;

    // Buffered channel for parallel workers
    let (sender, receiver): (channel::Sender<Vec<String>>, channel::Receiver<Vec<String>>) =
        channel::bounded(10000);

    // Printer thread using csv::Writer for safe escaping
    let printer_handle = thread::spawn(move || {
        let stdout = io::stdout();
        let handle = stdout.lock();
        let mut wtr = WriterBuilder::new().from_writer(handle);

        wtr.write_record(&["anchor", "positive", "hard_negative"]).ok();

        for record in receiver {
            wtr.write_record(&record).ok();
        }

        wtr.flush().ok();
    });

    // Clone Arc for workers
    let all_names = all_names.clone();

    // Parallel processing using for_each_with + your rng()
    anchor_rows.into_par_iter().for_each_with(sender.clone(), move |s, row| {
        let anchor: String = row.get("anchor");
        let positive: String = row.get("positive");
        let anchor_lei: String = row.get("lei");
        let anchor_lc = anchor.to_lowercase();

        let mut local_rng = rng(); // <- preserve your usage of rng()

        let negative = all_names
            .choose_multiple(&mut local_rng, RANDOM_SAMPLE_SIZE)
            .cloned()
            .collect::<Vec<_>>()
            .into_par_iter()
            .filter(|e| e.lei != anchor_lei)
            .map(|e| {
                let dist = levenshtein(&anchor_lc, &e.name.to_lowercase());
                (e.name.clone(), dist)
            })
            .min_by_key(|(_, dist)| *dist)
            .map(|(name, _)| name);

        if let Some(hard_neg) = negative {
            let record = vec![anchor, positive, hard_neg];
            s.send(record).ok(); // ignore send errors
        }
    });

    drop(sender);
    printer_handle.join().unwrap();

    Ok(())
}

