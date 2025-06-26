
use tokio_postgres::NoTls;
use std::env;
use rand::{prelude::IndexedRandom, rng};
use rayon::prelude::*;
use strsim::levenshtein;

#[derive(Clone)]
struct NameEntry {
    name: String,
    lei: String,
}

const BATCH_SIZE: i64 = 1000;
const RANDOM_SAMPLE_SIZE: usize = 100;

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
    let mut rng = rng();

    // Step 1: Load all names into memory for hard negative mining
    let name_rows = client.query("SELECT name, lei FROM names", &[]).await?;
    let all_names: Vec<NameEntry> = name_rows.into_iter()
        .map(|r| NameEntry {
            name: r.get("name"),
            lei: r.get("lei"),
        })
        .collect();

    println!("anchor,positive,hard_negative");

    // Step 2: Paginate through anchor-positive pairs
    let mut offset = 0;
    loop {
        let rows = client.query(
            "
            WITH ranked_pairs AS (
                SELECT
                    n1.lei,
                    n1.name AS anchor,
                    n2.name AS positive,
                    ROW_NUMBER() OVER () AS rn
                FROM names n1
                JOIN names n2 ON n1.lei = n2.lei AND n1.name != n2.name
            )
            SELECT lei, anchor, positive
            FROM ranked_pairs
            WHERE rn > $1 AND rn <= $2
            ",
            &[&offset, &(offset + BATCH_SIZE)],
        ).await?;

        if rows.is_empty() {
            break;
        }

        for row in rows {
            let anchor: String = row.get("anchor");
            let positive: String = row.get("positive");
            let anchor_lei: String = row.get("lei");


            let negative = all_names.choose_multiple(&mut rng, RANDOM_SAMPLE_SIZE)
                .cloned()
                .collect::<Vec<_>>()
                .par_iter()
                .filter(|e| e.lei != anchor_lei)
                .map(|e| (e.name.clone(), levenshtein(&anchor.to_lowercase(), &e.name.to_lowercase())))
                .filter(|(_, dist)| *dist < 5) // Optional: tune for hardness
                .min_by_key(|(_, dist)| *dist)
                .map(|(name, _)| name);

            if let Some(hard_neg) = negative {
                println!("\"{}\",\"{}\",\"{}\"", anchor, positive, hard_neg);
            }
        }

        offset += BATCH_SIZE;
    }

    Ok(())
}
