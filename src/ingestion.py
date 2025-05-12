import os
from pyspark.sql.functions import col, monotonically_increasing_id, floor
from pyspark.sql import SparkSession
from src.config import RAW_DATA_PATH, EXTERNAL_DATA_PATH, PROCESSED_DATA_PATH

def get_spark_session():
    try:
        spark = SparkSession.builder \
            .master("local[*]") \
            .appName("MonthlyNYCIngestion") \
            .config("spark.sql.shuffle.partitions", "4") \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()
        print("✅ Spark session initialized successfully.")
        return spark
    except Exception as e:
        print(f"❌ Failed to create Spark session: {e}")
        raise

def load_lookup_csv(spark):
    """Load location lookup CSV."""
    return spark.read.option("header", True).csv(EXTERNAL_DATA_PATH)

def enrich_with_location(df, lookup_df):
    """Enrich data with pickup and dropoff location details."""
    pickup_lookup = lookup_df.drop("service_zone") \
        .withColumnRenamed("LocationID", "PULocationID") \
        .withColumnRenamed("Zone", "pickup_zone") \
        .withColumnRenamed("latitude", "pickup_lat") \
        .withColumnRenamed("longitude", "pickup_lon") \
        .withColumnRenamed("Borough", "pickup_borough")

    dropoff_lookup = lookup_df.drop("Borough") \
        .withColumnRenamed("LocationID", "DOLocationID") \
        .withColumnRenamed("Zone", "dropoff_zone") \
        .withColumnRenamed("latitude", "dropoff_lat") \
        .withColumnRenamed("longitude", "dropoff_lon") \
        .withColumnRenamed("service_zone", "dropoff_service_zone")

    df = df.join(pickup_lookup, on="PULocationID", how="left") \
           .join(dropoff_lookup, on="DOLocationID", how="left")
    return df

def save_csv_in_chunks(df, output_dir, max_rows=1_000_000, month_tag=""):
    """Save the DataFrame as multiple CSV chunks if the row count exceeds max_rows."""
    df_count = df.count()
    num_chunks = (df_count // max_rows) + (1 if df_count % max_rows != 0 else 0)

    print(f"🔄 Saving {df_count} rows in ~{num_chunks} CSV chunks for {month_tag}...")

    df_with_index = df.withColumn("row_num", monotonically_increasing_id())
    df_with_chunk_id = df_with_index.withColumn("chunk_id", floor(col("row_num") / max_rows))

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Write each chunk to a separate CSV file
    for i in range(num_chunks):
        chunk = df_with_chunk_id.filter(col("chunk_id") == i).drop("row_num", "chunk_id")
        chunk_path = os.path.join(output_dir, f"{month_tag}_trip_data_part_{i+1}.csv")
        chunk.write.mode("overwrite").option("header", True).csv(chunk_path)
        print(f"📝 Saved chunk {i+1} to: {chunk_path}")

def run_monthly_ingestion():
    """Main ingestion pipeline to process monthly data."""
    try:
        spark = get_spark_session()
        lookup_df = load_lookup_csv(spark)

        # Get list of Parquet files
        input_files = sorted([f for f in os.listdir(RAW_DATA_PATH) if f.endswith(".parquet")])
        if not input_files:
            raise FileNotFoundError("No Parquet files found in RAW_DATA_PATH.")

        # Process each file
        for idx, filename in enumerate(input_files, 1):
            print(f"\n📁 Processing file {idx}/{len(input_files)}: {filename}")
            file_path = os.path.join(RAW_DATA_PATH, filename)

            # Derive month tag from filename (e.g., 'yellow_tripdata_2024-01.parquet' -> 'month_2024-01')
            month_tag = f"month_{filename[-12:-8]}" if "-" in filename else f"month_{idx:02d}"

            # Step 1: Read raw Parquet
            df = spark.read.parquet(file_path)
            print(f"📦 Loaded {df.count()} raw rows from {filename}")

            # Step 2: Clean data
            cleaned_df = df.dropna().filter("trip_distance > 0 AND fare_amount > 0")

            # Step 3: Enrich with location data
            enriched_df = enrich_with_location(cleaned_df, lookup_df)

            # Optional debug sample
            print(f"🧪 Sample rows for {month_tag}:")
            enriched_df.select("tpep_pickup_datetime", "pickup_zone", "dropoff_zone").show(3, truncate=False)

            # Step 4: Save as single Parquet
            parquet_output_path = os.path.join(PROCESSED_DATA_PATH, f"{month_tag}.parquet")
            enriched_df.coalesce(1).write.mode("overwrite").parquet(parquet_output_path)
            print(f"✅ Saved cleaned Parquet to: {parquet_output_path}")

            # Step 5: Save CSV chunks
            csv_output_dir = os.path.join(PROCESSED_DATA_PATH, "csv_chunks")
            save_csv_in_chunks(enriched_df, csv_output_dir, month_tag=month_tag)

        spark.stop()
        print("\n🛑 Spark session stopped.")

    except Exception as e:
        print(f"❌ Ingestion failed: {e}")

if __name__ == "__main__":
    run_monthly_ingestion()
