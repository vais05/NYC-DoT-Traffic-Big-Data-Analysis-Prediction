from pyspark.sql import SparkSession
import os
import ingestion

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("NYC Traffic Analysis - Ingestion") \
        .master("local[*]") \
        .getOrCreate()

    input_dir = r"C:\Users\VaishnaviM\Desktop\BIG_DATA\data\raw"
    output_dir = r"C:\Users\VaishnaviM\Desktop\BIG_DATA\data\processed"
    lookup_path = r"C:\Users\VaishnaviM\Desktop\BIG_DATA\data\taxi_zone_lookup.csv"

    trips_df = ingestion.read_parquet_files(spark, input_dir)
    print("Raw trip data:")
    trips_df.show(3)

    zones_df = ingestion.read_zone_lookup(spark, lookup_path)
    print("Zone lookup data:")
    zones_df.show(3)

    ingestion.write_processed_data(trips_df, output_dir)
    print(f"✅ Processed data written to: {output_dir}")

    spark.stop()
