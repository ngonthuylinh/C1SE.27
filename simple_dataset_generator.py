#!/usr/bin/env python3
"""
Simple Dataset Generator for Form Agent AI
"""

import pandas as pd
import random
import os
import json
from datetime import datetime, timedelta
from tqdm import tqdm
import time

def generate_record(record_id):
    """Generate a single record"""
    
    # Categories and keywords
    categories = {
        "it": ["cloud computing", "machine learning", "web development", "cybersecurity", "data science"],
        "economics": ["investment planning", "market analysis", "financial modeling", "portfolio management"], 
        "marketing": ["digital marketing", "social media", "content marketing", "brand management"]
    }
    
    # Choose random category
    category = random.choice(list(categories.keys()))
    keyword = random.choice(categories[category])
    
    # Form types
    form_types = ["registration", "survey", "application", "assessment"]
    form_type = random.choice(form_types)
    
    # Generate record
    record = {
        "record_id": record_id,
        "keyword": keyword,
        "category": category,
        "form_type": form_type,
        "complexity": random.choice(["Simple", "Moderate", "Complex"]),
        "num_fields": random.randint(5, 15),
        "created_date": (datetime.now() - timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d"),
        "form_title": f"{keyword.title()} {form_type.title()}",
        "estimated_time": random.randint(3, 12)
    }
    
    return record

def generate_dataset(total_records=500000000, batch_size=100000):
    """Generate large dataset in batches"""
    
    print(f"🚀 Generating {total_records:,} records in batches of {batch_size:,}")
    
    # Create output directory
    output_dir = "datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    generated_count = 0
    batch_number = 0
    start_time = time.time()
    
    with tqdm(total=total_records, desc="Generating Records") as pbar:
        
        while generated_count < total_records:
            current_batch_size = min(batch_size, total_records - generated_count)
            
            print(f"\n📊 Generating batch {batch_number + 1}: records {generated_count + 1:,} to {generated_count + current_batch_size:,}")
            
            # Generate batch
            batch_records = []
            for i in range(current_batch_size):
                record_id = generated_count + i + 1
                record = generate_record(record_id)
                batch_records.append(record)
            
            # Save batch
            batch_df = pd.DataFrame(batch_records)
            batch_file = f"{output_dir}/batch_{batch_number + 1:04d}.csv"
            batch_df.to_csv(batch_file, index=False)
            
            # Update counters
            generated_count += current_batch_size
            batch_number += 1
            pbar.update(current_batch_size)
            
            # Performance stats
            elapsed = time.time() - start_time
            rate = generated_count / elapsed if elapsed > 0 else 0
            
            print(f"✅ Batch {batch_number} saved: {batch_file}")
            print(f"📊 Progress: {generated_count:,}/{total_records:,} ({100*generated_count/total_records:.1f}%)")
            print(f"⚡ Rate: {rate:.0f} records/sec")
            
            # Save checkpoint every 50 batches
            if batch_number % 50 == 0:
                checkpoint = {
                    "generated_count": generated_count,
                    "batch_number": batch_number,
                    "timestamp": datetime.now().isoformat(),
                    "progress_percent": 100 * generated_count / total_records
                }
                with open(f"{output_dir}/checkpoint.json", "w") as f:
                    json.dump(checkpoint, f, indent=2)
                print(f"💾 Checkpoint saved")
    
    # Create sample files
    print(f"\n📝 Creating sample files...")
    create_samples(output_dir, batch_number)
    
    total_time = time.time() - start_time
    print(f"\n🎉 Generation completed!")
    print(f"📊 Generated: {generated_count:,} records in {batch_number} batches")
    print(f"⏱️  Total time: {total_time/3600:.2f} hours") 
    print(f"📁 Files location: {os.path.abspath(output_dir)}")
    
    return generated_count

def create_samples(output_dir, batch_count):
    """Create sample datasets"""
    
    # Load first batch for samples
    first_batch = f"{output_dir}/batch_0001.csv"
    if os.path.exists(first_batch):
        df = pd.read_csv(first_batch)
        
        # Create different sample sizes
        sample_sizes = [1000, 10000, 50000]
        for size in sample_sizes:
            if len(df) >= size:
                sample = df.head(size)
                sample_file = f"{output_dir}/sample_{size}.csv"
                sample.to_csv(sample_file, index=False)
                print(f"   ✅ {sample_file}")
        
        # Create category samples
        for category in ["it", "economics", "marketing"]:
            cat_data = df[df['category'] == category]
            if len(cat_data) > 0:
                sample_size = min(3000, len(cat_data))
                sample = cat_data.head(sample_size)
                cat_file = f"{output_dir}/sample_{category}_{sample_size}.csv"
                sample.to_csv(cat_file, index=False)
                print(f"   ✅ {cat_file}")

def main():
    """Main function"""
    print("🤖 Form Agent AI - Dataset Generator")
    print("📊 Generating 500,000,000 training records...")
    
    try:
        total_generated = generate_dataset(
            total_records=500000000,  # 500 million records
            batch_size=100000         # 100k per batch (5000 batches total)
        )
        print(f"🎉 Successfully generated {total_generated:,} records!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Generation interrupted by user. Partial data saved.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
