import os
import csv
from datasets import load_dataset
from tqdm import tqdm


def download_and_save(dataset_name="harvard-lil/cold-cases", split="train"):
    """
    Download the first 200,000 rows of data from HuggingFace Datasets
    and save it in data/raw/ with a progress bar.
    """
    # raw path
    raw_data_dir = os.path.join(os.path.dirname(__file__), "../../data/raw")
    os.makedirs(raw_data_dir, exist_ok=True)

    print(f" Download data from: {dataset_name} ...")

    # Select the first 100,000 rows from the specified split using slicing
    limited_split = f"{split}[:100000]"
    print(f"Loading first 100,000 rows from split: '{limited_split}'")
    ds = load_dataset(dataset_name, split=limited_split)

    # save in CSV
    output_filename = f"{dataset_name.replace('/', '_')}_{split}_first200k.csv"
    output_path = os.path.join(raw_data_dir, output_filename)

    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            headers = ds.column_names
            writer.writerow(headers)

            for row in tqdm(ds, desc=f"Saving to {output_filename}"):
                writer.writerow([row[header] for header in headers])

        print(f"\n Data saved: {output_path}")

    except Exception as e:
        print(f"An error occurred while saving the file: {e}")


if __name__ == "__main__":
    download_and_save()

