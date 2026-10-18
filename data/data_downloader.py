import pandas as pd
import requests
import io
import os

def download_data():
    url = "https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv"
    print(f"Downloading dataset from {url}...")
    response = requests.get(url)
    if response.status_code == 200:
        data = response.content.decode('utf-8')
        df = pd.read_csv(io.StringIO(data))
        output_path = "xai-healthcare/data/heart.csv"
        df.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path}")
        print(df.head())
    else:
        print(f"Failed to download dataset. Status code: {response.status_code}")

if __name__ == "__main__":
    download_data()
