import requests
import zipfile
import os

# Define URLs for Cricsheet data
urls = {
    "all_json": "https://cricsheet.org/downloads/all_json.zip",
}

# Directory to save data
data_dir = "../data/raw/cricsheet_data"

# Ensure the directory exists
os.makedirs(data_dir, exist_ok=True)

# Download and extract data
for format_name, url in urls.items():
    print(f"Downloading {format_name} data...")
    response = requests.get(url)
    zip_path = os.path.join(data_dir, f"{format_name}.zip")
    
    # Save the ZIP file
    with open(zip_path, "wb") as zip_file:
        zip_file.write(response.content)
    
    # Extract the ZIP file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        extract_path = os.path.join(data_dir, format_name)
        zip_ref.extractall(extract_path)
        print(f"Extracted {format_name} data to {extract_path}")
