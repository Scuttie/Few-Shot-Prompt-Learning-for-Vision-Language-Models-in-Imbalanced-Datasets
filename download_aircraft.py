import os
import urllib.request
import tarfile

def download_file(url, dest_path):
    """
    Downloads a file from a URL to a destination path.
    """
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return
    print(f"Downloading {url} ...")
    try:
        with urllib.request.urlopen(url) as response, open(dest_path, 'wb') as out_file:
            file_size = int(response.getheader('Content-Length'))
            downloaded = 0
            block_size = 8192
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                done = int(50 * downloaded / file_size)
                print(f"\r[{'=' * done}{' ' * (50 - done)}] {downloaded * 100 / file_size:.2f}%", end='')
        print(f"\nDownloaded {dest_path}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def extract_tar(file_path, extract_path):
    """
    Extracts a .tar.gz or .tgz file to the specified directory.
    """
    if not os.path.exists(extract_path):
        os.makedirs(extract_path, exist_ok=True)
    print(f"Extracting {file_path} to {extract_path} ...")
    try:
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(path=extract_path)
        print("Extraction completed.")
    except Exception as e:
        print(f"Error extracting {file_path}: {e}")

def main():
    # FGVC Aircraft dataset URL
    dataset_url = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"

    # Directory to store the dataset
    dataset_dir = "fgvc_aircraft"
    os.makedirs(dataset_dir, exist_ok=True)

    # Local file name
    tar_filename = "fgvc-aircraft-2013b.tar.gz"
    dest_path = os.path.join(dataset_dir, tar_filename)

    # 1. Download the tar.gz file
    download_file(dataset_url, dest_path)

    # 2. Extract the downloaded file
    extract_tar(dest_path, dataset_dir)

    print("FGVC Aircraft Dataset is downloaded and extracted successfully.")

if __name__ == "__main__":
    main()
