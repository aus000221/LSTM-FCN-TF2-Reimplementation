import requests
import os

# written by zz3128
def download_zip(URL = "https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/UCRArchive_2018.zip"):
    """
    download the latest 127 UCR dataset from the web.
    Currently we only support to download 127 UCR dataset 2018 version, if you want to download other zip.
    If you want to download other zip please modify the method or build yourself.
    
    param URL: the url to the UCRArchieve 2018 dataset zip
    """
    # URL of the ZIP file to download
    url = URL

    # Local path to save the downloaded file
    #output_file = url.split("/")[-1]
    output_file = "UCRArchive_2018.zip"
    

    try:
        # Send a GET request to the URL
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP errors

        # Open a local file in write-binary mode and save the content
        with open(output_file, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"File downloaded successfully as {output_file}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while downloading the datazip file: {e}")
        
    return None

import zipfile
def extract_zip(file_path = "UCRArchive_2018.zip", pwd = "someone"):
    # Extract the zip fileto the current location
    zip_file_path = file_path
    password = pwd

    # It can take about 5-10 minutes to extract the files
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall("UCRArchive_2018", pwd=bytes(password, "utf-8"))
        print("Files extracted successfully!")
        
    return None
