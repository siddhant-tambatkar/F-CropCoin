import os
import requests

# Replace with your Unsplash Access Key
access_key = "BvKNFoVhkwp5c5LIgWVZUxA7k4GYoguTVhVPKioPvGE"
search_url = "https://api.unsplash.com/search/photos"

crops = ['wheat', 'mungbean', 'Tea', 'millet', 'maize', 'lentil', 'jute', 'cofee', 'cotton', 'ground nut',
         'peas', 'rubber', 'sugarcane', 'tobacco', 'kidney beans', 'moth beans', 'coconut', 'blackgram',
         'adzuki beans', 'pigeon peas', 'chick peas', 'banana', 'grapes', 'apple', 'mango', 'muskmelon',
         'orange', 'papaya', 'watermelon', 'pomegranate']

headers = {"Authorization": f"Client-ID {access_key}"}

def download_image(query, file_path):
    params = {"query": query, "per_page": 1}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    if search_results["results"]:
        image_url = search_results["results"][0]["urls"]["regular"]
        image_data = requests.get(image_url)
        with open(file_path, "wb") as f:
            f.write(image_data.content)
    else:
        print(f"No images found for {query}")

for crop in crops:
    file_path = f"{crop.replace(' ', '_')}.jpg"
    try:
        download_image(crop, file_path)
        print(f"Downloaded image for {crop}")
    except Exception as e:
        print(f"Could not download image for {crop}: {e}")
