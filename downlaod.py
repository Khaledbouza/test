import json
import time
import random
import os
import requests
import gc  # Garbage collector to manage memory
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from datasets import load_dataset  # Hugging Face library for loading the dataset
from g4f.client import Client  # Importing g4f client

# Load the Open-Orca/OpenOrca dataset
def load_open_orca_dataset():
    dataset_path = os.path.expanduser("~/.cache/huggingface/datasets")
    dataset = load_dataset("Open-Orca/OpenOrca", cache_dir=dataset_path)
    return dataset['train']  # Assuming you want to use the 'train' split

dataset = load_open_orca_dataset()
dataset_iter = iter(dataset)  # Create an iterator from the dataset

# Ensure the 'openorca' folder exists
os.makedirs("openorca", exist_ok=True)

# Fetch and parse proxies from the given URL
async def fetch_proxies(proxy_url):
    async with aiohttp.ClientSession() as session:
        async with session.get(proxy_url) as response:
            return await response.text()

# Validate proxy by sending a request to the target website
async def is_proxy_working(proxy, test_url, verify_ssl=False):
    proxies = {"http": proxy, "https": proxy}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(test_url, proxy=proxies, timeout=10, ssl=verify_ssl) as response:
                if response.status == 200:
                    print(f"Proxy {proxy} is working for {test_url}")
                    return proxy
    except:
        return None

# Check proxies in parallel and return only the working ones
async def get_working_proxies(proxies, test_url, max_workers=10, verify_ssl=False):
    tasks = []
    async with aiohttp.ClientSession() as session:
        for proxy in proxies:
            tasks.append(is_proxy_working(proxy, test_url, verify_ssl))
        working_proxies = await asyncio.gather(*tasks)
    return [proxy for proxy in working_proxies if proxy]

# Function to translate text using g4f client and save the response
def translate_text_with_g4f(question, response, row_number, proxy=None, retries=3):
    # Combine question and response
    combined_text = f"Question: {question}\nAnswer: {response}"

    # Define the translation prompt with an example and detailed instructions
    example_translation = (\__________________________
        "Question: شنو اسمك؟\n"
        "Answer: اسمي جون.\n\n"
    )
    prompt_template = (
        "your role is tunisian arabic dailctic derja transltor Translate the following text into the Tunisian dialect. Ensure the output is structured as 'Question: ... Answer: ...'. "
        "Please capture the nuances of the Tunisian dialectic Arabic. make sure type and write in arabic tunisian lettre Here is an example to guide you:\n\n"
        f"{example_translation}"
        "Now, translate the following:\n\n"
    )
    full_input_text = f"{prompt_template}{combined_text}"

    for attempt in range(retries):
        try:
            # Set the proxy if available
            if proxy:
                os.environ['G4F_PROXY'] = f"http://{proxy}"

            # Create a g4f client and request translation
            client = Client()
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": full_input_text}]
            )

            response_text = response.choices[0].message.content
            print(f"Translation for row {row_number} completed.")

            # Print the response for tracking
            print(f"Response for row {row_number}: {response_text}")
            break

        except Exception as e:
            print(f"Error during translation for row {row_number} on attempt {attempt + 1}: {e}")
            response_text = "Error: No response or interaction issue."
            if attempt == retries - 1:
                print(f"Failed to translate row {row_number} after {retries} attempts.")

    # Save the response to a JSON file
    response_data = {
        "prompt": prompt_template,
        "text": combined_text,
        "response": response_text
    }
    
    file_name = f"openorca/response_row{row_number}_{random.randint(1000, 9999)}.json"
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(response_data, f, ensure_ascii=False, indent=4)
    print(f"Response for row {row_number} saved to {file_name}")

# Function to check and restart inactive workers
def check_and_restart_workers(executor, futures, proxies, test_url):
    active_workers = sum(1 for future in futures if not future.done())
    if active_workers < 4:
        print(f"Detected {4 - active_workers} inactive workers. Restarting them...")
        for _ in range(4 - active_workers):
            proxy = random.choice(proxies) if proxies else None
            if proxy:
                print(f"Switching to proxy: {proxy}")
            else:
                print("No working proxy found. Continuing without proxy.")
                proxy = None

            row = next(dataset_iter)
            question = row['question']
            response = row['response']

            futures.append(executor.submit(translate_text_with_g4f, question, response, random.randint(0, len(dataset) - 1), proxy))

# Main loop to process all rows in the dataset
def main_loop():
    proxy_url = "https://raw.githubusercontent.com/officialputuid/KangProxy/KangProxy/https/https.txt"
    test_url = "https://copilot.microsoft.com/"
    proxies = asyncio.run(fetch_proxies(proxy_url))
    proxies = proxies.splitlines()
    working_proxies = asyncio.run(get_working_proxies(proxies, test_url))
    proxy = None  # Initialize proxy variable
    lock = Lock()  # Lock for thread-safe operations

    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = []
        request_count = 0
        for i, row in enumerate(dataset):
            if request_count % 10 == 0:
                check_and_restart_workers(executor, futures, working_proxies, test_url)

            with lock:
                question = row['question']
                response = row['response']

            # Submit the translation task to the executor
            futures.append(executor.submit(translate_text_with_g4f, question, response, i, proxy))
            request_count += 1
        
        # Wait for all futures to complete
        for future in as_completed(futures):
            future.result()

# Start the main loop
main_loop()
