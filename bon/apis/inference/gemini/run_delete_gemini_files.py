import asyncio
import os
import time
from datetime import datetime, timedelta, timezone
from functools import partial

import google.generativeai as genai
from tqdm import tqdm

from bon.utils.utils import setup_environment

setup_environment()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
print(os.environ["GOOGLE_API_KEY"])


# Define a function to delete a file (not async)
def delete_file_sync(file, acceptable_time_diff):
    # Current datetime
    current_datetime = datetime.now(timezone.utc)

    # Calculate the time difference
    time_difference = current_datetime - file.update_time

    # Check if the difference is greater than 30 minutes
    if time_difference > timedelta(minutes=acceptable_time_diff):
        try:
            genai.delete_file(file)
            # print(f"Successfully deleted file {file}")
        except Exception as e:
            print(f"Failed to delete file {file}: {e}")
    else:
        print(f"SKIPPING FILE {file} because it is too recent")


# Define the main function to handle multiple deletions
async def main(file_list, batch_size, acceptable_time_diff):
    loop = asyncio.get_running_loop()
    semaphore = asyncio.Semaphore(batch_size)

    async def sem_delete_file(file, acceptable_time_diff):
        async with semaphore:
            await loop.run_in_executor(None, partial(delete_file_sync, file, acceptable_time_diff))

    # Create a list of tasks for each file deletion
    tasks = [sem_delete_file(file, acceptable_time_diff) for file in file_list]  # [:750]
    # Run the tasks concurrently
    await asyncio.gather(*tasks)


# List of files to delete
files_to_delete = []
batch_size = 500
acceptable_time_diff = 15
for f in reversed(list(genai.list_files())):
    files_to_delete.append(f)

print(f"Preparing to delete {len(files_to_delete)} files")
# replace with your file names
# Run the main function

for i in tqdm(range(0, len(files_to_delete) - (len(files_to_delete) % batch_size), batch_size)):
    start_time = time.time()
    file_batch = files_to_delete[i : i + batch_size]
    print(f"Deleting files {i}-{i+batch_size}")
    asyncio.run(main(file_batch, batch_size, acceptable_time_diff))

    end_time = time.time()

    # Buffer so we don't make too many requests in a minute
    wait = 70 - (end_time - start_time)
    print(f"{wait} seconds until next request")
    time.sleep(70 - (end_time - start_time))
