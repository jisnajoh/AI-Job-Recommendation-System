import requests
import pandas as pd

APP_ID = "44463cba"
APP_KEY = "37d12d7510bd6a36375704d1a2f0dfbc"

url = f"https://api.adzuna.com/v1/api/jobs/us/search/1"

params = {
    "app_id": APP_ID,
    "app_key": APP_KEY,
    "results_per_page": 50,
    "what": "data analyst",
    "content-type": "application/json"
}

response = requests.get(url, params=params)

data = response.json()

jobs = []

for job in data["results"]:
    jobs.append({
        "job_title": job["title"],
        "company": job["company"]["display_name"],
        "job_location": job["location"]["display_name"],
        "job_description": job["description"],
        "job_link": job["redirect_url"],
        "job_skills": "",
        "search_country": "United States",
        "job_level": "",
        "job_type": ""
    })

df = pd.DataFrame(jobs)

df.to_csv("live_jobs.csv", index=False)

print("Jobs saved successfully")