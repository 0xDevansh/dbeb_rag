import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

url = "https://beb.iitd.ac.in/rti.html"

response = requests.get(url, verify=False)
soup = BeautifulSoup(response.text, "html.parser")

os.makedirs("pdfs", exist_ok=True)

for link in soup.find_all("a", href=True):
    href = link["href"]
    if ".pdf" in href.lower():
        pdf_url = urljoin(url, href) 
        pdf_name = pdf_url.split("/")[-1]
        print(f"Downloading {pdf_name}...")
        r = requests.get(pdf_url, verify=False)
        with open(os.path.join("pdfs", pdf_name), "wb") as f:
            f.write(r.content)
