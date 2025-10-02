import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

url = "https://beb.iitd.ac.in/rti.html"

response = requests.get(url, verify=False)  # ignore SSL errors
soup = BeautifulSoup(response.text, "html.parser")

pdf_links = []

for link in soup.find_all("a", href=True):
    href = link["href"]
    if ".pdf" in href.lower():
        pdf_url = urljoin(url, href)
        pdf_links.append(pdf_url)

pprint.pp(pdf_links)
