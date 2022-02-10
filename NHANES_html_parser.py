from bs4 import BeautifulSoup
import urllib.request
import re

url = urllib.request.urlopen("https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DBQ_J.htm")

nhanes_dbq = url.read()

nhanes_dbq_html= BeautifulSoup(nhanes_dbq, "html.parser")

with open("nhanes_dbq.html", "w", encoding="utf-8") as file:
    file.write(str(nhanes_dbq_html))

body = nhanes_dbq_html.body

def element_length(html,index, tag, el_class):
    return len(str(html.find_all("dd", class_ = "info")[index]))

print(len(str(body.find_all("dd", class_ = "info")[2])))

print(element_length(body,2,"dd","info"))

