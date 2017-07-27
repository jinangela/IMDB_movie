import urllib2
from bs4 import BeautifulSoup

url = "http://www.imdb.com/title/tt0499549/?ref_=fn_t..."
html = urllib2.urlopen(url)
soup = BeautifulSoup(html, "lxml")

imgs = soup.findAll("div", {"class":"thumb-pic"})
for img in imgs:
        print img.a['href'].split("imgurl=")[1]