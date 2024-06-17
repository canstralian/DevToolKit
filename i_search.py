import requests 
from bs4 import BeautifulSoup

headers_Get = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:49.0) Gecko/20100101 Firefox/49.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }

def i_search(url):
    response = requests.get(url)
    try:
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        out = 'URL Links:\n'.join([p.text for p in soup.find_all('a')])
        out = ' '.join([p.text for p in soup.find_all('p')])
        if out == "" or out == None:
            out = ' '.join([p.text for p in soup.find_all('article')])
        
        return out
    except Exception as e:
        print (e)
        return "An Error occured when fetching this website. Please check the URL and try again, or use a different URL"



def b_search(q):
    #s = requests.Session()
    #url = q
    #r = s.get(url, headers=headers_Get)
    r=requests.get(q)
    soup = BeautifulSoup(r.text, "html.parser")
    output = []
    for searchWrapper in soup.find_all('article'): #this line may change in future based on google's web page structure
        url = searchWrapper.find('a')["href"] 
        text = searchWrapper.find('a').text.strip()
        result = {'text': text, 'url': url}
        output.append(result)

    return output  
def google(q):
    s = requests.Session()
    q = '+'.join(q.split())
    url = 'https://www.google.com/search?q=' + q + '&ie=utf-8&oe=utf-8'
    r = s.get(url, headers=headers_Get)

    soup = BeautifulSoup(r.text, "html.parser")
    output = []
    for searchWrapper in soup.find_all('h3', {'class':'r'}): #this line may change in future based on google's web page structure
        url = searchWrapper.find('a')["href"] 
        text = searchWrapper.find('a').text.strip()
        result = {'text': text, 'url': url}
        output.append(result)

    return output