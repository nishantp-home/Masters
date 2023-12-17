from bs4 import BeautifulSoup

# Let's create an example html to play with
html = ['<html><heading style="font-size:20px"><i>This is the title<br><br></i></heading>',
        '<body><b>This is the body</b><p id="para1">This is para1<a href="www.google.com">Google</a></p>',
        '<p id="para2">This is para 2</p></body></html>']

# Create one string out of above list
html=''.join(html)

# Instantiate a soup object. This automatically identifies a structure in the html and creates a parse tree
# You can navigate the structure/tree in the soup and extract pieces that you are interested in
soup = BeautifulSoup(html)

# Print html in an easy-to-read formatted view
print(soup.prettify())

# At the top of the hierarchy in the parse tree is the <html></html> tag
# Then comes the <body></body> tag
# Within the body, the heading and paragraphs are 'siblings'
# The body is the parent of these tags, and the html tag is the parent of body tag
# Each tag has attributes - name, contents (a list), text, parent and siblings

# name attribute os jut the name of the tag
soup.html.name
soup.body.name

# text attribute mush together all the text in all the children of that tag
soup.body.text

# Contents is a list of the children of that tag
# In our example, the html tag has only 1 child, the body has 4 children
soup.html.contents
soup.body.contents

# parents and siblings referencing helps you navigating the parse tree
soup.body.parent.name
soup.b.nextSibling
soup.p.previousSibling

# finaAll, find are methods to search for specific tags, or  tags with certain attributes
bold = soup.findAll('b')
# This will find all tags which have the text in bold (enclosed in <b></b> tags) and returns a list
print(bold)
# to extract only the text, take each element of this list and get the text attribute
print(bold[0].text)

# Let's get all the text that is in the paragraphs (enclosed in <p></p> tags) as a single string
paras = ''.join([p.text for p in soup.findAll('p')])
print(paras)

# findAll can look for attributes as well. Let's find the text 
# for the paragraph with id 'para2'
soup.findAll(id="para2")[0].text

# Any text with font size 20
font20 = ''.join([p.text for p in soup.findAll(style="font-size:20px")])
print(font20)

# You can also pass in a list or a dictionary of tag names to search for
soup.findAll(['b', 'p'])
soup.findAll({'b':True, 'p':True})

# Find all links on a page and iterate through them to do some scraping
# find returns the first tag that matches the search, in this case, we have only 1 link on our page.
links = soup.find('a')

# Links are generally  of the form <a href='link'>'link-text'</a>

# Extract the url and the text separately
print(links['href'] + ' is the url')
print(links.text + ' is the text')


# Find text of the first paragraph after the first occurrency of the text "Google"
soup.find(text="Google").findNext('p').text

# A little shortcut to using findAll- if you call the tag itself as a function, you can use it in place of findAll
# with the same arguments
soup.body('p')