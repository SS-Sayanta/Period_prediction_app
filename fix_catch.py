import re
with open('public/index.html', 'r', encoding='utf-8') as f:
    text = f.read()

text = re.sub(r'\.catch\(function\(\)\{\s*(.*?)\s*\}\)', r'.catch(function(err){ console.error("Network Error:", err); \1 })', text)
text = re.sub(r'\.catch\(function\(([^)]+)\)\{\s*(?!console\.error)(.*?)\s*\}\)', r'.catch(function(\1){ console.error("Network Error:", \1); \2 })', text)

with open('public/index.html', 'w', encoding='utf-8') as f:
    f.write(text)
print('Done!')
