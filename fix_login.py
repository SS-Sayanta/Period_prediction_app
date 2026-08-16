with open('public/index.html', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('/api/auth/', '/auth/')
text = text.replace("var pass  = $('loginPass').value;", "var password  = $('loginPass').value;")
text = text.replace("body: JSON.stringify({ email: email, password: pass })", "body: JSON.stringify({ email: email, password: password })")
text = text.replace("console.error(\"Network Error:\", err);", "console.error(\"Auth Exception:\", err);")

with open('public/index.html', 'w', encoding='utf-8') as f:
    f.write(text)
print('Updated index.html')
