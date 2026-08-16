import re

with open('public/index.html', 'r', encoding='utf-8') as f:
    text = f.read()

replacement = '''
      .then(function(r) {
        return r.text().then(function(txt) {
          var d;
          try { d = JSON.parse(txt); } catch(e) { d = { detail: txt || r.statusText }; }
          if (!r.ok) console.error("API Error: Status", r.status, d.detail);
          return { ok: r.ok, d: d };
        });
      })
'''.strip()

# Replace the single line matching the pattern
pattern = r'\.then\(function\(r\)\{\s*return r\.json\(\)\.then\(function\(d\)\{\s*if\(!r\.ok\) console\.error\("API Error: Status", r\.status\);\s*return \{ok:r\.ok,d:d\};\s*\}\);\s*\}\)'
text = re.sub(pattern, replacement, text)

with open('public/index.html', 'w', encoding='utf-8') as f:
    f.write(text)

print("Fixed frontend fetch parsing")
