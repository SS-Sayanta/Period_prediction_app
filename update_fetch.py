import re
with open('public/index.html', 'r', encoding='utf-8') as f:
    text = f.read()

# For async/await fetch
text = re.sub(
    r'(const res = await fetch\([^;]+;)\s*if \(!res\.ok\) throw new Error\(([^)]+)\);',
    r'\1\n          if (!res.ok) { console.error("API Error (Status " + res.status + "):", \2); throw new Error(\2); }',
    text
)

text = re.sub(
    r'(const res = await fetch\([^;]+;)\s*if \(res\.ok\) \{',
    r'\1\n          if (!res.ok) { console.error("API Error: Status " + res.status); }\n          if (res.ok) {',
    text
)

# Replace 'return r.json()' or 'return r.json().then(...)' to log if !r.ok
def fix_then(match):
    body = match.group(1)
    if '!r.ok' not in body and 'r.ok' in body:
        return match.group(0).replace('return {ok:r.ok', 'if(!r.ok) console.error("API Error: Status", r.status); return {ok:r.ok')
    if '!r.ok' not in body and 'r.ok' not in body:
        # e.g. return r.json()
        return match.group(0).replace('return r.json();', 'if(!r.ok) console.error("API Error: Status", r.status); return r.json();')
    return match.group(0)

text = re.sub(r'\.then\(function\(r\)\{\s*(.*?)\s*\}\)', fix_then, text)

with open('public/index.html', 'w', encoding='utf-8') as f:
    f.write(text)
print("Updated index.html")
