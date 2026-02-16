"""Convert JSON cookies (from browser extension) to Netscape cookies.txt format."""
import json
import sys
import os

def convert_json_to_netscape(json_path, output_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cookies = data.get('cookies', data) if isinstance(data, dict) else data
    
    lines = [
        '# Netscape HTTP Cookie File',
        '# https://curl.haxx.se/rfc/cookie_spec.html',
        '# This is a generated file!  Do not edit.',
        ''
    ]
    for c in cookies:
        domain = c['domain']
        flag = 'TRUE' if domain.startswith('.') else 'FALSE'
        path = c.get('path', '/')
        secure = 'TRUE' if c.get('secure', False) else 'FALSE'
        exp = str(int(c.get('expirationDate', 0)))
        name = c['name']
        value = c['value']
        lines.append(f'{domain}\t{flag}\t{path}\t{secure}\t{exp}\t{name}\t{value}')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    
    print(f'Converted {len(cookies)} cookies to {output_path}')

if __name__ == '__main__':
    json_path = sys.argv[1] if len(sys.argv) > 1 else 'cookies_raw.json'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'cookies.txt'
    convert_json_to_netscape(json_path, output_path)
