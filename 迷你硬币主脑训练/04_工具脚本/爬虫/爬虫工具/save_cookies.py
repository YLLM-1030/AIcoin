"""保存 YouTube cookies 到文件"""
import browser_cookie3

cj = browser_cookie3.chrome(domain_name='youtube.com')
with open(r'C:\Users\Autogram-coin\Desktop\新建训练\youtube_cookies.txt', 'w') as f:
    f.write('# Netscape HTTP Cookie File\n')
    for c in cj:
        if 'youtube' in c.domain or 'ytimg' in c.domain:
            secure = "TRUE" if c.secure else "FALSE"
            expires = str(int(c.expires)) if c.expires else "0"
            f.write(f'{c.domain}\tTRUE\t{c.path}\t{secure}\t{expires}\t{c.name}\t{c.value}\n')
print(f"已保存 {len(cj)} 个 cookies")
