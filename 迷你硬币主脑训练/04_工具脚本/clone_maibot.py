import os, subprocess
dst = os.path.expanduser("~/Desktop/maibot_source")
subprocess.run(["git", "clone", "--depth=1",
    "https://github.com/Mai-with-u/MaiBot.git", dst])
print(f"克隆到 {dst}")
