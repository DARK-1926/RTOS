"""Stream one member (a sequence zip) out of the remote EuRoC archive via HTTP range requests, verify CRC, save to disk."""
import sys, os, zipfile, io, shutil
sys.path.insert(0, os.path.dirname(__file__))
from httpzip import HttpFile
outer, member, dst = sys.argv[1:4]
f = HttpFile(f"https://huggingface.co/datasets/GlowBond/EuRoC_MAV_Dataset/resolve/main/{outer}")
z = zipfile.ZipFile(io.BufferedReader(f, buffer_size=32 << 20))
with z.open(member) as src, open(dst + ".part", "wb") as out:   # zipfile checks the CRC at end of stream
    shutil.copyfileobj(src, out, 8 << 20)
os.replace(dst + ".part", dst); print("ok", dst, os.path.getsize(dst))
