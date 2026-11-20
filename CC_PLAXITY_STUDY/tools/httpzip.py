"""Read members of a remote zip via HTTP range requests (only the bytes needed)."""
import io, sys, zipfile, urllib.request
class HttpFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url; self.pos = 0
        req = urllib.request.Request(url, method="HEAD"); r = urllib.request.urlopen(req)
        self.size = int(r.headers["Content-Length"]); self.url = r.url
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off); return self.pos
    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        if n == 0 or self.pos >= self.size: return b""
        end = min(self.size, self.pos + n) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}", "User-Agent": "research-script"})
        d = urllib.request.urlopen(req).read(); self.pos += len(d); return d
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)
def open_zip(url): return zipfile.ZipFile(io.BufferedReader(HttpFile(url), buffer_size=1 << 20))
if __name__ == "__main__":
    z = open_zip(sys.argv[1]); names = z.infolist()
    print(len(names)); [print(i.filename, i.file_size, i.compress_type) for i in names[:int(sys.argv[2]) if len(sys.argv) > 2 else 15]]
