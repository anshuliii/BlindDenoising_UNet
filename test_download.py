import urllib.request

print("Trying to download BSD300...")
try:
    urllib.request.urlretrieve(
        "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz",
        "test.tgz"
    )
    print("SUCCESS - download worked!")
except Exception as e:
    print(f"FAILED - {e}")