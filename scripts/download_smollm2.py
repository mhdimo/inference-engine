import os
import urllib.request
import sys

def main():
    # Target: Qwen2.5-1.5B-Instruct, plain Q4_0 quant (GGML tensor type 2).
    # The engine dequantizes types {0 F32, 1 F16, 2 Q4_0, 3 Q4_1, 8 Q8_0, 14 Q6_K};
    # this file's only non-Q4_0 tensor is the Q6_K output.weight, which is now supported.
    target_path = "qwen2.5-1.5b-instruct-q4_0.gguf"
    url = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_0.gguf"

    if os.path.exists(target_path):
        print(f"[CACHE] Model weights already exist locally at {target_path}")
        sys.exit(0)

    print(f"[DOWN] Downloading Qwen2.5-1.5B-Instruct (Q4_0) model weights (approx 1.0GB)...")

    req = urllib.request.Request(
        url,
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    )

    try:
        with urllib.request.urlopen(req) as response, open(target_path, 'wb') as out_file:
            total_size = int(response.info().get('Content-Length', 0))
            downloaded = 0
            block_size = 1024 * 64

            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                percent = min(100.0, (downloaded / total_size) * 100.0) if total_size > 0 else 0
                sys.stdout.write(f"\r[DOWN] Progress: {percent:.1f}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)")
                sys.stdout.flush()

        print(f"\n[DOWN] Successfully downloaded weights to {target_path}")
    except Exception as e:
        print(f"\n[ERR] Failed to download model weights: {e}")
        if os.path.exists(target_path):
            os.remove(target_path)
        sys.exit(1)

if __name__ == "__main__":
    main()
