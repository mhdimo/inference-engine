import struct
import sys

def read_str(data, offset):
    length = struct.unpack_from('<Q', data, offset)[0]
    offset += 8
    s = data[offset:offset+length].decode('utf-8')
    offset += length
    return s, offset

def main():
    path = "build/Qwen3.5-2B-UD-Q8_K_XL.gguf"
    # Read the first 20MB of header/metadata
    with open(path, "rb") as f:
        data = f.read(1024 * 1024 * 20)

    magic = data[:4]
    if magic != b"GGUF":
        print(f"Not a GGUF file: {magic}")
        return
        
    version, tensor_count, kv_count = struct.unpack_from('<IQQ', data, 4)
    print(f"GGUF Version: {version}, Tensors: {tensor_count}, KV Metadata keys: {kv_count}")
    
    offset = 24
    for _ in range(kv_count):
        try:
            key, offset = read_str(data, offset)
            val_type = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            val_str = ""
            if val_type == 0:
                val_str = str(struct.unpack_from('<B', data, offset)[0])
                offset += 1
            elif val_type == 1:
                val_str = str(struct.unpack_from('<b', data, offset)[0])
                offset += 1
            elif val_type == 2:
                val_str = str(struct.unpack_from('<H', data, offset)[0])
                offset += 2
            elif val_type == 3:
                val_str = str(struct.unpack_from('<h', data, offset)[0])
                offset += 2
            elif val_type == 4:
                val_str = str(struct.unpack_from('<I', data, offset)[0])
                offset += 4
            elif val_type == 5:
                val_str = str(struct.unpack_from('<i', data, offset)[0])
                offset += 4
            elif val_type == 6:
                val_str = str(struct.unpack_from('<f', data, offset)[0])
                offset += 4
            elif val_type == 7:
                val_str = str(struct.unpack_from('<?', data, offset)[0])
                offset += 1
            elif val_type == 8:
                val_str, offset = read_str(data, offset)
            elif val_type == 9:
                array_type = struct.unpack_from('<I', data, offset)[0]
                array_len = struct.unpack_from('<Q', data, offset+4)[0]
                offset += 12
                val_str = f"Array[{array_type}] len={array_len}"
                for _ in range(array_len):
                    if array_type == 8: _, offset = read_str(data, offset)
                    elif array_type in [0, 1, 7]: offset += 1
                    elif array_type in [2, 3]: offset += 2
                    elif array_type in [4, 5, 6]: offset += 4
                    else: offset += 8
            else:
                offset += 8
            
            if "architecture" in key or "name" in key:
                print(f"  {key} = {val_str}")
        except Exception as e:
            print("Error parsing metadata KV:", e)
            break

    print("\nListing all blk.0.* tensors:")
    for i in range(tensor_count):
        try:
            name, offset = read_str(data, offset)
            ndim = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            dims = []
            for _ in range(ndim):
                dims.append(struct.unpack_from('<Q', data, offset)[0])
                offset += 8
            ttype = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            toffset = struct.unpack_from('<Q', data, offset)[0]
            offset += 8
            
            if name.startswith("blk.0."):
                print(f"  {name}: dims={dims}, type={ttype}")
        except Exception as e:
            print("Error parsing tensor metadata:", e)
            break

if __name__ == "__main__":
    main()
