import struct

def read_str(data, offset):
    length = struct.unpack_from('<Q', data, offset)[0]
    offset += 8
    s = data[offset:offset+length].decode('utf-8')
    offset += length
    return s, offset

def main():
    path = "qwen2.5-1.5b-instruct-q4_0.gguf"
    with open(path, "rb") as f:
        data = f.read(1024 * 1024 * 10)

    magic = data[:4]
    version, tensor_count, kv_count = struct.unpack_from('<IQQ', data, 4)
    
    offset = 24
    for _ in range(kv_count):
        key, offset = read_str(data, offset)
        val_type = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        if val_type == 0: offset += 1
        elif val_type == 1: offset += 1
        elif val_type == 2: offset += 2
        elif val_type == 3: offset += 2
        elif val_type == 4: offset += 4
        elif val_type == 5: offset += 5 # wait, int32 is 4
        elif val_type == 6: offset += 4
        elif val_type == 7: offset += 1
        elif val_type == 8: _, offset = read_str(data, offset)
        elif val_type == 9:
            array_type = struct.unpack_from('<I', data, offset)[0]
            array_len = struct.unpack_from('<Q', data, offset+4)[0]
            offset += 12
            for _ in range(array_len):
                if array_type == 8: _, offset = read_str(data, offset)
                elif array_type in [0, 1, 7]: offset += 1
                elif array_type in [2, 3]: offset += 2
                elif array_type in [4, 5, 6]: offset += 4
                else: offset += 8
        else:
            offset += 8

    # Read Tensors Info
    biases = []
    for i in range(tensor_count):
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
        
        if "bias" in name:
            biases.append((name, dims, ttype))

    print(f"Total bias tensors found: {len(biases)}")
    # Print the unique suffixes of bias tensors
    suffixes = set()
    for name, dims, ttype in biases:
        parts = name.split(".")
        if len(parts) >= 3:
            suffixes.add(".".join(parts[2:]))
        else:
            suffixes.add(name)
    print("Unique bias names:", suffixes)
    for name, dims, ttype in biases[:10]:
        print(f"  {name}: dims={dims}, type={ttype}")

if __name__ == "__main__":
    main()
