import struct
import sys

def main():
    path = "SmollM2-135M-Instruct-Q4_0.gguf"
    print(f"Opening {path}...")
    
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            print("Invalid magic")
            return
            
        version = struct.unpack("<I", f.read(4))[0]
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        metadata_kv_count = struct.unpack("<Q", f.read(8))[0]
        
        print(f"Version: {version}, Tensors: {tensor_count}, Metadata KVs: {metadata_kv_count}")
        
        def read_str():
            length = struct.unpack("<Q", f.read(8))[0]
            val = f.read(length)
            return val.decode('utf-8', errors='ignore')
            
        def skip_val(vtype):
            if vtype == 8: # STRING
                read_str()
            elif vtype == 9: # ARRAY
                elem_type = struct.unpack("<I", f.read(4))[0]
                length = struct.unpack("<Q", f.read(8))[0]
                for _ in range(length):
                    if elem_type == 8:
                        read_str()
                    else:
                        size = 0
                        if elem_type in [0, 1, 7]: # uint8, int8, bool
                            size = 1
                        elif elem_type in [2, 3]: # uint16, int16
                            size = 2
                        elif elem_type in [4, 5, 6]: # uint32, int32, float32
                            size = 4
                        elif elem_type in [10, 11, 12]: # uint64, int64, float64
                            size = 8
                        f.read(size)
            else:
                size = 0
                if vtype in [0, 1, 7]: # uint8, int8, bool
                    size = 1
                elif vtype in [2, 3]: # uint16, int16
                    size = 2
                elif vtype in [4, 5, 6]: # uint32, int32, float32
                    size = 4
                elif vtype in [10, 11, 12]: # uint64, int64, float64
                    size = 8
                f.read(size)
                
        for i in range(metadata_kv_count):
            key = read_str()
            vtype = struct.unpack("<I", f.read(4))[0]
            skip_val(vtype)
            
        print("\n--- Tensors ---")
        types_seen = set()
        for i in range(tensor_count):
            name = read_str()
            ndim = struct.unpack("<I", f.read(4))[0]
            dims = []
            for _ in range(ndim):
                dims.append(struct.unpack("<Q", f.read(8))[0])
            ttype = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            types_seen.add(ttype)
            if i < 15 or i >= tensor_count - 5:
                print(f"Tensor {i}: '{name}', ndim={ndim}, dims={dims}, type={ttype}, offset={offset}")
                
        print(f"\nAll tensor types present in GGUF file: {types_seen}")

if __name__ == "__main__":
    main()
