import struct
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: create_mock_gguf.py <output_path>")
        sys.exit(1)
        
    output_path = sys.argv[1]

    # Helper functions
    def pack_str(s):
        encoded = s.encode('utf-8')
        return struct.pack('<Q', len(encoded)) + encoded

    def pack_kv(key, val_type, val_bytes):
        return pack_str(key) + struct.pack('<I', val_type) + val_bytes

    # 1. Build Metadata
    # Types: 4 = UINT32, 8 = STRING
    kvs = []
    kvs.append(pack_kv("general.name", 8, pack_str("Mock-LLaMA")))
    kvs.append(pack_kv("llama.block_count", 4, struct.pack('<I', 1)))
    kvs.append(pack_kv("general.alignment", 4, struct.pack('<I', 32)))
    
    metadata_bytes = b"".join(kvs)

    # 2. Build Tensor Info
    # Tensor 1: "w_q", shape [64, 64], type 2 (Q4_0), offset 0
    # Tensor 2: "w_o", shape [64, 64], type 0 (FP32), offset 2560
    tensors = []
    
    # Tensor 1 Info
    tensors.append(pack_str("w_q"))
    tensors.append(struct.pack('<I', 2)) # ndim
    tensors.append(struct.pack('<QQ', 64, 64)) # dimensions
    tensors.append(struct.pack('<I', 2)) # ggml_type: 2 = Q4_0
    tensors.append(struct.pack('<Q', 0)) # offset
    
    # Tensor 2 Info
    tensors.append(pack_str("w_o"))
    tensors.append(struct.pack('<I', 2)) # ndim
    tensors.append(struct.pack('<QQ', 64, 64)) # dimensions
    tensors.append(struct.pack('<I', 0)) # ggml_type: 0 = FP32
    tensors.append(struct.pack('<Q', 2304)) # offset

    tensor_info_bytes = b"".join(tensors)

    # Calculate current header + info size to align the tensor data block
    header_size = 4 + 4 + 8 + 8 # magic (4) + version (4) + tensor_count (8) + metadata_kv_count (8) = 24 bytes
    total_header_and_infos_size = header_size + len(metadata_bytes) + len(tensor_info_bytes)
    
    alignment = 32
    aligned_size = (total_header_and_infos_size + alignment - 1) // alignment * alignment
    padding_len = aligned_size - total_header_and_infos_size
    padding_bytes = b"\x00" * padding_len

    # 3. Build Tensor Data
    # w_q data: Q4_0 blocks of size 18 bytes each. Total blocks = 64 * 64 / 32 = 128 blocks.
    # Write: scale (d) = 1.0f (half-precision float), qs = 16 bytes of 0x33 (dequantizes to 3.0f)
    w_q_data = b""
    for _ in range(128):
        block_bytes = struct.pack('<e', 1.0) + b"\x33" * 16
        w_q_data += block_bytes

    # w_o data: FP32 elements. Total elements = 64 * 64 = 4096 floats.
    # Write: 4096 floats of value 1.5f
    w_o_data = b""
    for _ in range(4096):
        w_o_data += struct.pack('<f', 1.5)

    # 4. Assemble the whole file
    magic = b"GGUF"
    version = 3
    tensor_count = 2
    metadata_count = len(kvs)

    header = struct.pack('<4sIQQ', magic, version, tensor_count, metadata_count)

    with open(output_path, "wb") as f:
        f.write(header)
        f.write(metadata_bytes)
        f.write(tensor_info_bytes)
        f.write(padding_bytes)
        f.write(w_q_data)
        f.write(w_o_data)

    print(f"Mock GGUF file successfully written to {output_path}")

if __name__ == "__main__":
    main()
