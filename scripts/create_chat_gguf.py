import struct
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: create_chat_gguf.py <output_path>")
        sys.exit(1)
        
    output_path = sys.argv[1]

    # Helper functions
    def pack_str(s):
        encoded = s.encode('utf-8')
        return struct.pack('<Q', len(encoded)) + encoded

    def pack_kv(key, val_type, val_bytes):
        return pack_str(key) + struct.pack('<I', val_type) + val_bytes

    def pack_array(elem_type, items):
        # Array format in GGUF: element_type (uint32) + length (uint64) + items
        header = struct.pack('<IQ', elem_type, len(items))
        return header + b"".join(items)

    # 1. Build Vocab tokens array
    # Types: 8 = STRING, 9 = ARRAY
    tokens = [
        " The",      # 0
        " robot",    # 1
        " learns",   # 2
        " to",       # 3
        " code",     # 4
        " is",       # 5
        " awesome",  # 6
        " and",      # 7
        " smart",    # 8
        " very",     # 9
        " fast",     # 10
        " help",     # 11
        " hello",    # 12
        " hello!",   # 13
        " thank",    # 14
        " you"       # 15
    ]
    
    token_bytes = [pack_str(t) for t in tokens]
    vocab_kv = pack_kv("tokenizer.ggml.tokens", 9, pack_array(8, token_bytes))

    # Other metadata
    kvs = [
        pack_kv("general.name", 8, pack_str("DeepChat-2L")),
        pack_kv("llama.block_count", 4, struct.pack('<I', 2)),
        pack_kv("general.alignment", 4, struct.pack('<I', 32)),
        vocab_kv
    ]
    metadata_bytes = b"".join(kvs)

    # Dimensions
    dim = 64
    vocab_size = 16
    
    # 2. Build Tensors description
    # 2 layers. In each layer l (0 and 1):
    # - "w_q_l" (shape [64, 64], Q4_0)
    # - "w_k_l" (shape [64, 64], Q4_0)
    # - "w_v_l" (shape [64, 64], Q4_0)
    # - "w_o_l" (shape [64, 64], Q4_0)
    # - "w_gate_l" (shape [64, 64], Q4_0)
    # - "w_up_l" (shape [64, 64], Q4_0)
    # - "w_down_l" (shape [64, 64], Q4_0)
    # - "gamma_attn_l" (shape [64], FP32)
    # - "gamma_mlp_l" (shape [64], FP32)
    # Final layer:
    # - "gamma_final" (shape [64], FP32)
    # - "w_logits" (shape [64, 16], FP32)

    tensor_list = []
    
    # Q4_0 weights count: 14 tensors of shape [64, 64]
    # FP32 weights count: 4 tensors of shape [64] + 1 tensor of shape [64, 16]
    
    q4_tensors = [
        "w_q_0", "w_k_0", "w_v_0", "w_o_0", "w_gate_0", "w_up_0", "w_down_0",
        "w_q_1", "w_k_1", "w_v_1", "w_o_1", "w_gate_1", "w_up_1", "w_down_1"
    ]
    
    fp32_tensors = [
        ("gamma_attn_0", [64]), ("gamma_mlp_0", [64]),
        ("gamma_attn_1", [64]), ("gamma_mlp_1", [64]),
        ("gamma_final", [64]),
        ("w_logits", [64, 16])
    ]

    current_offset = 0
    
    # Pack Q4_0 tensors info
    for name in q4_tensors:
        tensor_list.append(pack_str(name))
        tensor_list.append(struct.pack('<I', 2)) # ndim
        tensor_list.append(struct.pack('<QQ', 64, 64)) # dims
        tensor_list.append(struct.pack('<I', 2)) # type = Q4_0
        tensor_list.append(struct.pack('<Q', current_offset))
        current_offset += 2304 # (64*64/32) * 18 = 2304 bytes each

    # Pack FP32 tensors info
    for name, dims in fp32_tensors:
        ndim = len(dims)
        tensor_list.append(pack_str(name))
        tensor_list.append(struct.pack('<I', ndim))
        if ndim == 1:
            tensor_list.append(struct.pack('<Q', dims[0]))
            size = dims[0] * 4
        else:
            tensor_list.append(struct.pack('<QQ', dims[0], dims[1]))
            size = dims[0] * dims[1] * 4
            
        tensor_list.append(struct.pack('<I', 0)) # type = FP32
        tensor_list.append(struct.pack('<Q', current_offset))
        current_offset += size

    tensor_info_bytes = b"".join(tensor_list)

    # Alignment padding
    header_size = 4 + 4 + 8 + 8
    total_headers_size = header_size + len(metadata_bytes) + len(tensor_info_bytes)
    alignment = 32
    aligned_size = (total_headers_size + alignment - 1) // alignment * alignment
    padding_len = aligned_size - total_headers_size
    padding_bytes = b"\x00" * padding_len

    # 3. Build Tensors Binary Data
    tensor_data = b""

    # Q4_0 blocks: Write scale factor d = 0.1f (packed as half-float), qs = 0x33
    for _ in range(len(q4_tensors)):
        for _ in range(128):
            block = struct.pack('<e', 0.1) + b"\x33" * 16
            tensor_data += block

    # FP32 vectors (gamma_attn, gamma_mlp, gamma_final): 1.0f
    for _ in range(5):
        tensor_data += struct.pack('<f', 1.0) * 64

    # w_logits matrix: shape [64, 16]. Fill with 0.05f to simulate output weights
    tensor_data += struct.pack('<f', 0.05) * (64 * 16)

    # 4. Write GGUF file
    magic = b"GGUF"
    version = 3
    tensor_count = len(q4_tensors) + len(fp32_tensors)
    metadata_count = len(kvs)

    header = struct.pack('<4sIQQ', magic, version, tensor_count, metadata_count)

    with open(output_path, "wb") as f:
        f.write(header)
        f.write(metadata_bytes)
        f.write(tensor_info_bytes)
        f.write(padding_bytes)
        f.write(tensor_data)

    print(f"Chat GGUF model written to {output_path}")

if __name__ == "__main__":
    main()
