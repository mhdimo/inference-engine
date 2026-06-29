#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstddef>

namespace core {

enum class GGUFType : uint32_t {
    FP32 = 0,
    FP16 = 1,
    Q4_0 = 2,
};

struct GGUFTensorInfo {
    std::string name;
    uint32_t ndim;
    std::vector<uint64_t> dims;
    uint32_t type;       // ggml_type (0=FP32, 1=FP16, 2=Q4_0)
    uint64_t offset;     // Offset relative to the start of the tensor data block
    uint64_t size_bytes; // Size of tensor data block in bytes
};

class GGUFLoader {
public:
    GGUFLoader(const std::string& filepath);
    ~GGUFLoader();

    // Map file using mmap and parse metadata/tensor headers
    bool load();

    // Retrieve metadata values
    std::string get_metadata_string(const std::string& key) const;
    uint32_t get_metadata_uint32(const std::string& key) const;
    float get_metadata_float(const std::string& key) const;
    bool has_metadata(const std::string& key) const;

    // Get loaded tensor info and direct mapped memory pointer
    const std::vector<GGUFTensorInfo>& tensors() const { return tensor_infos_; }
    const GGUFTensorInfo* get_tensor_info(const std::string& name) const;
    const void* get_tensor_data(const GGUFTensorInfo& info) const;

    // Get loaded vocabulary array
    const std::vector<std::string>& vocabulary() const { return vocabulary_; }
    const std::vector<std::string>& merges() const { return merges_; }

private:
    std::string filepath_;
    int fd_ = -1;
    void* mmapped_data_ = nullptr;
    size_t file_size_ = 0;

    uint32_t version_ = 0;
    uint64_t tensor_count_ = 0;
    uint64_t metadata_kv_count_ = 0;
    uint64_t tensor_data_offset_ = 0; // File offset where the binary data starts

    std::unordered_map<std::string, std::string> metadata_strings_;
    std::unordered_map<std::string, uint32_t> metadata_uint32s_;
    std::unordered_map<std::string, float> metadata_floats_;
    std::vector<std::string> vocabulary_;
    std::vector<std::string> merges_;
    std::vector<GGUFTensorInfo> tensor_infos_;
};

} // namespace core
