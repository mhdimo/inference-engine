#include "gguf_loader.hpp"
#include "../core/logger.hpp"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>

namespace core {

enum class GGUFValueType : uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
};

GGUFLoader::GGUFLoader(const std::string& filepath) : filepath_(filepath) {}

GGUFLoader::~GGUFLoader() {
    if (mmapped_data_ && mmapped_data_ != MAP_FAILED) {
        munmap(mmapped_data_, file_size_);
    }
    if (fd_ != -1) {
        close(fd_);
    }
}

// Inline helper templates to read primitive values from binary memory pointer
template <typename T>
inline T read_val(const char*& ptr) {
    T val;
    std::memcpy(&val, ptr, sizeof(T));
    ptr += sizeof(T);
    return val;
}

inline std::string read_str(const char*& ptr) {
    uint64_t len = read_val<uint64_t>(ptr);
    std::string s(ptr, len);
    ptr += len;
    return s;
}

// Function to skip array value fields in metadata
void skip_array(const char*& ptr) {
    GGUFValueType type = static_cast<GGUFValueType>(read_val<uint32_t>(ptr));
    uint64_t len = read_val<uint64_t>(ptr);
    for (uint64_t i = 0; i < len; ++i) {
        if (type == GGUFValueType::STRING) {
            uint64_t slen = read_val<uint64_t>(ptr);
            ptr += slen;
        } else if (type == GGUFValueType::ARRAY) {
            skip_array(ptr);
        } else {
            // Primitive types
            size_t size = 0;
            switch (type) {
                case GGUFValueType::UINT8:  case GGUFValueType::INT8:  case GGUFValueType::BOOL: size = 1; break;
                case GGUFValueType::UINT16: case GGUFValueType::INT16:                           size = 2; break;
                case GGUFValueType::UINT32: case GGUFValueType::INT32: case GGUFValueType::FLOAT32: size = 4; break;
                case GGUFValueType::UINT64: case GGUFValueType::INT64: case GGUFValueType::FLOAT64: size = 8; break;
                default: break;
            }
            ptr += size;
        }
    }
}

bool GGUFLoader::load() {
    fd_ = open(filepath_.c_str(), O_RDONLY);
    if (fd_ == -1) {
        LOG_ERR("Failed to open GGUF file: " + filepath_);
        return false;
    }

    struct stat sb;
    if (fstat(fd_, &sb) == -1) {
        LOG_ERR("Failed to get file stats for GGUF: " + filepath_);
        return false;
    }
    file_size_ = sb.st_size;

    mmapped_data_ = mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (mmapped_data_ == MAP_FAILED) {
        LOG_ERR("Failed to mmap GGUF file: " + filepath_);
        return false;
    }

    const char* start_ptr = static_cast<const char*>(mmapped_data_);
    const char* ptr = start_ptr;

    // 1. Read and parse header
    uint32_t magic = read_val<uint32_t>(ptr);
    if (magic != 0x46554747) { // "GGUF" in ASCII hex little-endian
        LOG_ERR("Invalid GGUF magic bytes");
        return false;
    }

    version_ = read_val<uint32_t>(ptr);
    if (version_ != 2 && version_ != 3) {
        LOG_ERR("Unsupported GGUF version: " + std::to_string(version_));
        return false;
    }

    tensor_count_ = read_val<uint64_t>(ptr);
    metadata_kv_count_ = read_val<uint64_t>(ptr);

    // 2. Parse metadata key-value pairs
    uint32_t alignment = 32; // Default GGUF alignment limit
    for (uint64_t i = 0; i < metadata_kv_count_; ++i) {
        std::string key = read_str(ptr);
        GGUFValueType type = static_cast<GGUFValueType>(read_val<uint32_t>(ptr));

        if (type == GGUFValueType::STRING) {
            std::string val = read_str(ptr);
            metadata_strings_[key] = val;
        } else if (type == GGUFValueType::UINT32) {
            uint32_t val = read_val<uint32_t>(ptr);
            metadata_uint32s_[key] = val;
            if (key == "general.alignment") {
                alignment = val;
            }
        } else if (type == GGUFValueType::INT32) {
            int32_t val = read_val<int32_t>(ptr);
            metadata_uint32s_[key] = static_cast<uint32_t>(val);
        } else if (type == GGUFValueType::FLOAT32) {
            float val = read_val<float>(ptr);
            metadata_floats_[key] = val;
        } else if (type == GGUFValueType::BOOL || type == GGUFValueType::UINT8 || type == GGUFValueType::INT8 ||
                   type == GGUFValueType::UINT16 || type == GGUFValueType::INT16) {
            size_t size = (type == GGUFValueType::UINT16 || type == GGUFValueType::INT16) ? 2 : 1;
            uint32_t val = 0;
            std::memcpy(&val, ptr, size);
            ptr += size;
            metadata_uint32s_[key] = val;
        } else if (type == GGUFValueType::ARRAY) {
            GGUFValueType elem_type = static_cast<GGUFValueType>(read_val<uint32_t>(ptr));
            uint64_t len = read_val<uint64_t>(ptr);
            if (key == "tokenizer.ggml.tokens" && elem_type == GGUFValueType::STRING) {
                vocabulary_.resize(len);
                for (uint64_t idx = 0; idx < len; ++idx) {
                    vocabulary_[idx] = read_str(ptr);
                }
            } else if (key == "tokenizer.ggml.merges" && elem_type == GGUFValueType::STRING) {
                merges_.resize(len);
                for (uint64_t idx = 0; idx < len; ++idx) {
                    merges_[idx] = read_str(ptr);
                }
            } else {
                // Manually skip elements of the array
                for (uint64_t idx = 0; idx < len; ++idx) {
                    if (elem_type == GGUFValueType::STRING) {
                        uint64_t slen = read_val<uint64_t>(ptr);
                        ptr += slen;
                    } else if (elem_type == GGUFValueType::ARRAY) {
                        skip_array(ptr);
                    } else {
                        size_t size = 0;
                        switch (elem_type) {
                            case GGUFValueType::UINT8:  case GGUFValueType::INT8:  case GGUFValueType::BOOL: size = 1; break;
                            case GGUFValueType::UINT16: case GGUFValueType::INT16:                           size = 2; break;
                            case GGUFValueType::UINT32: case GGUFValueType::INT32: case GGUFValueType::FLOAT32: size = 4; break;
                            case GGUFValueType::UINT64: case GGUFValueType::INT64: case GGUFValueType::FLOAT64: size = 8; break;
                            default: break;
                        }
                        ptr += size;
                    }
                }
            }
        } else {
            // Skip other types (we read their corresponding sizes)
            size_t size = 0;
            switch (type) {
                case GGUFValueType::UINT8:  case GGUFValueType::INT8:  case GGUFValueType::BOOL: size = 1; break;
                case GGUFValueType::UINT16: case GGUFValueType::INT16:                           size = 2; break;
                case GGUFValueType::UINT32: case GGUFValueType::INT32: case GGUFValueType::FLOAT32: size = 4; break;
                case GGUFValueType::UINT64: case GGUFValueType::INT64: case GGUFValueType::FLOAT64: size = 8; break;
                default: break;
            }
            ptr += size;
        }
    }

    // 3. Parse tensor info block headers
    for (uint64_t i = 0; i < tensor_count_; ++i) {
        GGUFTensorInfo info;
        info.name = read_str(ptr);
        info.ndim = read_val<uint32_t>(ptr);
        info.dims.resize(info.ndim);
        uint64_t elements = 1;
        for (uint32_t d = 0; d < info.ndim; ++d) {
            info.dims[d] = read_val<uint64_t>(ptr);
            elements *= info.dims[d];
        }
        info.type = read_val<uint32_t>(ptr);
        info.offset = read_val<uint64_t>(ptr);

        // Compute size in bytes based on ggml type
        // ggml_type: 0=FP32 (4 bytes), 1=FP16 (2 bytes), 2=Q4_0 (20 bytes per 32 elements)
        if (info.type == 0) {
            info.size_bytes = elements * sizeof(float);
        } else if (info.type == 1) {
            info.size_bytes = elements * 2; // FP16
        } else if (info.type == 2) {
            // Q4_0 block size 32 elements takes 18 bytes
            info.size_bytes = (elements / 32) * 18;
        } else if (info.type == 3) {
            // Q4_1 block size 32 elements takes 24 bytes
            info.size_bytes = (elements / 32) * 24;
        } else if (info.type == 8) {
            // Q8_0 block size 32 elements takes 34 bytes
            info.size_bytes = (elements / 32) * 34;
        } else if (info.type == 14) {
            // Q6_K block: 256 elements take 210 bytes (ql[128] + qh[64] + scales[16] + fp16 d[2])
            info.size_bytes = (elements / 256) * 210;
        } else {
            LOG_ERR("Unsupported GGUF tensor type: " + std::to_string(info.type) + " for " + info.name);
            throw std::runtime_error("Unsupported GGUF tensor data type: " + std::to_string(info.type));
        }

        tensor_infos_.push_back(info);
    }

    // 4. Align tensor binary data block offset
    uint64_t raw_offset = ptr - start_ptr;
    tensor_data_offset_ = (raw_offset + alignment - 1) / alignment * alignment;

    return true;
}

std::string GGUFLoader::get_metadata_string(const std::string& key) const {
    auto it = metadata_strings_.find(key);
    return (it != metadata_strings_.end()) ? it->second : "";
}

uint32_t GGUFLoader::get_metadata_uint32(const std::string& key) const {
    auto it = metadata_uint32s_.find(key);
    return (it != metadata_uint32s_.end()) ? it->second : 0;
}

float GGUFLoader::get_metadata_float(const std::string& key) const {
    auto it = metadata_floats_.find(key);
    return (it != metadata_floats_.end()) ? it->second : 0.0f;
}

bool GGUFLoader::has_metadata(const std::string& key) const {
    return metadata_strings_.count(key) || metadata_uint32s_.count(key) || metadata_floats_.count(key);
}

const GGUFTensorInfo* GGUFLoader::get_tensor_info(const std::string& name) const {
    for (const auto& info : tensor_infos_) {
        if (info.name == name) {
            return &info;
        }
    }
    return nullptr;
}

const void* GGUFLoader::get_tensor_data(const GGUFTensorInfo& info) const {
    const char* start_ptr = static_cast<const char*>(mmapped_data_);
    return start_ptr + tensor_data_offset_ + info.offset;
}

} // namespace core
