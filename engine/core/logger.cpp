#include "logger.hpp"
#include <iostream>

namespace core {

void log(LogLevel level, const std::string& message) {
    switch (level) {
        case LogLevel::INFO:
            std::cout << "[INFO] " << message << std::endl;
            break;
        case LogLevel::WARNING:
            std::cout << "[WARN] " << message << std::endl;
            break;
        case LogLevel::ERROR:
            std::cerr << "[ERROR] " << message << std::endl;
            break;
    }
}

} // namespace core
