#pragma once
#include <string>

namespace core {

enum class LogLevel {
    INFO,
    WARNING,
    ERROR
};

void log(LogLevel level, const std::string& message);

#define LOG_INFO(msg) ::core::log(::core::LogLevel::INFO, msg)
#define LOG_WARN(msg) ::core::log(::core::LogLevel::WARNING, msg)
#define LOG_ERR(msg) ::core::log(::core::LogLevel::ERROR, msg)

} // namespace core
