#pragma once

#include <string>
#include <vector>

namespace uffdejavu {
    class UDPipeline {
        public:
            UDPipeline(const std::string& vertFilepath, const std::string& fragFilepath);
            // ~UDPipeline();

            // UDPipeline(const UDPipeline&) = delete;
            // UDPipeline& operator=(const UDPipeline&) = delete;

        private:
            // void createGraphicsPipeline();
            static std::vector<char> readFile(const std::string& filepath);

            void createGraphicsPipeline(const std::string& vertFilepath, const std::string& fragFilepath);
    };
}