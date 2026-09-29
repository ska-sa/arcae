#include <cstddef>
#include <string>
#include <thread>
#include <vector>

#include <casacore/casa/Containers/Record.h>
#include <casacore/casa/Json/JsonKVMap.h>
#include <casacore/casa/Json/JsonParser.h>

#include <gtest/gtest.h>

using ::casacore::JsonParser;
using ::casacore::Record;

// casacore's JSON parser is a flex scanner and bison parser whose input,
// position and result are process wide statics. arcae parses JSON on its
// isolation threads, so concurrent parses must be serialised
// (vcpkg/overlay-ports/casacore/002-json-parser-mutex.patch).
TEST(JsonParserTest, ConcurrentParse) {
  constexpr std::size_t kThreads = 16;
  constexpr std::size_t kIterations = 500;
  std::vector<std::size_t> failures(kThreads, 0);
  std::vector<std::thread> threads;

  for (std::size_t t = 0; t < kThreads; ++t) {
    threads.emplace_back([t, &failures]() {
      // Distinct, non-trivial documents per thread, so that one thread
      // reading another's input or result is detectable
      auto name = "thread_" + std::to_string(t);
      auto json = R"({"name": ")" + name + R"(", "id": )" + std::to_string(t) +
                  R"(, "values": [1, 2, 3], "nested": {"option": "user"}})";

      for (std::size_t i = 0; i < kIterations; ++i) {
        try {
          auto record = JsonParser::parse(json).toRecord();
          if (std::string(record.asString("name")) != name ||
              record.asInt64("id") != static_cast<casacore::Int64>(t) ||
              std::string(record.asRecord("nested").asString("option")) != "user") {
            ++failures[t];
          }
        } catch (const std::exception&) {
          ++failures[t];
        }
      }
    });
  }

  for (auto& thread : threads) thread.join();
  for (std::size_t t = 0; t < kThreads; ++t) {
    EXPECT_EQ(failures[t], 0u) << "thread " << t;
  }
}
