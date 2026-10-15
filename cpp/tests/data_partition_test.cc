#include <optional>

#include <casacore/casa/Arrays/IPosition.h>
#include <casacore/casa/Arrays/Slicer.h>
#include <casacore/casa/Utilities/DataType.h>

#include "arcae/data_partition.h"
#include "arcae/result_shape.h"
#include "arcae/selection.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <arrow/testing/gtest_util.h>

using ::arcae::detail::DataPartition;
using ::arcae::detail::IndexType;
using ::arcae::detail::ResultShapeData;
using ::arcae::detail::RowSplitOptions;
using ::arcae::detail::SelectionBuilder;
using ::arcae::detail::SpanPairs;

using ::casacore::DataType;
using ::casacore::IPosition;
using ::casacore::Slicer;

namespace {

TEST(DataPartitionTest, Fixed) {
  auto result_shape =
      ResultShapeData{"DATA", IPosition{2, 3, 4}, 3, DataType::TpComplex, std::nullopt};

  auto selection = SelectionBuilder()
                       .Order('F')                 // 1 range in dim 0
                       .Add({3, 0, 1})             // 2 ranges in dim 1
                       .Add({2, 1, -1, 5, 6, -1})  // 3 ranges in dim 2
                       .Build();                   // 1 x 2 x 3

  ASSERT_OK_AND_ASSIGN(auto partition, DataPartition::Make(selection, result_shape));
  EXPECT_EQ(partition.nChunks(), 4);

  EXPECT_THAT(partition.Chunk(0).Disk(0), ::testing::ElementsAre(0, 1));
  EXPECT_THAT(partition.Chunk(0).Mem(0), ::testing::ElementsAre(0, 1));
  EXPECT_THAT(partition.Chunk(0).Disk(1), ::testing::ElementsAre(0, 1));
  EXPECT_THAT(partition.Chunk(0).Mem(1), ::testing::ElementsAre(1, 2));
  EXPECT_THAT(partition.Chunk(0).Disk(2), ::testing::ElementsAre(-1, -1));
  EXPECT_THAT(partition.Chunk(0).Mem(2), ::testing::ElementsAre(2, 5));
  EXPECT_THAT(partition.Chunk(0).ReferenceRows().rowVector(),
              ::testing::ElementsAre(-1, -1));
  EXPECT_EQ(partition.Chunk(0).SectionSlicer(),
            Slicer(IPosition({0, 0}), IPosition({1, 1}), Slicer::endIsLast));

  EXPECT_THAT(partition.Chunk(1).Disk(0), ::testing::ElementsAre(0, 1));
  EXPECT_THAT(partition.Chunk(1).Mem(0), ::testing::ElementsAre(0, 1));
  EXPECT_THAT(partition.Chunk(1).Disk(1), ::testing::ElementsAre(0, 1));
  EXPECT_THAT(partition.Chunk(1).Mem(1), ::testing::ElementsAre(1, 2));
  EXPECT_THAT(partition.Chunk(1).Disk(2), ::testing::ElementsAre(1, 2, 5, 6));
  EXPECT_THAT(partition.Chunk(1).Mem(2), ::testing::ElementsAre(1, 0, 3, 4));
  EXPECT_THAT(partition.Chunk(1).ReferenceRows().rowVector(),
              ::testing::ElementsAre(1, 2, 5, 6));
  EXPECT_EQ(partition.data_chunks_[1].SectionSlicer(),
            Slicer(IPosition({0, 0}), IPosition({1, 1}), Slicer::endIsLast));

  EXPECT_THAT(partition.Chunk(2).Disk(0), ::testing::ElementsAre(0, 1));
  EXPECT_THAT(partition.Chunk(2).Mem(0), ::testing::ElementsAre(0, 1));
  EXPECT_THAT(partition.Chunk(2).Disk(1), ::testing::ElementsAre(3));
  EXPECT_THAT(partition.Chunk(2).Mem(1), ::testing::ElementsAre(0));
  EXPECT_THAT(partition.Chunk(2).Disk(2), ::testing::ElementsAre(-1, -1));
  EXPECT_THAT(partition.Chunk(2).Mem(2), ::testing::ElementsAre(2, 5));
  EXPECT_THAT(partition.Chunk(2).ReferenceRows().rowVector(),
              ::testing::ElementsAre(-1, -1));
  EXPECT_EQ(partition.Chunk(2).SectionSlicer(),
            Slicer(IPosition({0, 3}), IPosition({1, 3}), Slicer::endIsLast));

  EXPECT_THAT(partition.Chunk(3).Disk(0), ::testing::ElementsAre(0, 1));
  EXPECT_THAT(partition.Chunk(3).Mem(0), ::testing::ElementsAre(0, 1));
  EXPECT_THAT(partition.Chunk(3).Disk(1), ::testing::ElementsAre(3));
  EXPECT_THAT(partition.Chunk(3).Mem(1), ::testing::ElementsAre(0));
  EXPECT_THAT(partition.Chunk(3).Disk(2), ::testing::ElementsAre(1, 2, 5, 6));
  EXPECT_THAT(partition.Chunk(3).Mem(2), ::testing::ElementsAre(1, 0, 3, 4));
  EXPECT_THAT(partition.Chunk(3).ReferenceRows().rowVector(),
              ::testing::ElementsAre(1, 2, 5, 6));
  EXPECT_EQ(partition.Chunk(3).SectionSlicer(),
            Slicer(IPosition({0, 3}), IPosition({1, 3}), Slicer::endIsLast));
}

TEST(DataPartitionTest, Variable) {
  auto result_shape =
      ResultShapeData{"DATA", std::nullopt, 3, DataType::TpComplex,
                      std::vector<IPosition>{IPosition({3, 4}), IPosition({1, 4})}};

  auto selection = SelectionBuilder()
                       .Order('F')      // 1 range in dim 0 (varies per row)
                       .Add({3, 0, 1})  // 2 ranges in dim 1
                       .Add({-1, 0})    // 2 rows
                       .Build();

  ASSERT_OK_AND_ASSIGN(auto partition, DataPartition::Make(selection, result_shape));
  EXPECT_EQ(partition.nChunks(), 4);  // 1x1x1 + 1x2x2

  EXPECT_THAT(partition.Chunk(0).Disk(0), ::testing::ElementsAre(0, 1, 2));
  EXPECT_THAT(partition.Chunk(0).Mem(0), ::testing::ElementsAre(0, 1, 2));
  EXPECT_THAT(partition.Chunk(0).Disk(1), ::testing::ElementsAre(0, 1));
  EXPECT_THAT(partition.Chunk(0).Mem(1), ::testing::ElementsAre(1, 2));
  EXPECT_THAT(partition.Chunk(0).Disk(2), ::testing::ElementsAre(-1));
  EXPECT_THAT(partition.Chunk(0).Mem(2), ::testing::ElementsAre(0));
  EXPECT_EQ(partition.Chunk(0).RowSlicer(),
            Slicer(IPosition({-1}), IPosition({-1}), Slicer::endIsLast));
  EXPECT_EQ(partition.Chunk(0).SectionSlicer(),
            Slicer(IPosition({0, 0}), IPosition({2, 1}), Slicer::endIsLast));

  EXPECT_THAT(partition.Chunk(1).Disk(0), ::testing::ElementsAre(0, 1, 2));
  EXPECT_THAT(partition.Chunk(1).Mem(0), ::testing::ElementsAre(0, 1, 2));
  EXPECT_THAT(partition.Chunk(1).Disk(1), ::testing::ElementsAre(3));
  EXPECT_THAT(partition.Chunk(1).Mem(1), ::testing::ElementsAre(0));
  EXPECT_THAT(partition.Chunk(1).Disk(2), ::testing::ElementsAre(-1));
  EXPECT_THAT(partition.Chunk(1).Mem(2), ::testing::ElementsAre(0));
  EXPECT_EQ(partition.Chunk(1).RowSlicer(),
            Slicer(IPosition({-1}), IPosition({-1}), Slicer::endIsLast));
  EXPECT_EQ(partition.Chunk(1).SectionSlicer(),
            Slicer(IPosition({0, 3}), IPosition({2, 3}), Slicer::endIsLast));

  EXPECT_THAT(partition.Chunk(2).Disk(0), ::testing::ElementsAre(0));
  EXPECT_THAT(partition.Chunk(2).Mem(0), ::testing::ElementsAre(0));
  EXPECT_THAT(partition.Chunk(2).Disk(1), ::testing::ElementsAre(0, 1));
  EXPECT_THAT(partition.Chunk(2).Mem(1), ::testing::ElementsAre(1, 2));
  EXPECT_THAT(partition.Chunk(2).Disk(2), ::testing::ElementsAre(0));
  EXPECT_THAT(partition.Chunk(2).Mem(2), ::testing::ElementsAre(1));
  EXPECT_EQ(partition.Chunk(2).RowSlicer(),
            Slicer(IPosition({0}), IPosition({0}), Slicer::endIsLast));
  EXPECT_EQ(partition.Chunk(2).SectionSlicer(),
            Slicer(IPosition({0, 0}), IPosition({0, 1}), Slicer::endIsLast));

  EXPECT_THAT(partition.Chunk(3).Disk(0), ::testing::ElementsAre(0));
  EXPECT_THAT(partition.Chunk(3).Mem(0), ::testing::ElementsAre(0));
  EXPECT_THAT(partition.Chunk(3).Disk(1), ::testing::ElementsAre(3));
  EXPECT_THAT(partition.Chunk(3).Mem(1), ::testing::ElementsAre(0));
  EXPECT_THAT(partition.Chunk(3).Disk(2), ::testing::ElementsAre(0));
  EXPECT_THAT(partition.Chunk(3).Mem(2), ::testing::ElementsAre(1));
  EXPECT_EQ(partition.Chunk(3).RowSlicer(),
            Slicer(IPosition({0}), IPosition({0}), Slicer::endIsLast));
  EXPECT_EQ(partition.Chunk(3).SectionSlicer(),
            Slicer(IPosition({0, 3}), IPosition({0, 3}), Slicer::endIsLast));
}

// Each row of a {2, 3} complex cell is 2 x 3 x 8 = 48 bytes
static constexpr std::size_t kRowBytes = 2 * 3 * 8;

// Rows are split into at most max_splits chunks of consecutive rows,
// whose sizes differ by at most one row
TEST(DataPartitionTest, SplitRows) {
  auto result_shape =
      ResultShapeData{"DATA", IPosition{2, 3, 10}, 3, DataType::TpComplex, std::nullopt};
  auto selection = SelectionBuilder().Order('F').Build();

  ASSERT_OK_AND_ASSIGN(auto partition,
                       DataPartition::Make(selection, result_shape, RowSplitOptions{4}));
  ASSERT_EQ(partition.nChunks(), 4);
  EXPECT_THAT(partition.Chunk(0).Disk(2), ::testing::ElementsAre(0, 1));
  EXPECT_THAT(partition.Chunk(1).Disk(2), ::testing::ElementsAre(2, 3, 4));
  EXPECT_THAT(partition.Chunk(2).Disk(2), ::testing::ElementsAre(5, 6));
  EXPECT_THAT(partition.Chunk(3).Disk(2), ::testing::ElementsAre(7, 8, 9));
  EXPECT_EQ(partition.nElements(), 2 * 3 * 10);

  for (std::size_t c = 0; c < partition.nChunks(); ++c) {
    const auto& chunk = partition.Chunk(c);
    EXPECT_TRUE(chunk.IsContiguous());
    EXPECT_EQ(chunk.FlatOffset(), 2 * 3 * chunk.Mem(2)[0]);
    EXPECT_EQ(chunk.SectionSlicer(),
              Slicer(IPosition({0, 0}), IPosition({1, 2}), Slicer::endIsLast));
  }
}

// Chunks are not split below the minimum number of bytes
TEST(DataPartitionTest, SplitRowsMinimumBytes) {
  auto result_shape =
      ResultShapeData{"DATA", IPosition{2, 3, 10}, 3, DataType::TpComplex, std::nullopt};
  auto selection = SelectionBuilder().Order('F').Build();

  ASSERT_OK_AND_ASSIGN(
      auto partition,
      DataPartition::Make(selection, result_shape, RowSplitOptions{4, 5 * kRowBytes}));
  ASSERT_EQ(partition.nChunks(), 2);
  EXPECT_THAT(partition.Chunk(0).Disk(2), ::testing::ElementsAre(0, 1, 2, 3, 4));
  EXPECT_THAT(partition.Chunk(1).Disk(2), ::testing::ElementsAre(5, 6, 7, 8, 9));

  // A read just over twice the minimum is split into two equal chunks,
  // rather than two of the minimum and a smaller remainder
  ASSERT_OK_AND_ASSIGN(partition, DataPartition::Make(selection, result_shape,
                                                      RowSplitOptions{4, 4 * kRowBytes}));
  ASSERT_EQ(partition.nChunks(), 2);
  EXPECT_THAT(partition.Chunk(0).Disk(2), ::testing::ElementsAre(0, 1, 2, 3, 4));
  EXPECT_THAT(partition.Chunk(1).Disk(2), ::testing::ElementsAre(5, 6, 7, 8, 9));

  // A read less than twice the minimum is not split
  ASSERT_OK_AND_ASSIGN(partition, DataPartition::Make(selection, result_shape,
                                                      RowSplitOptions{4, 6 * kRowBytes}));
  EXPECT_EQ(partition.nChunks(), 1);
  ASSERT_OK_AND_ASSIGN(
      partition,
      DataPartition::Make(selection, result_shape, RowSplitOptions{4, 100 * kRowBytes}));
  EXPECT_EQ(partition.nChunks(), 1);
}

// Selected rows with gaps and in arbitrary order are split in disk order,
// while missing rows remain in a chunk of their own
TEST(DataPartitionTest, SplitSelectedRows) {
  auto result_shape =
      ResultShapeData{"DATA", IPosition{2, 3, 8}, 3, DataType::TpComplex, std::nullopt};
  auto selection = SelectionBuilder()
                       .Order('F')
                       .AddEmpty()
                       .AddEmpty()
                       .Add({8, -1, 5, 6, 7, 4, -1, 0})
                       .Build();

  ASSERT_OK_AND_ASSIGN(auto partition,
                       DataPartition::Make(selection, result_shape, RowSplitOptions{3}));
  ASSERT_EQ(partition.nChunks(), 4);
  EXPECT_THAT(partition.Chunk(0).Disk(2), ::testing::ElementsAre(-1, -1));
  EXPECT_THAT(partition.Chunk(1).Disk(2), ::testing::ElementsAre(0, 4));
  EXPECT_THAT(partition.Chunk(1).Mem(2), ::testing::ElementsAre(7, 5));
  EXPECT_THAT(partition.Chunk(2).Disk(2), ::testing::ElementsAre(5, 6));
  EXPECT_THAT(partition.Chunk(2).Mem(2), ::testing::ElementsAre(2, 3));
  EXPECT_THAT(partition.Chunk(3).Disk(2), ::testing::ElementsAre(7, 8));
  EXPECT_THAT(partition.Chunk(3).Mem(2), ::testing::ElementsAre(4, 0));
  EXPECT_EQ(partition.nElements(), 2 * 3 * 8);
}

}  // namespace
