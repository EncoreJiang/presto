/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.facebook.presto.operator;

import com.facebook.presto.operator.window.WindowFunction;
import com.facebook.presto.spi.block.Block;
import com.facebook.presto.spi.block.SortOrder;
import com.facebook.presto.spi.type.Type;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Ordering;
import com.google.common.primitives.Ints;
import com.google.common.util.concurrent.ListenableFuture;
import io.airlift.units.DataSize;

import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

public class TopNRowNumberOperator
        implements Operator
{
    public static class TopNRowNumberOperatorFactory
            implements OperatorFactory
    {
        private final int operatorId;

        private final List<Type> sourceTypes;
        private final List<Integer> outputChannels;
        private final List<WindowFunctionDefinition> windowFunctionDefinitions;
        private final List<Integer> partitionChannels;
        private final List<Type> partitionTypes;
        private final List<Integer> sortChannels;
        private final List<SortOrder> sortOrder;
        private final boolean partial;
        private final int maxRowCountPerPartition;
        private final int expectedPositions;

        private final List<Type> types;
        private boolean closed;

        public TopNRowNumberOperatorFactory(
                int operatorId,
                List<? extends Type> sourceTypes,
                List<Integer> outputChannels,
                List<WindowFunctionDefinition> windowFunctionDefinitions,
                List<Integer> partitionChannels,
                List<? extends Type> partitionTypes,
                List<Integer> sortChannels,
                List<SortOrder> sortOrder,
                boolean partial,
                int maxRowCountPerPartition,
                int expectedPositions)
        {
            this.operatorId = operatorId;
            this.sourceTypes = ImmutableList.copyOf(sourceTypes);
            this.outputChannels = ImmutableList.copyOf(checkNotNull(outputChannels, "outputChannels is null"));
            this.windowFunctionDefinitions = windowFunctionDefinitions;
            this.partitionChannels = ImmutableList.copyOf(checkNotNull(partitionChannels, "partitionChannels is null"));
            this.partitionTypes = ImmutableList.copyOf(checkNotNull(partitionTypes, "partitionTypes is null"));
            this.sortChannels = ImmutableList.copyOf(checkNotNull(sortChannels));
            this.sortOrder = ImmutableList.copyOf(checkNotNull(sortOrder));
            this.partial = partial;
            checkArgument(maxRowCountPerPartition > 0, "maxRowCountPerPartition must be > 0");
            this.maxRowCountPerPartition = maxRowCountPerPartition;
            this.expectedPositions = expectedPositions;

            this.types = toTypes(sourceTypes, outputChannels, toWindowFunctions(windowFunctionDefinitions));
        }

        @Override
        public List<Type> getTypes()
        {
            return types;
        }

        @Override
        public Operator createOperator(DriverContext driverContext)
        {
            checkState(!closed, "Factory is already closed");
            OperatorContext operatorContext = driverContext.addOperatorContext(operatorId, TopNRowNumberOperator.class.getSimpleName());
            return new TopNRowNumberOperator(
                    operatorContext,
                    sourceTypes,
                    outputChannels,
                    windowFunctionDefinitions,
                    partitionChannels,
                    partitionTypes,
                    sortChannels,
                    sortOrder,
                    partial,
                    maxRowCountPerPartition,
                    expectedPositions);
        }

        @Override
        public void close()
        {
            closed = true;
        }
    }

    private static final DataSize OVERHEAD_PER_VALUE = new DataSize(100, DataSize.Unit.BYTE); // for estimating in-memory size. This is a completely arbitrary number

    private final OperatorContext operatorContext;
    private final boolean partial;
    private TopNPartitionedBuffer topNPartitionedBuffer;
    private boolean finishing;

    private final List<Type> types;

    public TopNRowNumberOperator(
            OperatorContext operatorContext,
            List<? extends Type> sourceTypes,
            List<Integer> outputChannels,
            List<WindowFunctionDefinition> windowFunctionDefinitions,
            List<Integer> partitionChannels,
            List<Type> partitionTypes,
            List<Integer> sortChannels,
            List<SortOrder> sortOrder,
            boolean partial,
            int maxRowCountPerPartition,
            int expectedPositions)
    {
        this.operatorContext = checkNotNull(operatorContext, "operatorContext is null");
        this.partial = partial;
        checkArgument(maxRowCountPerPartition > 0, "maxRowCountPerPartition must be > 0");

        List<WindowFunction> windowFunctions = toWindowFunctions(checkNotNull(windowFunctionDefinitions, "windowFunctionDefinitions is null"));
        this.types = toTypes(sourceTypes, outputChannels, windowFunctions);
        topNPartitionedBuffer = new TopNPartitionedBuffer(
                operatorContext,
                partitionChannels,
                partitionTypes,
                sortChannels,
                sortOrder,
                outputChannels,
                types,
                maxRowCountPerPartition,
                partial,
                expectedPositions);
    }

    @Override
    public OperatorContext getOperatorContext()
    {
        return operatorContext;
    }

    @Override
    public List<Type> getTypes()
    {
        return types;
    }

    @Override
    public void finish()
    {
        finishing = true;
    }

    @Override
    public boolean isFinished()
    {
        return finishing && topNPartitionedBuffer.isEmpty();
    }

    @Override
    public ListenableFuture<?> isBlocked()
    {
        return NOT_BLOCKED;
    }

    @Override
    public boolean needsInput()
    {
        return !finishing && !topNPartitionedBuffer.isFull();
    }

    @Override
    public void addInput(Page page)
    {
        checkState(!finishing, "Operator is already finishing");
        checkNotNull(page, "page is null");
        checkState(!topNPartitionedBuffer.isFull(), "Aggregation buffer is full");
        topNPartitionedBuffer.processPage(page);
    }

    @Override
    public Page getOutput()
    {
        if (!finishing && partial && topNPartitionedBuffer.isFull()) {
            return topNPartitionedBuffer.getPage();
        }
        if (finishing && !topNPartitionedBuffer.isEmpty()) {
            return topNPartitionedBuffer.getPage();
        }
        return null;
    }

    private static List<Type> toTypes(List<? extends Type> sourceTypes, List<Integer> outputChannels, List<WindowFunction> windowFunctions)
    {
        ImmutableList.Builder<Type> types = ImmutableList.builder();
        for (int channel : outputChannels) {
            types.add(sourceTypes.get(channel));
        }
        for (WindowFunction function : windowFunctions) {
            types.add(function.getType());
        }
        return types.build();
    }

    private static List<WindowFunction> toWindowFunctions(List<WindowFunctionDefinition> windowFunctionDefinitions)
    {
        ImmutableList.Builder<WindowFunction> builder = ImmutableList.builder();
        for (WindowFunctionDefinition windowFunctionDefinition : windowFunctionDefinitions) {
            builder.add(windowFunctionDefinition.createWindowFunction());
        }
        return builder.build();
    }

    private static class TopNPartitionedBuffer
    {
        private final int maxRowCountPerPartition;
        private final List<Integer> sortChannels;
        private final List<SortOrder> sortOrder;
        private final int[] outputChannels;
        private final List<Type> outputTypes;
        private final boolean partial;
        private long memorySize;
        private final MemoryManager memoryManager;

        private final GroupByHash groupByHash;

        private final Map<Long, PartitionBuilder> partitionRows;

        public TopNPartitionedBuffer(OperatorContext operatorContext,
                List<Integer> partitionChannels,
                List<Type> partitionTypes,
                List<Integer> sortChannels,
                List<SortOrder> sortOrder,
                List<Integer> outputChannels,
                List<Type> outputTypes,
                int maxRowCountPerPartition,
                boolean partial,
                int expectedPositions)
        {
            this.sortChannels = sortChannels;
            this.sortOrder = sortOrder;
            this.outputChannels = Ints.toArray(outputChannels);
            this.outputTypes = outputTypes;
            this.memoryManager = new MemoryManager(operatorContext);
            this.partitionRows = new HashMap<>();
            this.maxRowCountPerPartition = maxRowCountPerPartition;
            this.partial = partial;
            this.groupByHash = new GroupByHash(partitionTypes, Ints.toArray(partitionChannels), expectedPositions);
        }

        private void processPage(Page page)
        {
            GroupByIdBlock partitionIds = groupByHash.getGroupIds(page);
            long sizeDelta = 0;
            for (int position = 0; position < page.getPositionCount(); position++) {
                long partitionId = partitionIds.getGroupId(position);
                if (!partitionRows.containsKey(partitionId)) {
                    partitionRows.put(partitionId, new PartitionBuilder(sortChannels, sortOrder, maxRowCountPerPartition));
                }
                PartitionBuilder partitionBuilder = partitionRows.get(partitionId);
                Block[] row = page.getValues(position);
                sizeDelta += partitionBuilder.addRow(row);
            }
            memorySize += sizeDelta;
        }

        private Page getPage()
        {
            PageBuilder pageBuilder = new PageBuilder(outputTypes);
            pageBuilder.reset();
            Optional<Long> partitionToFlush = getNextPartitionToFlush();
            while (!pageBuilder.isFull() && partitionToFlush.isPresent()) {
                long partitionId = partitionToFlush.get();
                if (pageBuilder.canAppend(partitionRows.get(partitionId).getPositionCount(), partitionRows.get(partitionId).getPartitionSize())) {
                    flushPartition(partitionId, pageBuilder);
                }
                partitionToFlush = getNextPartitionToFlush();
            }
            return pageBuilder.build();
        }

        private boolean isFull()
        {
            return !memoryManager.canUse(memorySize);
        }

        private boolean isEmpty()
        {
            return partitionRows.isEmpty();
        }

        private Optional<Long> getNextPartitionToFlush()
        {
            for (long partitionId : partitionRows.keySet()) {
                if (!partitionRows.get(partitionId).isEmpty()) {
                    return Optional.of(partitionId);
                }
            }
            return Optional.absent();
        }

        private void flushPartition(long partitionId, PageBuilder pageBuilder)
        {
            PartitionBuilder partitionBuilder = partitionRows.get(partitionId);
            long partitionSize = partitionBuilder.getPartitionSize();
            Iterator<Block[]> outputIterator = partitionBuilder.build();
            partitionRows.remove(partitionId);
            memorySize -= partitionSize;

            long rowNumber = 1;
            while (outputIterator.hasNext()) {
                Block[] row = outputIterator.next();
                int channel = 0;
                while (channel < outputChannels.length) {
                    row[outputChannels[channel]].appendTo(0, pageBuilder.getBlockBuilder(channel));
                    channel++;
                }
                if (!partial) {
                    pageBuilder.getBlockBuilder(channel).appendLong(rowNumber);
                    rowNumber++;
                }
            }
        }
    }

    private static class PartitionBuilder
    {
        private final long maxRowCountPerPartition;
        private final PriorityQueue<Block[]> candidateRows;
        private long partitionSize;

        private PartitionBuilder(List<Integer> sortChannels, List<SortOrder> sortOrders, long maxRowCountPerPartition)
        {
            this.maxRowCountPerPartition = maxRowCountPerPartition;
            Ordering<Block[]> comparator = Ordering.from(new RowComparator(sortChannels, sortOrders)).reverse();
            this.candidateRows = new PriorityQueue<>((int) maxRowCountPerPartition, comparator);
        }

        private long addRow(Block[] row)
        {
            checkState(candidateRows.size() <= maxRowCountPerPartition);
            long sizeDelta = sizeOfRow(row);
            candidateRows.add(row);
            if (candidateRows.size() > maxRowCountPerPartition) {
                Block[] oldRow = candidateRows.remove();
                sizeDelta -= sizeOfRow(oldRow);
            }
            partitionSize += sizeDelta;
            return sizeDelta;
        }

        private Iterator<Block[]> build()
        {
            ImmutableList.Builder<Block[]> sortedCandidates = ImmutableList.builder();
            long sizeDelta = 0;
            while (!candidateRows.isEmpty()) {
                Block[] row = candidateRows.remove();
                sizeDelta += sizeOfRow(row);
                sortedCandidates.add(row);
            }
            partitionSize -= sizeDelta;
            return sortedCandidates.build().reverse().iterator();
        }

        private boolean isEmpty()
        {
            return candidateRows.isEmpty();
        }

        private static long sizeOfRow(Block[] row)
        {
            long size = OVERHEAD_PER_VALUE.toBytes();
            for (Block value : row) {
                size += value.getSizeInBytes();
            }
            return size;
        }

        private long getPartitionSize()
        {
            return partitionSize;
        }

        public int getPositionCount()
        {
            return candidateRows.size();
        }
    }

    private static class RowComparator
            implements Comparator<Block[]>
    {
        private final List<Integer> sortChannels;
        private final List<SortOrder> sortOrders;

        public RowComparator(List<Integer> sortChannels, List<SortOrder> sortOrders)
        {
            checkNotNull(sortChannels, "sortChannels is null");
            checkNotNull(sortOrders, "sortOrders is null");
            checkArgument(sortChannels.size() == sortOrders.size(), "sortFields size (%s) doesn't match sortOrders size (%s)", sortChannels.size(), sortOrders.size());

            this.sortChannels = ImmutableList.copyOf(sortChannels);
            this.sortOrders = ImmutableList.copyOf(sortOrders);
        }

        @Override
        public int compare(Block[] leftRow, Block[] rightRow)
        {
            for (int index = 0; index < sortChannels.size(); index++) {
                int channel = sortChannels.get(index);
                SortOrder sortOrder = sortOrders.get(index);

                Block left = leftRow[channel];
                Block right = rightRow[channel];

                int comparison = left.compareTo(sortOrder, 0, right, 0);
                if (comparison != 0) {
                    return comparison;
                }
            }
            return 0;
        }
    }
}
