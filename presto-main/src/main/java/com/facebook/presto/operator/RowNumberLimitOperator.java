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
import com.facebook.presto.spi.type.Type;
import com.facebook.presto.util.array.BooleanBigArray;
import com.google.common.base.Optional;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Multimap;
import com.google.common.primitives.Ints;
import com.google.common.util.concurrent.ListenableFuture;
import io.airlift.units.DataSize;

import java.util.List;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

public class RowNumberLimitOperator
        implements Operator
{
    public static class RowNumberLimitOperatorFactory
            implements OperatorFactory
    {
        private final int operatorId;

        private final List<Type> sourceTypes;
        private final List<Integer> outputChannels;
        private final List<WindowFunctionDefinition> windowFunctionDefinitions;
        private final List<Integer> partitionChannels;
        private final int expectedPositions;
        private final List<Type> types;
        private boolean closed;

        private final List<Type> partitionTypes;
        private final int maxRowCountPerPartition;

        public RowNumberLimitOperatorFactory(
                int operatorId,
                List<? extends Type> sourceTypes,
                List<Integer> outputChannels,
                List<WindowFunctionDefinition> windowFunctionDefinitions,
                List<Integer> partitionChannels,
                List<? extends Type> partitionTypes,
                int expectedPositions,
                int maxRowCountPerPartition)
        {
            this.operatorId = operatorId;
            this.sourceTypes = ImmutableList.copyOf(sourceTypes);
            this.outputChannels = ImmutableList.copyOf(checkNotNull(outputChannels, "outputChannels is null"));
            this.windowFunctionDefinitions = windowFunctionDefinitions;
            this.partitionChannels = ImmutableList.copyOf(checkNotNull(partitionChannels, "partitionChannels is null"));
            this.partitionTypes = ImmutableList.copyOf(checkNotNull(partitionTypes, "partitionTypes is null"));
            this.expectedPositions = expectedPositions;
            this.maxRowCountPerPartition = maxRowCountPerPartition;

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

            OperatorContext operatorContext = driverContext.addOperatorContext(operatorId, RowNumberLimitOperator.class.getSimpleName());
            return new RowNumberLimitOperator(
                    operatorContext,
                    sourceTypes,
                    outputChannels,
                    windowFunctionDefinitions,
                    partitionChannels,
                    partitionTypes,
                    expectedPositions,
                    maxRowCountPerPartition);
        }

        @Override
        public void close()
        {
            closed = true;
        }
    }

    private static final DataSize OVERHEAD_PER_VALUE = new DataSize(100, DataSize.Unit.BYTE); // for estimating in-memory size. This is a completely arbitrary number

    private final OperatorContext operatorContext;

    private final int[] outputChannels;
    private final int maxRowCountPerPartition;

    private boolean finishing;
    private final PageBuilder pageBuilder;
    private long memorySize;
    private final MemoryManager memoryManager;

    private long maxPartitionId = -1;
    private final GroupByHash groupByHash;

    private final Multimap<Long, Block[]> candidateRows;
    private final BooleanBigArray partitionDone;
    private boolean allPartitionsDone;

    private final List<Type> types;

    public RowNumberLimitOperator(
            OperatorContext operatorContext,
            List<Type> sourceTypes,
            List<Integer> outputChannels,
            List<WindowFunctionDefinition> windowFunctionDefinitions,
            List<Integer> partitionChannels,
            List<Type> partitionTypes,
            int expectedPositions,
            int maxRowCountPerPartition)
    {
        this.operatorContext = checkNotNull(operatorContext, "operatorContext is null");
        this.maxRowCountPerPartition = maxRowCountPerPartition;
        this.memoryManager = new MemoryManager(operatorContext);
        this.candidateRows = HashMultimap.create();
        this.partitionDone = new BooleanBigArray(false);
        this.groupByHash = new GroupByHash(partitionTypes, Ints.toArray(partitionChannels), expectedPositions);
        this.outputChannels = Ints.toArray(checkNotNull(outputChannels, "outputChannels is null"));
        List<WindowFunction> windowFunctions = toWindowFunctions(checkNotNull(windowFunctionDefinitions, "windowFunctionDefinitions is null"));
        this.types = toTypes(sourceTypes, outputChannels, windowFunctions);
        this.pageBuilder = new PageBuilder(this.types);
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
        return finishing && allPartitionsDone;
    }

    @Override
    public ListenableFuture<?> isBlocked()
    {
        return NOT_BLOCKED;
    }

    @Override
    public boolean needsInput()
    {
        return !finishing && !isFull();
    }

    @Override
    public void addInput(Page page)
    {
        checkState(!finishing, "Operator is already finishing");
        checkNotNull(page, "page is null");
        checkState(!isFull(), "Buffer is full");

        GroupByIdBlock partitionIds = groupByHash.getGroupIds(page);
        long sizeDelta = 0;
        for (int position = 0; position < page.getPositionCount(); position++) {
            long partitionId = partitionIds.getGroupId(position);
            maxPartitionId = Math.max(partitionId, maxPartitionId);
            if (!candidateRows.containsKey(partitionId)) {
                partitionDone.ensureCapacity(partitionId);
                partitionDone.set(partitionId, false);
            }
            if (!partitionDone.get(partitionId) && candidateRows.get(partitionId).size() < maxRowCountPerPartition) {
                Block[] row = page.getValues(position);
                sizeDelta += sizeOfRow(row);
                candidateRows.put(partitionId, row);
            }
        }
        memorySize += sizeDelta;
    }

    @Override
    public Page getOutput()
    {
        Optional<Long> partitionToFlush;
        if (!finishing) {
            partitionToFlush = getCompletePartitionToFlush();
        }
        else {
            partitionToFlush = getPartitionToFlush();
        }
        if (partitionToFlush.isPresent()) {
            checkState(!partitionDone.get(partitionToFlush.get()), "partition already flushed");
            return flushPartition(partitionToFlush.get());
        }
        else if (finishing && existingPartitionsDone()) {
            allPartitionsDone = true;
        }
        return null;
    }

    private boolean isFull()
    {
        return !memoryManager.canUse(memorySize);
    }

    private boolean existingPartitionsDone()
    {
        for (long partitionId = 0; partitionId <= maxPartitionId; partitionId++) {
            if (!partitionDone.get(partitionId)) {
                return false;
            }
        }
        return true;
    }

    private Optional<Long> getCompletePartitionToFlush()
    {
        for (long partitionId : candidateRows.keySet()) {
            if (candidateRows.get(partitionId).size() == maxRowCountPerPartition) {
                return Optional.of(partitionId);
            }
        }
        return Optional.absent();
    }

    private Optional<Long> getPartitionToFlush()
    {
        for (long partitionId : candidateRows.keySet()) {
            if (!candidateRows.get(partitionId).isEmpty()) {
                return Optional.of(partitionId);
            }
        }
        return Optional.absent();
    }

    private Page flushPartition(long partitionId)
    {
        pageBuilder.reset();
        long rowNumber = 1;
        long sizeDelta = 0;
        for (Block[] row : candidateRows.get(partitionId)) {
            checkState(!pageBuilder.isFull(), "Task exceeded memory limit");
            sizeDelta += sizeOfRow(row);
            int channel = 0;
            while (channel < outputChannels.length) {
                row[outputChannels[channel]].appendTo(0, pageBuilder.getBlockBuilder(channel));
                channel++;
            }
            pageBuilder.getBlockBuilder(channel).appendLong(rowNumber);
            rowNumber++;
        }
        candidateRows.removeAll(partitionId);
        partitionDone.set(partitionId, true);
        memorySize -= sizeDelta;
        return pageBuilder.build();
    }

    private static long sizeOfRow(Block[] row)
    {
        long size = OVERHEAD_PER_VALUE.toBytes();
        for (Block value : row) {
            size += value.getSizeInBytes();
        }
        return size;
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
}
