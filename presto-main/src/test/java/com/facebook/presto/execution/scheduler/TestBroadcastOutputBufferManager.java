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
package com.facebook.presto.execution.scheduler;

import com.facebook.presto.OutputBuffers;
import com.facebook.presto.execution.StageId;
import com.facebook.presto.execution.TaskId;
import org.testng.annotations.Test;

import java.util.concurrent.atomic.AtomicReference;

import static com.facebook.presto.OutputBuffers.INITIAL_EMPTY_OUTPUT_BUFFERS;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertNull;

public class TestBroadcastOutputBufferManager
{
    private static final StageId STAGE_ID = new StageId("query", "stage");

    @Test
    public void test()
            throws Exception
    {
        AtomicReference<OutputBuffers> outputBufferTarget = new AtomicReference<>();
        BroadcastOutputBufferManager hashOutputBufferManager = new BroadcastOutputBufferManager(outputBufferTarget::set);
        assertNull(outputBufferTarget.get());

        hashOutputBufferManager.addOutputBuffer(new TaskId(STAGE_ID, "0"), 0);
        OutputBuffers expectedOutputBuffers = INITIAL_EMPTY_OUTPUT_BUFFERS.withBuffer(new TaskId(STAGE_ID, "0"), 0);
        assertEquals(outputBufferTarget.get(), expectedOutputBuffers);

        hashOutputBufferManager.addOutputBuffer(new TaskId(STAGE_ID, "1"), 0);
        expectedOutputBuffers = expectedOutputBuffers.withBuffer(new TaskId(STAGE_ID, "1"), 0);
        assertEquals(outputBufferTarget.get(), expectedOutputBuffers);

        hashOutputBufferManager.addOutputBuffer(new TaskId(STAGE_ID, "2"), 0);
        expectedOutputBuffers = expectedOutputBuffers.withBuffer(new TaskId(STAGE_ID, "2"), 0);
        assertEquals(outputBufferTarget.get(), expectedOutputBuffers);

        // set no more buffers, which causes buffers to be created
        hashOutputBufferManager.noMoreOutputBuffers();
        expectedOutputBuffers = expectedOutputBuffers.withNoMoreBufferIds();
        assertEquals(outputBufferTarget.get(), expectedOutputBuffers);

        // try to add another buffer, which should not result in an error
        // and output buffers should not change
        hashOutputBufferManager.addOutputBuffer(new TaskId(STAGE_ID, "4"), 0);
        assertEquals(outputBufferTarget.get(), expectedOutputBuffers);

        // try to set no more buffers again, which should not result in an error
        // and output buffers should not change
        hashOutputBufferManager.noMoreOutputBuffers();
        assertEquals(outputBufferTarget.get(), expectedOutputBuffers);
    }
}
