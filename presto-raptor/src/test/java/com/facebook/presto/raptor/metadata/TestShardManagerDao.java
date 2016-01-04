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
package com.facebook.presto.raptor.metadata;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import org.skife.jdbi.v2.DBI;
import org.skife.jdbi.v2.Handle;
import org.skife.jdbi.v2.IDBI;
import org.skife.jdbi.v2.exceptions.UnableToExecuteStatementException;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

import java.sql.SQLException;
import java.sql.Timestamp;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import static com.facebook.presto.raptor.metadata.SchemaDaoUtil.createTablesWithRetry;
import static io.airlift.testing.Assertions.assertInstanceOf;
import static java.util.concurrent.TimeUnit.DAYS;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.fail;

@Test(singleThreaded = true)
public class TestShardManagerDao
{
    private ShardManagerDao dao;
    private IDBI dbi;
    private Handle dummyHandle;

    @BeforeMethod
    public void setup()
            throws Exception
    {
        dbi = new DBI("jdbc:h2:mem:test" + System.nanoTime());
        dummyHandle = dbi.open();
        dao = dbi.onDemand(ShardManagerDao.class);
        createTablesWithRetry(dbi);
    }

    @AfterMethod(alwaysRun = true)
    public void teardown()
    {
        dummyHandle.close();
    }

    @Test
    public void testExternalBatches()
    {
        assertFalse(dao.externalBatchExists("foo"));
        assertFalse(dao.externalBatchExists("bar"));

        dao.insertExternalBatch("foo");

        assertTrue(dao.externalBatchExists("foo"));
        assertFalse(dao.externalBatchExists("bar"));

        try {
            dao.insertExternalBatch("foo");
            fail("expected exception");
        }
        catch (UnableToExecuteStatementException e) {
            assertInstanceOf(e.getCause(), SQLException.class);
            assertTrue(((SQLException) e.getCause()).getSQLState().startsWith("23"));
        }
    }

    @Test
    public void testInsertCreatedShard()
            throws Exception
    {
        long transactionId = dao.insertTransaction();
        dao.insertCreatedShard(UUID.randomUUID(), transactionId);
        dao.deleteCreatedShards(transactionId);
    }

    @Test
    public void testInsertCreatedShardNode()
            throws Exception
    {
        int nodeId = dao.insertNode("node");
        long transactionId = dao.insertTransaction();
        dao.insertCreatedShardNode(UUID.randomUUID(), nodeId, transactionId);
        dao.deleteCreatedShardNodes(transactionId);
    }

    @Test
    public void testInsertDeletedShards()
            throws Exception
    {
        dao.insertDeletedShards(ImmutableList.of(UUID.randomUUID(), UUID.randomUUID()));
        dao.insertDeletedShards(0);
    }

    @Test
    public void testInsertDeletedShardNode()
            throws Exception
    {
        List<UUID> shardUuids = ImmutableList.of(UUID.randomUUID(), UUID.randomUUID());
        List<Integer> nodeIds = ImmutableList.of(dao.insertNode("node1"), dao.insertNode("node2"));
        dao.insertDeletedShardNodes(shardUuids, nodeIds);
        dao.insertDeletedShardNodes(0);
    }

    @Test
    public void testNodeInsert()
            throws Exception
    {
        assertEquals(dao.getAllNodesInUse(), ImmutableSet.of());

        String nodeName = UUID.randomUUID().toString();
        int nodeId = dao.insertNode(nodeName);
        assertEquals(dao.getNodeId(nodeName), (Integer) nodeId);

        assertEquals(dao.getAllNodesInUse(), ImmutableSet.of(nodeName));
    }

    @Test
    public void testInsertShard()
    {
        long tableId = createTable("test");
        long shardId = dao.insertShard(UUID.randomUUID(), tableId, 13, 42, 84);

        String sql = "SELECT table_id, row_count, compressed_size, uncompressed_size " +
                "FROM shards WHERE shard_id = ?";
        List<Map<String, Object>> shards = dbi.withHandle(handle -> handle.select(sql, shardId));

        assertEquals(shards.size(), 1);
        Map<String, Object> shard = shards.get(0);

        assertEquals(shard.get("table_id"), tableId);
        assertEquals(shard.get("row_count"), 13L);
        assertEquals(shard.get("compressed_size"), 42L);
        assertEquals(shard.get("uncompressed_size"), 84L);
    }

    @Test
    public void testInsertShardNodeUsingShardUuid()
            throws Exception
    {
        int nodeId = dao.insertNode("node");

        long tableId = createTable("test");
        UUID shard = UUID.randomUUID();
        dao.insertShard(shard, tableId, 0, 0, 0);

        dao.insertShardNode(shard, nodeId);

        assertEquals(dao.getShardNodes(tableId), ImmutableList.of(new ShardNode(shard, "node")));
    }

    @Test
    public void testNodeShards()
            throws Exception
    {
        assertEquals(dao.getAllNodesInUse(), ImmutableSet.of());

        String nodeName1 = UUID.randomUUID().toString();
        int nodeId1 = dao.insertNode(nodeName1);

        String nodeName2 = UUID.randomUUID().toString();
        int nodeId2 = dao.insertNode(nodeName2);

        assertEquals(dao.getAllNodesInUse(), ImmutableSet.of(nodeName1, nodeName2));

        UUID shardUuid1 = UUID.randomUUID();
        UUID shardUuid2 = UUID.randomUUID();
        UUID shardUuid3 = UUID.randomUUID();
        UUID shardUuid4 = UUID.randomUUID();

        long tableId = createTable("test");

        long shardId1 = dao.insertShard(shardUuid1, tableId, 1, 11, 111);
        long shardId2 = dao.insertShard(shardUuid2, tableId, 2, 22, 222);
        long shardId3 = dao.insertShard(shardUuid3, tableId, 3, 33, 333);
        long shardId4 = dao.insertShard(shardUuid4, tableId, 4, 44, 444);

        assertEquals(dao.getShards(tableId), ImmutableList.of(shardUuid1, shardUuid2, shardUuid3, shardUuid4));

        assertEquals(dao.getNodeShards(nodeName1).size(), 0);
        assertEquals(dao.getNodeShards(nodeName2).size(), 0);

        dao.insertShardNode(shardId1, nodeId1);
        dao.insertShardNode(shardId2, nodeId1);
        dao.insertShardNode(shardId3, nodeId1);
        dao.insertShardNode(shardId4, nodeId1);
        dao.insertShardNode(shardId1, nodeId2);
        dao.insertShardNode(shardId4, nodeId2);

        ShardMetadata shard1 = new ShardMetadata(tableId, shardId1, shardUuid1, 1, 11, 111);
        ShardMetadata shard2 = new ShardMetadata(tableId, shardId2, shardUuid2, 2, 22, 222);
        ShardMetadata shard3 = new ShardMetadata(tableId, shardId3, shardUuid3, 3, 33, 333);
        ShardMetadata shard4 = new ShardMetadata(tableId, shardId4, shardUuid4, 4, 44, 444);

        assertEquals(dao.getNodeShards(nodeName1), ImmutableSet.of(shard1, shard2, shard3, shard4));
        assertEquals(dao.getNodeShards(nodeName2), ImmutableSet.of(shard1, shard4));

        dao.dropShardNodes(tableId);

        assertEquals(dao.getShardNodes(tableId), ImmutableList.of());

        dao.dropShards(tableId);

        assertEquals(dao.getShards(tableId), ImmutableList.of());
    }

    @Test
    public void testShardSelection()
            throws Exception
    {
        assertEquals(dao.getAllNodesInUse(), ImmutableSet.of());

        String nodeName1 = UUID.randomUUID().toString();
        int nodeId1 = dao.insertNode(nodeName1);

        assertEquals(dao.getAllNodesInUse(), ImmutableSet.of(nodeName1));

        String nodeName2 = UUID.randomUUID().toString();
        int nodeId2 = dao.insertNode(nodeName2);

        assertEquals(dao.getAllNodesInUse(), ImmutableSet.of(nodeName1, nodeName2));

        long tableId = createTable("test");

        UUID shardUuid1 = UUID.randomUUID();
        UUID shardUuid2 = UUID.randomUUID();
        UUID shardUuid3 = UUID.randomUUID();
        UUID shardUuid4 = UUID.randomUUID();

        long shardId1 = dao.insertShard(shardUuid1, tableId, 0, 0, 0);
        long shardId2 = dao.insertShard(shardUuid2, tableId, 0, 0, 0);
        long shardId3 = dao.insertShard(shardUuid3, tableId, 0, 0, 0);
        long shardId4 = dao.insertShard(shardUuid4, tableId, 0, 0, 0);

        List<UUID> shards = dao.getShards(tableId);
        assertEquals(shards.size(), 4);

        assertTrue(shards.contains(shardUuid1));
        assertTrue(shards.contains(shardUuid2));
        assertTrue(shards.contains(shardUuid3));
        assertTrue(shards.contains(shardUuid4));

        assertEquals(dao.getShardNodes(tableId).size(), 0);

        dao.insertShardNode(shardId1, nodeId1);
        dao.insertShardNode(shardId1, nodeId2);
        dao.insertShardNode(shardId2, nodeId1);
        dao.insertShardNode(shardId3, nodeId1);
        dao.insertShardNode(shardId4, nodeId1);
        dao.insertShardNode(shardId4, nodeId2);

        assertEquals(dao.getShards(tableId), shards);

        List<ShardNode> shardNodes = dao.getShardNodes(tableId);
        assertEquals(shardNodes.size(), 6);

        assertContainsShardNode(shardNodes, nodeName1, shardUuid1);
        assertContainsShardNode(shardNodes, nodeName2, shardUuid1);
        assertContainsShardNode(shardNodes, nodeName1, shardUuid2);
        assertContainsShardNode(shardNodes, nodeName1, shardUuid3);
        assertContainsShardNode(shardNodes, nodeName1, shardUuid4);
        assertContainsShardNode(shardNodes, nodeName2, shardUuid4);
    }

    @Test
    public void testDeletedShardNodes()
    {
        String nodeName1 = UUID.randomUUID().toString();
        int nodeId1 = dao.insertNode(nodeName1);

        String nodeName2 = UUID.randomUUID().toString();
        int nodeId2 = dao.insertNode(nodeName2);

        ImmutableList<UUID> shards = ImmutableList.of(UUID.randomUUID());

        // insert shard on both nodes
        dao.insertDeletedShardNodes(shards, ImmutableList.of(nodeId1));
        dao.insertDeletedShardNodes(shards, ImmutableList.of(nodeId2));

        // verify we should clean from both
        assertEquals(dao.getCleanableShardNodesBatch(nodeName1, future()), shards);
        assertEquals(dao.getCleanableShardNodesBatch(nodeName2, future()), shards);

        // clean on first node
        dao.updateCleanedShardNodes(shards, nodeId1);
        assertEquals(dao.getCleanableShardNodesBatch(nodeName1, future()), ImmutableList.of());
        assertEquals(dao.getCleanableShardNodesBatch(nodeName2, future()), shards);

        // clean on second node
        dao.updateCleanedShardNodes(shards, nodeId2);
        assertEquals(dao.getCleanableShardNodesBatch(nodeName1, future()), ImmutableList.of());
        assertEquals(dao.getCleanableShardNodesBatch(nodeName2, future()), ImmutableList.of());

        // verify we should purge from both
        assertEquals(dao.getPurgableShardNodesBatch(nodeName1, future()), shards);
        assertEquals(dao.getPurgableShardNodesBatch(nodeName2, future()), shards);

        // purge on first node
        dao.updatePurgedShardNodes(shards, nodeId1);
        assertEquals(dao.getPurgableShardNodesBatch(nodeName1, future()), ImmutableList.of());
        assertEquals(dao.getPurgableShardNodesBatch(nodeName2, future()), shards);

        // purge on second node
        dao.updatePurgedShardNodes(shards, nodeId2);
        assertEquals(dao.getPurgableShardNodesBatch(nodeName1, future()), ImmutableList.of());
        assertEquals(dao.getPurgableShardNodesBatch(nodeName2, future()), ImmutableList.of());
    }

    private long createTable(String name)
    {
        return dbi.onDemand(MetadataDao.class).insertTable("test", name, false);
    }

    private static void assertContainsShardNode(List<ShardNode> nodes, String nodeName, UUID shardUuid)
    {
        assertTrue(nodes.contains(new ShardNode(shardUuid, nodeName)));
    }

    private static Timestamp future()
    {
        return new Timestamp(System.currentTimeMillis() + DAYS.toMillis(1));
    }
}
