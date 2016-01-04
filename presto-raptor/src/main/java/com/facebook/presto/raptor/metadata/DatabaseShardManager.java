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

import com.facebook.presto.raptor.NodeSupplier;
import com.facebook.presto.raptor.RaptorColumnHandle;
import com.facebook.presto.spi.PrestoException;
import com.facebook.presto.spi.predicate.TupleDomain;
import com.facebook.presto.spi.type.Type;
import com.google.common.base.Joiner;
import com.google.common.base.Throwables;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.common.util.concurrent.ExecutionError;
import com.google.common.util.concurrent.UncheckedExecutionException;
import io.airlift.log.Logger;
import org.h2.jdbc.JdbcConnection;
import org.skife.jdbi.v2.Handle;
import org.skife.jdbi.v2.IDBI;
import org.skife.jdbi.v2.ResultIterator;
import org.skife.jdbi.v2.exceptions.DBIException;
import org.skife.jdbi.v2.util.ByteArrayMapper;

import javax.inject.Inject;

import java.sql.Connection;
import java.sql.JDBCType;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.StringJoiner;
import java.util.UUID;

import static com.facebook.presto.raptor.RaptorErrorCode.RAPTOR_ERROR;
import static com.facebook.presto.raptor.RaptorErrorCode.RAPTOR_EXTERNAL_BATCH_ALREADY_EXISTS;
import static com.facebook.presto.raptor.metadata.SchemaDaoUtil.createTablesWithRetry;
import static com.facebook.presto.raptor.metadata.ShardPredicate.jdbcType;
import static com.facebook.presto.raptor.storage.ShardStats.MAX_BINARY_INDEX_SIZE;
import static com.facebook.presto.raptor.util.ArrayUtil.intArrayFromBytes;
import static com.facebook.presto.raptor.util.ArrayUtil.intArrayToBytes;
import static com.facebook.presto.raptor.util.DatabaseUtil.metadataError;
import static com.facebook.presto.raptor.util.DatabaseUtil.onDemandDao;
import static com.facebook.presto.raptor.util.DatabaseUtil.runIgnoringConstraintViolation;
import static com.facebook.presto.raptor.util.DatabaseUtil.runTransaction;
import static com.facebook.presto.raptor.util.UuidUtil.uuidFromBytes;
import static com.facebook.presto.raptor.util.UuidUtil.uuidToBytes;
import static com.facebook.presto.spi.StandardErrorCode.INTERNAL_ERROR;
import static com.facebook.presto.spi.StandardErrorCode.TRANSACTION_CONFLICT;
import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.collect.Iterables.partition;
import static java.lang.String.format;
import static java.sql.Statement.RETURN_GENERATED_KEYS;
import static java.util.Arrays.asList;
import static java.util.Collections.nCopies;
import static java.util.Objects.requireNonNull;
import static java.util.stream.Collectors.toSet;

public class DatabaseShardManager
        implements ShardManager, ShardRecorder
{
    private static final Logger log = Logger.get(DatabaseShardManager.class);

    private static final String INDEX_TABLE_PREFIX = "x_shards_t";

    private final IDBI dbi;
    private final ShardManagerDao dao;
    private final NodeSupplier nodeSupplier;

    private final LoadingCache<String, Integer> nodeIdCache = CacheBuilder.newBuilder()
            .maximumSize(10_000)
            .build(new CacheLoader<String, Integer>()
            {
                @Override
                public Integer load(String nodeIdentifier)
                {
                    return loadNodeId(nodeIdentifier);
                }
            });

    @Inject
    public DatabaseShardManager(@ForMetadata IDBI dbi, NodeSupplier nodeSupplier)
    {
        this.dbi = requireNonNull(dbi, "dbi is null");
        this.dao = onDemandDao(dbi, ShardManagerDao.class);
        this.nodeSupplier = requireNonNull(nodeSupplier, "nodeSupplier is null");

        createTablesWithRetry(dbi);
    }

    @Override
    public void createTable(long tableId, List<ColumnInfo> columns)
    {
        StringJoiner tableColumns = new StringJoiner(",\n  ", "  ", ",\n").setEmptyValue("");

        for (ColumnInfo column : columns) {
            String columnType = sqlColumnType(column.getType());
            if (columnType != null) {
                tableColumns.add(minColumn(column.getColumnId()) + " " + columnType);
                tableColumns.add(maxColumn(column.getColumnId()) + " " + columnType);
            }
        }

        String sql = "" +
                "CREATE TABLE " + shardIndexTable(tableId) + " (\n" +
                "  shard_id BIGINT NOT NULL PRIMARY KEY,\n" +
                "  shard_uuid BINARY(16) NOT NULL,\n" +
                "  node_ids VARBINARY(128) NOT NULL,\n" +
                tableColumns +
                "  UNIQUE (shard_uuid)\n" +
                ")";

        try (Handle handle = dbi.open()) {
            handle.execute(sql);
        }
        catch (DBIException e) {
            throw metadataError(e);
        }
    }

    @Override
    public void dropTable(long tableId)
    {
        runTransaction(dbi, (handle, status) -> {
            lockTable(handle, tableId);

            ShardManagerDao shardManagerDao = handle.attach(ShardManagerDao.class);
            shardManagerDao.insertDeletedShardNodes(tableId);
            shardManagerDao.insertDeletedShards(tableId);
            shardManagerDao.dropShardNodes(tableId);
            shardManagerDao.dropShards(tableId);

            MetadataDao dao = handle.attach(MetadataDao.class);
            dao.dropColumns(tableId);
            dao.dropTable(tableId);
            return null;
        });

        // TODO: add a cleanup process for leftover index tables
        // It is not possible to drop the index tables in a transaction.
        try (Handle handle = dbi.open()) {
            handle.execute("DROP TABLE " + shardIndexTable(tableId));
        }
        catch (DBIException e) {
            log.warn(e, "Failed to drop index table %s", shardIndexTable(tableId));
        }
    }

    @Override
    public void addColumn(long tableId, ColumnInfo column)
    {
        String columnType = sqlColumnType(column.getType());
        if (columnType == null) {
            return;
        }

        String sql = format("ALTER TABLE %s ADD COLUMN (%s %s, %s %s)",
                shardIndexTable(tableId),
                minColumn(column.getColumnId()), columnType,
                maxColumn(column.getColumnId()), columnType);

        try (Handle handle = dbi.open()) {
            handle.execute(sql);
        }
        catch (DBIException e) {
            throw metadataError(e);
        }
    }

    @Override
    public void commitShards(long transactionId, long tableId, List<ColumnInfo> columns, Collection<ShardInfo> shards, Optional<String> externalBatchId)
    {
        // attempt to fail up front with a proper exception
        if (externalBatchId.isPresent() && dao.externalBatchExists(externalBatchId.get())) {
            throw new PrestoException(RAPTOR_EXTERNAL_BATCH_ALREADY_EXISTS, "External batch already exists: " + externalBatchId.get());
        }

        Map<String, Integer> nodeIds = toNodeIdMap(shards);

        runTransaction(dbi, (handle, status) -> {
            ShardManagerDao dao = handle.attach(ShardManagerDao.class);
            commitTransaction(dao, transactionId);
            externalBatchId.ifPresent(dao::insertExternalBatch);

            lockTable(handle, tableId);
            insertShardsAndIndex(tableId, columns, shards, nodeIds, handle);
            return null;
        });
    }

    @Override
    public void replaceShardUuids(long transactionId, long tableId, List<ColumnInfo> columns, Set<UUID> oldShardUuids, Collection<ShardInfo> newShards)
    {
        Map<String, Integer> nodeIds = toNodeIdMap(newShards);

        runTransaction(dbi, (handle, status) -> {
            commitTransaction(handle.attach(ShardManagerDao.class), transactionId);
            lockTable(handle, tableId);
            for (List<ShardInfo> shards : partition(newShards, 1000)) {
                insertShardsAndIndex(tableId, columns, shards, nodeIds, handle);
            }
            for (List<UUID> uuids : partition(oldShardUuids, 1000)) {
                deleteShardsAndIndex(tableId, ImmutableSet.copyOf(uuids), handle);
            }
            return null;
        });
    }

    private static void deleteShardsAndIndex(long tableId, Set<UUID> shardUuids, Handle handle)
            throws SQLException
    {
        String args = Joiner.on(",").join(nCopies(shardUuids.size(), "?"));

        ImmutableSet.Builder<Long> shardIdSet = ImmutableSet.builder();
        ImmutableList.Builder<UUID> shardUuidList = ImmutableList.builder();
        ImmutableList.Builder<Integer> nodeIdList = ImmutableList.builder();

        String selectShardNodes = format(
                "SELECT shard_id, shard_uuid, node_ids FROM %s WHERE shard_uuid IN (%s) FOR UPDATE",
                shardIndexTable(tableId), args);

        try (PreparedStatement statement = handle.getConnection().prepareStatement(selectShardNodes)) {
            bindUuids(statement, shardUuids);
            try (ResultSet rs = statement.executeQuery()) {
                while (rs.next()) {
                    shardIdSet.add(rs.getLong("shard_id"));
                    UUID shardUuid = uuidFromBytes(rs.getBytes("shard_uuid"));
                    for (Integer nodeId : intArrayFromBytes(rs.getBytes("node_ids"))) {
                        shardUuidList.add(shardUuid);
                        nodeIdList.add(nodeId);
                    }
                }
            }
        }

        Set<Long> shardIds = shardIdSet.build();
        if (shardIds.size() != shardUuids.size()) {
            throw transactionConflict();
        }

        ShardManagerDao dao = handle.attach(ShardManagerDao.class);
        dao.insertDeletedShards(shardUuids);
        dao.insertDeletedShardNodes(shardUuidList.build(), nodeIdList.build());

        String where = " WHERE shard_id IN (" + args + ")";
        String deleteFromShardNodes = "DELETE FROM shard_nodes " + where;
        String deleteFromShards = "DELETE FROM shards " + where;
        String deleteFromShardIndex = "DELETE FROM " + shardIndexTable(tableId) + where;

        try (PreparedStatement statement = handle.getConnection().prepareStatement(deleteFromShardNodes)) {
            bindLongs(statement, shardIds);
            statement.executeUpdate();
        }

        for (String sql : asList(deleteFromShards, deleteFromShardIndex)) {
            try (PreparedStatement statement = handle.getConnection().prepareStatement(sql)) {
                bindLongs(statement, shardIds);
                if (statement.executeUpdate() != shardIds.size()) {
                    throw transactionConflict();
                }
            }
        }
    }

    private static void bindUuids(PreparedStatement statement, Iterable<UUID> uuids)
            throws SQLException
    {
        int i = 1;
        for (UUID uuid : uuids) {
            statement.setBytes(i, uuidToBytes(uuid));
            i++;
        }
    }

    private static void bindLongs(PreparedStatement statement, Iterable<Long> values)
            throws SQLException
    {
        int i = 1;
        for (long value : values) {
            statement.setLong(i, value);
            i++;
        }
    }

    private static void insertShardsAndIndex(long tableId, List<ColumnInfo> columns, Collection<ShardInfo> shards, Map<String, Integer> nodeIds, Handle handle)
            throws SQLException
    {
        Connection connection = handle.getConnection();
        try (IndexInserter indexInserter = new IndexInserter(connection, tableId, columns)) {
            for (List<ShardInfo> batch : partition(shards, batchSize(connection))) {
                List<Long> shardIds = insertShards(connection, tableId, batch);
                insertShardNodes(connection, nodeIds, shardIds, batch);

                for (int i = 0; i < batch.size(); i++) {
                    ShardInfo shard = batch.get(i);
                    Set<Integer> shardNodes = shard.getNodeIdentifiers().stream()
                            .map(nodeIds::get)
                            .collect(toSet());
                    indexInserter.insert(shardIds.get(i), shard.getShardUuid(), shardNodes, shard.getColumnStats());
                }
                indexInserter.execute();
            }
        }
    }

    private static int batchSize(Connection connection)
    {
        // H2 does not return generated keys properly
        // https://github.com/h2database/h2database/issues/156
        return (connection instanceof JdbcConnection) ? 1 : 1000;
    }

    private Map<String, Integer> toNodeIdMap(Collection<ShardInfo> shards)
    {
        Set<String> identifiers = shards.stream()
                .map(ShardInfo::getNodeIdentifiers)
                .flatMap(Collection::stream)
                .collect(toSet());
        return Maps.toMap(identifiers, this::getOrCreateNodeId);
    }

    @Override
    public Set<ShardMetadata> getNodeShards(String nodeIdentifier)
    {
        return dao.getNodeShards(nodeIdentifier);
    }

    @Override
    public ResultIterator<ShardNodes> getShardNodes(long tableId, TupleDomain<RaptorColumnHandle> effectivePredicate)
    {
        return new ShardIterator(tableId, effectivePredicate, dbi);
    }

    @Override
    public void assignShard(long tableId, UUID shardUuid, String nodeIdentifier)
    {
        int nodeId = getOrCreateNodeId(nodeIdentifier);

        runTransaction(dbi, (handle, status) -> {
            ShardManagerDao dao = handle.attach(ShardManagerDao.class);

            Set<Integer> nodes = new HashSet<>(fetchLockedNodeIds(handle, tableId, shardUuid));
            if (nodes.add(nodeId)) {
                updateNodeIds(handle, tableId, shardUuid, nodes);
                dao.insertShardNode(shardUuid, nodeId);
            }

            return null;
        });
    }

    @Override
    public void unassignShard(long tableId, UUID shardUuid, String nodeIdentifier)
    {
        int nodeId = getOrCreateNodeId(nodeIdentifier);

        runTransaction(dbi, (handle, status) -> {
            ShardManagerDao dao = handle.attach(ShardManagerDao.class);

            Set<Integer> nodes = new HashSet<>(fetchLockedNodeIds(handle, tableId, shardUuid));
            if (nodes.remove(nodeId)) {
                updateNodeIds(handle, tableId, shardUuid, nodes);
                dao.deleteShardNode(shardUuid, nodeId);
                dao.insertDeletedShardNodes(ImmutableList.of(shardUuid), ImmutableList.of(nodeId));
            }

            return null;
        });
    }

    @Override
    public Map<String, Long> getNodeBytes()
    {
        String sql = "" +
                "SELECT n.node_identifier, x.size\n" +
                "FROM (\n" +
                "  SELECT node_id, sum(compressed_size) size\n" +
                "  FROM shards s\n" +
                "  JOIN shard_nodes sn ON (s.shard_id = sn.shard_id)\n" +
                "  GROUP BY node_id\n" +
                ") x\n" +
                "JOIN nodes n ON (x.node_id = n.node_id)";

        try (Handle handle = dbi.open()) {
            return handle.createQuery(sql)
                    .fold(ImmutableMap.<String, Long>builder(), (map, rs, ctx) -> {
                        map.put(rs.getString("node_identifier"), rs.getLong("size"));
                        return map;
                    })
                    .build();
        }
    }

    @Override
    public long beginTransaction()
    {
        return dao.insertTransaction();
    }

    @Override
    public void rollbackTransaction(long transactionId)
    {
        dao.finalizeTransaction(transactionId, false);
    }

    private static void commitTransaction(ShardManagerDao dao, long transactionId)
    {
        if (dao.finalizeTransaction(transactionId, true) != 1) {
            throw new PrestoException(TRANSACTION_CONFLICT, "Transaction commit failed. Please retry the operation.");
        }
        dao.deleteCreatedShards(transactionId);
        dao.deleteCreatedShardNodes(transactionId);
    }

    @Override
    public void recordCreatedShard(long transactionId, UUID shardUuid, String nodeIdentifier)
    {
        int nodeId = getOrCreateNodeId(nodeIdentifier);
        runTransaction(dbi, (handle, status) -> {
            ShardManagerDao dao = handle.attach(ShardManagerDao.class);
            dao.insertCreatedShard(shardUuid, transactionId);
            dao.insertCreatedShardNode(shardUuid, nodeId, transactionId);
            return null;
        });
    }

    private int getOrCreateNodeId(String nodeIdentifier)
    {
        try {
            return nodeIdCache.getUnchecked(nodeIdentifier);
        }
        catch (UncheckedExecutionException | ExecutionError e) {
            throw Throwables.propagate(e.getCause());
        }
    }

    private int loadNodeId(String nodeIdentifier)
    {
        Integer id = dao.getNodeId(nodeIdentifier);
        if (id != null) {
            return id;
        }

        // creating a node is idempotent
        runIgnoringConstraintViolation(() -> dao.insertNode(nodeIdentifier));

        id = dao.getNodeId(nodeIdentifier);
        if (id == null) {
            throw new PrestoException(INTERNAL_ERROR, "node does not exist after insert");
        }
        return id;
    }

    private static List<Long> insertShards(Connection connection, long tableId, List<ShardInfo> shards)
            throws SQLException
    {
        String sql = "" +
                "INSERT INTO shards (shard_uuid, table_id, create_time, row_count, compressed_size, uncompressed_size)\n" +
                "VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?, ?)";

        try (PreparedStatement statement = connection.prepareStatement(sql, RETURN_GENERATED_KEYS)) {
            for (ShardInfo shard : shards) {
                statement.setBytes(1, uuidToBytes(shard.getShardUuid()));
                statement.setLong(2, tableId);
                statement.setLong(3, shard.getRowCount());
                statement.setLong(4, shard.getCompressedSize());
                statement.setLong(5, shard.getUncompressedSize());
                statement.addBatch();
            }
            statement.executeBatch();

            ImmutableList.Builder<Long> builder = ImmutableList.builder();
            try (ResultSet keys = statement.getGeneratedKeys()) {
                while (keys.next()) {
                    builder.add(keys.getLong(1));
                }
            }
            List<Long> shardIds = builder.build();

            if (shardIds.size() != shards.size()) {
                throw new PrestoException(RAPTOR_ERROR, "Wrong number of generated keys for inserted shards");
            }
            return shardIds;
        }
    }

    private static void insertShardNodes(Connection connection, Map<String, Integer> nodeIds, List<Long> shardIds, List<ShardInfo> shards)
            throws SQLException
    {
        checkArgument(shardIds.size() == shards.size(), "lists are not the same size");
        String sql = "INSERT INTO shard_nodes (shard_id, node_id) VALUES (?, ?)";
        try (PreparedStatement statement = connection.prepareStatement(sql)) {
            for (int i = 0; i < shards.size(); i++) {
                for (String identifier : shards.get(i).getNodeIdentifiers()) {
                    statement.setLong(1, shardIds.get(i));
                    statement.setInt(2, nodeIds.get(identifier));
                    statement.addBatch();
                }
            }
            statement.executeBatch();
        }
    }

    private static Collection<Integer> fetchLockedNodeIds(Handle handle, long tableId, UUID shardUuid)
    {
        String sql = format(
                "SELECT node_ids FROM %s WHERE shard_uuid = ? FOR UPDATE",
                shardIndexTable(tableId));

        byte[] nodeArray = handle.createQuery(sql)
                .bind(0, uuidToBytes(shardUuid))
                .map(ByteArrayMapper.FIRST)
                .first();

        return intArrayFromBytes(nodeArray);
    }

    private static void updateNodeIds(Handle handle, long tableId, UUID shardUuid, Set<Integer> nodeIds)
    {
        String sql = format(
                "UPDATE %s SET node_ids = ? WHERE shard_uuid = ?",
                shardIndexTable(tableId));

        handle.execute(sql, intArrayToBytes(nodeIds), uuidToBytes(shardUuid));
    }

    private static void lockTable(Handle handle, long tableId)
    {
        if (handle.attach(MetadataDao.class).getLockedTableId(tableId) == null) {
            throw transactionConflict();
        }
    }

    private static PrestoException transactionConflict()
    {
        return new PrestoException(TRANSACTION_CONFLICT, "Table was updated by a different transaction. Please retry the operation.");
    }

    public static String shardIndexTable(long tableId)
    {
        return INDEX_TABLE_PREFIX + tableId;
    }

    public static String minColumn(long columnId)
    {
        return format("c%s_min", columnId);
    }

    public static String maxColumn(long columnId)
    {
        return format("c%s_max", columnId);
    }

    private static String sqlColumnType(Type type)
    {
        JDBCType jdbcType = jdbcType(type);
        if (jdbcType != null) {
            switch (jdbcType) {
                case BOOLEAN:
                    return "boolean";
                case BIGINT:
                    return "bigint";
                case DOUBLE:
                    return "double";
                case INTEGER:
                    return "int";
                case VARBINARY:
                    return format("varbinary(%s)", MAX_BINARY_INDEX_SIZE);
            }
        }
        return null;
    }
}
