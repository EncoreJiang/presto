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

import com.facebook.presto.raptor.metadata.Table.TableMapper;
import com.facebook.presto.spi.SchemaTableName;
import org.skife.jdbi.v2.sqlobject.Bind;
import org.skife.jdbi.v2.sqlobject.GetGeneratedKeys;
import org.skife.jdbi.v2.sqlobject.SqlQuery;
import org.skife.jdbi.v2.sqlobject.SqlUpdate;
import org.skife.jdbi.v2.sqlobject.customizers.Mapper;

import java.util.List;

public interface MetadataDao
{
    @SqlQuery("SELECT table_id FROM tables\n" +
            "WHERE schema_name = :schemaName\n" +
            "  AND table_name = :tableName")
    @Mapper(TableMapper.class)
    Table getTableInformation(
            @Bind("schemaName") String schemaName,
            @Bind("tableName") String tableName);

    @SqlQuery("SELECT t.schema_name, t.table_name,\n" +
            "  c.column_id, c.column_name, c.ordinal_position, c.data_type\n" +
            "FROM tables t\n" +
            "JOIN columns c ON (t.table_id = c.table_id)\n" +
            "WHERE t.table_id = :tableId\n" +
            "  AND c.column_id = :columnId\n" +
            "ORDER BY c.ordinal_position\n")
    TableColumn getTableColumn(
            @Bind("tableId") long tableId,
            @Bind("columnId") long columnId);

    @SqlQuery("SELECT t.schema_name, t.table_name,\n" +
            "  c.column_id, c.column_name, c.ordinal_position, c.data_type\n" +
            "FROM tables t\n" +
            "JOIN columns c ON (t.table_id = c.table_id)\n" +
            "WHERE t.table_id = :tableId\n" +
            "ORDER BY c.ordinal_position")
    List<TableColumn> getTableColumns(@Bind("tableId") long tableId);

    @SqlQuery("SELECT schema_name, table_name\n" +
            "FROM tables\n" +
            "WHERE (schema_name = :schemaName OR :schemaName IS NULL)")
    @Mapper(SchemaTableNameMapper.class)
    List<SchemaTableName> listTables(
            @Bind("schemaName") String schemaName);

    @SqlQuery("SELECT DISTINCT schema_name FROM tables")
    List<String> listSchemaNames();

    @SqlQuery("SELECT t.schema_name, t.table_name, c.column_id, c.column_name, c.data_type\n" +
            "FROM tables t\n" +
            "JOIN columns c ON (t.table_id = c.table_id)\n" +
            "WHERE (schema_name = :schemaName OR :schemaName IS NULL)\n" +
            "  AND (table_name = :tableName OR :tableName IS NULL)\n" +
            "ORDER BY schema_name, table_name, ordinal_position")
    List<TableColumn> listTableColumns(
            @Bind("schemaName") String schemaName,
            @Bind("tableName") String tableName);

    @SqlQuery("SELECT t.schema_name, t.table_name, c.column_id, c.column_name, c.data_type\n" +
            "FROM tables t\n" +
            "JOIN columns c ON (t.table_id = c.table_id)\n" +
            "WHERE t.table_id = :tableId\n" +
            "ORDER BY c.ordinal_position")
    List<TableColumn> listTableColumns(@Bind("tableId") long tableId);

    @SqlQuery("SELECT t.schema_name, t.table_name,\n" +
            "  c.column_id, c.column_name, c.ordinal_position, c.data_type\n" +
            "FROM tables t\n" +
            "JOIN columns c ON (t.table_id = c.table_id)\n" +
            "WHERE t.table_id = :tableId\n" +
            "  AND c.sort_ordinal_position IS NOT NULL\n" +
            "ORDER BY c.sort_ordinal_position")
    List<TableColumn> listSortColumns(@Bind("tableId") long tableId);

    @SqlQuery("SELECT schema_name, table_name, data\n" +
            "FROM views\n" +
            "WHERE (schema_name = :schemaName OR :schemaName IS NULL)")
    @Mapper(SchemaTableNameMapper.class)
    List<SchemaTableName> listViews(
            @Bind("schemaName") String schemaName);

    @SqlQuery("SELECT schema_name, table_name, data\n" +
            "FROM views\n" +
            "WHERE (schema_name = :schemaName OR :schemaName IS NULL)\n" +
            "  AND (table_name = :tableName OR :tableName IS NULL)\n" +
            "ORDER BY schema_name, table_name\n")
    @Mapper(ViewResult.Mapper.class)
    List<ViewResult> getViews(
            @Bind("schemaName") String schemaName,
            @Bind("tableName") String tableName);

    @SqlUpdate("INSERT INTO tables (schema_name, table_name, compaction_enabled)\n" +
            "VALUES (:schemaName, :tableName, :compactionEnabled)")
    @GetGeneratedKeys
    long insertTable(
            @Bind("schemaName") String schemaName,
            @Bind("tableName") String tableName,
            @Bind("compactionEnabled") boolean compactionEnabled);

    @SqlUpdate("INSERT INTO columns (table_id, column_id, column_name, ordinal_position, data_type, sort_ordinal_position)\n" +
            "VALUES (:tableId, :columnId, :columnName, :ordinalPosition, :dataType, :sortOrdinalPosition)")
    void insertColumn(
            @Bind("tableId") long tableId,
            @Bind("columnId") long columnId,
            @Bind("columnName") String columnName,
            @Bind("ordinalPosition") int ordinalPosition,
            @Bind("dataType") String dataType,
            @Bind("sortOrdinalPosition") Integer sortOrdinalPosition);

    @SqlUpdate("UPDATE tables SET\n" +
            "  schema_name = :newSchemaName\n" +
            ", table_name = :newTableName\n" +
            "WHERE table_id = :tableId")
    void renameTable(
            @Bind("tableId") long tableId,
            @Bind("newSchemaName") String newSchemaName,
            @Bind("newTableName") String newTableName);

    @SqlUpdate("UPDATE columns SET column_name = :target\n" +
            "WHERE table_id = :tableId\n" +
            "  AND column_id = :columnId")
    void renameColumn(
            @Bind("tableId") long tableId,
            @Bind("columnId") long columnId,
            @Bind("target") String target);

    @SqlUpdate("INSERT INTO views (schema_name, table_name, data)\n" +
            "VALUES (:schemaName, :tableName, :data)")
    void insertView(
            @Bind("schemaName") String schemaName,
            @Bind("tableName") String tableName,
            @Bind("data") String data);

    @SqlUpdate("DELETE FROM tables WHERE table_id = :tableId")
    int dropTable(@Bind("tableId") long tableId);

    @SqlUpdate("DELETE FROM columns WHERE table_id = :tableId")
    int dropColumns(@Bind("tableId") long tableId);

    @SqlUpdate("DELETE FROM views\n" +
            "WHERE schema_name = :schemaName\n" +
            "  AND table_name = :tableName")
    int dropView(
            @Bind("schemaName") String schemaName,
            @Bind("tableName") String tableName);

    // JDBI returns 0 as the column_id when the temporal_column_id is set to NULL
    // jdbi issue https://github.com/jdbi/jdbi/issues/154
    @SqlQuery("SELECT temporal_column_id\n" +
            "FROM tables\n" +
            "WHERE table_id = :tableId\n" +
            "  AND temporal_column_id IS NOT NULL")
    Long getTemporalColumnId(@Bind("tableId") long tableId);

    @SqlUpdate("UPDATE tables SET\n" +
            "temporal_column_id = :columnId\n" +
            "WHERE table_id = :tableId")
    void updateTemporalColumnId(
            @Bind("tableId") long tableId,
            @Bind("columnId") long columnId);

    @SqlQuery("SELECT compaction_enabled FROM tables WHERE table_id = :tableId")
    boolean isCompactionEnabled(@Bind("tableId") long tableId);

    @SqlQuery("SELECT table_id FROM tables WHERE table_id = :tableId FOR UPDATE")
    Long getLockedTableId(@Bind("tableId") long tableId);
}
