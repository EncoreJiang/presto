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

import org.skife.jdbi.v2.sqlobject.SqlUpdate;

public interface SchemaDao
{
    @SqlUpdate("CREATE TABLE IF NOT EXISTS tables (\n" +
            "  table_id BIGINT PRIMARY KEY AUTO_INCREMENT,\n" +
            "  schema_name VARCHAR(255) NOT NULL,\n" +
            "  table_name VARCHAR(255) NOT NULL,\n" +
            "  temporal_column_id BIGINT,\n" +
            "  compaction_enabled BOOLEAN NOT NULL,\n" +
            "  UNIQUE (schema_name, table_name)\n" +
            ")")
    void createTableTables();

    @SqlUpdate("CREATE TABLE IF NOT EXISTS columns (\n" +
            "  table_id BIGINT NOT NULL,\n" +
            "  column_id BIGINT NOT NULL,\n" +
            "  column_name VARCHAR(255) NOT NULL,\n" +
            "  ordinal_position INT NOT NULL,\n" +
            "  data_type VARCHAR(255) NOT NULL,\n" +
            "  sort_ordinal_position INT,\n" +
            "  PRIMARY KEY (table_id, column_id),\n" +
            "  UNIQUE (table_id, column_name),\n" +
            "  UNIQUE (table_id, ordinal_position),\n" +
            "  UNIQUE (table_id, sort_ordinal_position),\n" +
            "  FOREIGN KEY (table_id) REFERENCES tables (table_id)\n" +
            ")")
    void createTableColumns();

    @SqlUpdate("CREATE TABLE IF NOT EXISTS views (\n" +
            "  schema_name VARCHAR(255) NOT NULL,\n" +
            "  table_name VARCHAR(255) NOT NULL,\n" +
            "  data TEXT NOT NULL,\n" +
            "  PRIMARY KEY (schema_name, table_name)\n" +
            ")")
    void createTableViews();

    @SqlUpdate("CREATE TABLE IF NOT EXISTS nodes (\n" +
            "  node_id INT PRIMARY KEY AUTO_INCREMENT,\n" +
            "  node_identifier VARCHAR(255) NOT NULL,\n" +
            "  UNIQUE (node_identifier)\n" +
            ")")
    void createTableNodes();

    @SqlUpdate("CREATE TABLE IF NOT EXISTS shards (\n" +
            "  shard_id BIGINT PRIMARY KEY AUTO_INCREMENT,\n" +
            "  shard_uuid BINARY(16) NOT NULL,\n" +
            "  table_id BIGINT NOT NULL,\n" +
            "  create_time DATETIME NOT NULL,\n" +
            "  row_count BIGINT NOT NULL,\n" +
            "  compressed_size BIGINT NOT NULL,\n" +
            "  uncompressed_size BIGINT NOT NULL,\n" +
            "  UNIQUE (shard_uuid),\n" +
            "  FOREIGN KEY (table_id) REFERENCES tables (table_id)\n" +
            ")")
    void createTableShards();

    @SqlUpdate("CREATE TABLE IF NOT EXISTS shard_nodes (\n" +
            "  shard_id BIGINT NOT NULL,\n" +
            "  node_id INT NOT NULL,\n" +
            "  PRIMARY KEY (shard_id, node_id),\n" +
            "  FOREIGN KEY (shard_id) REFERENCES shards (shard_id),\n" +
            "  FOREIGN KEY (node_id) REFERENCES nodes (node_id)\n" +
            ")")
    void createTableShardNodes();

    @SqlUpdate("CREATE TABLE IF NOT EXISTS external_batches (\n" +
            "  external_batch_id VARCHAR(255) PRIMARY KEY,\n" +
            "  successful BOOLEAN NOT NULL\n" +
            ")")
    void createTableExternalBatches();

    @SqlUpdate("CREATE TABLE IF NOT EXISTS transactions (\n" +
            "  transaction_id BIGINT PRIMARY KEY AUTO_INCREMENT,\n" +
            "  successful BOOLEAN,\n" +
            "  start_time DATETIME NOT NULL,\n" +
            "  end_time DATETIME\n" +
            ")")
    void createTableTransactions();

    @SqlUpdate("CREATE TABLE IF NOT EXISTS created_shards (\n" +
            "  shard_uuid BINARY(16) NOT NULL,\n" +
            "  transaction_id BIGINT NOT NULL,\n" +
            "  PRIMARY KEY (shard_uuid),\n" +
            "  FOREIGN KEY (transaction_id) REFERENCES transactions (transaction_id)\n" +
            ")")
    void createTableCreatedShards();

    @SqlUpdate("CREATE TABLE IF NOT EXISTS created_shard_nodes (\n" +
            "  shard_uuid BINARY(16) NOT NULL,\n" +
            "  node_id INT NOT NULL,\n" +
            "  transaction_id BIGINT NOT NULL,\n" +
            "  PRIMARY KEY (shard_uuid, node_id),\n" +
            "  FOREIGN KEY (node_id) REFERENCES nodes (node_id),\n" +
            "  FOREIGN KEY (transaction_id) REFERENCES transactions (transaction_id)\n" +
            ")")
    void createTableCreatedShardNodes();

    @SqlUpdate("CREATE TABLE IF NOT EXISTS deleted_shards (\n" +
            "  shard_uuid BINARY(16) NOT NULL,\n" +
            "  delete_time DATETIME NOT NULL,\n" +
            "  clean_time DATETIME,\n" +
            "  purge_time DATETIME\n" +
            ")")
    void createTableDeletedShards();

    @SqlUpdate("CREATE TABLE IF NOT EXISTS deleted_shard_nodes (\n" +
            "  shard_uuid BINARY(16) NOT NULL,\n" +
            "  node_id INT NOT NULL,\n" +
            "  delete_time DATETIME NOT NULL,\n" +
            "  clean_time DATETIME,\n" +
            "  purge_time DATETIME,\n" +
            "  FOREIGN KEY (node_id) REFERENCES nodes (node_id)\n" +
            ")")
    void createTableDeletedShardNodes();
}
