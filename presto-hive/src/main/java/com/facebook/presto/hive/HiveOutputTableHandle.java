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
package com.facebook.presto.hive;

import com.facebook.presto.spi.ConnectorOutputTableHandle;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.common.collect.ImmutableList;

import java.util.List;
import java.util.OptionalInt;

import static java.util.Objects.requireNonNull;

public class HiveOutputTableHandle
        extends HiveWritableTableHandle
        implements ConnectorOutputTableHandle
{
    private final List<String> partitionedBy;
    private final String tableOwner;
    private final OptionalInt retentionDays;

    @JsonCreator
    public HiveOutputTableHandle(
            @JsonProperty("clientId") String clientId,
            @JsonProperty("schemaName") String schemaName,
            @JsonProperty("tableName") String tableName,
            @JsonProperty("inputColumns") List<HiveColumnHandle> inputColumns,
            @JsonProperty("filePrefix") String filePrefix,
            @JsonProperty("locationHandle") LocationHandle locationHandle,
            @JsonProperty("hiveStorageFormat") HiveStorageFormat hiveStorageFormat,
            @JsonProperty("partitionedBy") List<String> partitionedBy,
            @JsonProperty("tableOwner") String tableOwner,
            @JsonProperty("retentionDays") OptionalInt retentionDays)
    {
        super(
                clientId,
                schemaName,
                tableName,
                inputColumns,
                filePrefix,
                requireNonNull(locationHandle, "locationHandle is null"),
                hiveStorageFormat);

        this.partitionedBy = ImmutableList.copyOf(requireNonNull(partitionedBy, "partitionedBy is null"));
        this.tableOwner = requireNonNull(tableOwner, "tableOwner is null");
        this.retentionDays = requireNonNull(retentionDays, "retentionDays is null");
    }

    @JsonProperty
    public List<String> getPartitionedBy()
    {
        return partitionedBy;
    }

    @JsonProperty
    public String getTableOwner()
    {
        return tableOwner;
    }

    @JsonProperty
    public OptionalInt getRetentionDays()
    {
        return retentionDays;
    }
}
