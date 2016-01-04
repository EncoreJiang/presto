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
package com.facebook.presto.connector.system;

import com.facebook.presto.spi.ConnectorHandleResolver;
import com.facebook.presto.spi.NodeManager;
import com.facebook.presto.spi.SystemTable;
import com.facebook.presto.spi.connector.ConnectorMetadata;
import com.facebook.presto.spi.connector.ConnectorRecordSetProvider;
import com.facebook.presto.spi.connector.ConnectorSplitManager;
import com.facebook.presto.spi.connector.ConnectorTransactionHandle;
import com.facebook.presto.spi.transaction.IsolationLevel;
import com.facebook.presto.transaction.InternalConnector;
import com.facebook.presto.transaction.TransactionHandle;
import com.facebook.presto.transaction.TransactionId;

import java.util.Set;
import java.util.function.Function;

import static java.util.Objects.requireNonNull;

public class SystemConnector
        implements InternalConnector
{
    private final String connectorId;
    private final SystemHandleResolver handleResolver;
    private final ConnectorMetadata metadata;
    private final ConnectorSplitManager splitManager;
    private final ConnectorRecordSetProvider recordSetProvider;
    private final Function<TransactionId, TransactionHandle> transactionHandleFunction;

    public SystemConnector(
            String connectorId,
            NodeManager nodeManager,
            Set<SystemTable> tables,
            Function<TransactionId, TransactionHandle> transactionHandleFunction)
    {
        requireNonNull(connectorId, "connectorId is null");
        requireNonNull(nodeManager, "nodeManager is null");
        requireNonNull(tables, "tables is null");
        requireNonNull(transactionHandleFunction, "transactionHandleFunction is null");

        this.connectorId = connectorId;
        this.handleResolver = new SystemHandleResolver(connectorId);
        this.metadata = new SystemTablesMetadata(connectorId, tables);
        this.splitManager = new SystemSplitManager(nodeManager, tables);
        this.recordSetProvider = new SystemRecordSetProvider(tables);
        this.transactionHandleFunction = transactionHandleFunction;
    }

    @Override
    public ConnectorTransactionHandle beginTransaction(TransactionId transactionId, IsolationLevel isolationLevel, boolean readOnly)
    {
        return new SystemTransactionHandle(connectorId, transactionHandleFunction.apply(transactionId));
    }

    @Override
    public ConnectorMetadata getMetadata(ConnectorTransactionHandle transactionHandle)
    {
        return metadata;
    }

    @Override
    public ConnectorSplitManager getSplitManager()
    {
        return splitManager;
    }

    @Override
    public ConnectorHandleResolver getHandleResolver()
    {
        return handleResolver;
    }

    @Override
    public ConnectorRecordSetProvider getRecordSetProvider()
    {
        return recordSetProvider;
    }
}
