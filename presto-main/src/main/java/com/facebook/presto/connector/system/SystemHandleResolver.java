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

import com.facebook.presto.spi.ColumnHandle;
import com.facebook.presto.spi.ConnectorHandleResolver;
import com.facebook.presto.spi.ConnectorSplit;
import com.facebook.presto.spi.ConnectorTableHandle;
import com.facebook.presto.spi.ConnectorTableLayoutHandle;
import com.facebook.presto.spi.connector.ConnectorTransactionHandle;

import static java.util.Objects.requireNonNull;

public class SystemHandleResolver
        implements ConnectorHandleResolver
{
    private final String connectorId;

    public SystemHandleResolver(String connectorId)
    {
        this.connectorId = requireNonNull(connectorId, "connectorId is null");
    }

    @Override
    public boolean canHandle(ConnectorTableHandle tableHandle)
    {
        return (tableHandle instanceof SystemTableHandle) &&
                ((SystemTableHandle) tableHandle).getConnectorId().equals(connectorId);
    }

    @Override
    public boolean canHandle(ColumnHandle columnHandle)
    {
        return (columnHandle instanceof SystemColumnHandle) &&
                ((SystemColumnHandle) columnHandle).getConnectorId().equals(connectorId);
    }

    @Override
    public boolean canHandle(ConnectorSplit split)
    {
        return (split instanceof SystemSplit) &&
                ((SystemSplit) split).getConnectorId().equals(connectorId);
    }

    @Override
    public boolean canHandle(ConnectorTableLayoutHandle handle)
    {
        return (handle instanceof SystemTableLayoutHandle) &&
                ((SystemTableLayoutHandle) handle).getConnectorId().equals(connectorId);
    }

    @Override
    public boolean canHandle(ConnectorTransactionHandle transactionHandle)
    {
        return (transactionHandle instanceof SystemTransactionHandle) &&
                ((SystemTransactionHandle) transactionHandle).getConnectorId().equals(connectorId);
    }

    @Override
    public Class<? extends ConnectorTableHandle> getTableHandleClass()
    {
        return SystemTableHandle.class;
    }

    @Override
    public Class<? extends ConnectorTableLayoutHandle> getTableLayoutHandleClass()
    {
        return SystemTableLayoutHandle.class;
    }

    @Override
    public Class<? extends ColumnHandle> getColumnHandleClass()
    {
        return SystemColumnHandle.class;
    }

    @Override
    public Class<? extends ConnectorSplit> getSplitClass()
    {
        return SystemSplit.class;
    }

    @Override
    public Class<? extends ConnectorTransactionHandle> getTransactionHandleClass()
    {
        return SystemTransactionHandle.class;
    }
}
