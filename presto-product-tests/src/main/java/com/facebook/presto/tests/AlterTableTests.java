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
package com.facebook.presto.tests;

import com.facebook.presto.tests.ImmutableTpchTablesRequirements.ImmutableNationTable;
import com.facebook.presto.tests.utils.PrestoDDLUtils.Table;
import com.teradata.tempto.ProductTest;
import com.teradata.tempto.Requires;
import org.testng.annotations.Test;

import java.io.IOException;

import static com.facebook.presto.tests.TestGroups.ALTER_TABLE;
import static com.facebook.presto.tests.TestGroups.SMOKE;
import static com.facebook.presto.tests.utils.PrestoDDLUtils.createPrestoTable;
import static com.teradata.tempto.assertions.QueryAssert.assertThat;
import static com.teradata.tempto.query.QueryExecutor.query;
import static com.teradata.tempto.query.QueryType.UPDATE;
import static java.lang.String.format;

@Requires(ImmutableNationTable.class)
public class AlterTableTests
        extends ProductTest
{
    @Test(groups = {ALTER_TABLE, SMOKE})
    public void renameTable()
            throws IOException
    {
        String tableName = "to_be_renamed_table_name";
        String renamedTableName = "renamed_table_name";
        try (Table table = createPrestoTable(tableName, "CREATE TABLE %s AS SELECT * FROM nation")) {
            assertThat(query(format("ALTER TABLE %s RENAME TO %s", table.getNameInDatabase(), renamedTableName), UPDATE))
                    .hasRowsCount(1);
            assertThat(query(format("SELECT * FROM %s", renamedTableName)))
                    .hasRowsCount(25);
            // rename back to original name
            assertThat(query(format("ALTER TABLE %s RENAME TO %s", renamedTableName, table.getNameInDatabase()), UPDATE))
                    .hasRowsCount(1);
        }
    }
}
