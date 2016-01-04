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
package com.facebook.presto.hive.parquet.predicate;

import parquet.column.ColumnDescriptor;
import parquet.column.page.DictionaryPage;

public class ParquetDictionaryDescriptor
{
    private final ColumnDescriptor columnDescriptor;
    private final DictionaryPage dictionaryPage;

    public ParquetDictionaryDescriptor(ColumnDescriptor columnDescriptor, DictionaryPage dictionaryPage)
    {
        this.columnDescriptor = columnDescriptor;
        this.dictionaryPage = dictionaryPage;
    }

    public ColumnDescriptor getColumnDescriptor()
    {
        return columnDescriptor;
    }

    public DictionaryPage getDictionaryPage()
    {
        return dictionaryPage;
    }
}
