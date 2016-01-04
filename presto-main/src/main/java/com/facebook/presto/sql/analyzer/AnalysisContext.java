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
package com.facebook.presto.sql.analyzer;

import com.facebook.presto.sql.tree.Query;
import com.google.common.base.Preconditions;

import java.util.HashMap;
import java.util.Map;

public class AnalysisContext
{
    private final AnalysisContext parent;
    private final Map<String, Query> namedQueries = new HashMap<>();
    private RelationType lateralTupleDescriptor = new RelationType();
    private boolean approximate;

    public AnalysisContext(AnalysisContext parent)
    {
        this.parent = parent;
        this.approximate = parent.approximate;
    }

    public AnalysisContext()
    {
        parent = null;
    }

    public void setLateralTupleDescriptor(RelationType lateralTupleDescriptor)
    {
        this.lateralTupleDescriptor = lateralTupleDescriptor;
    }

    public RelationType getLateralTupleDescriptor()
    {
        return lateralTupleDescriptor;
    }

    public boolean isApproximate()
    {
        return approximate;
    }

    public void setApproximate(boolean approximate)
    {
        this.approximate = approximate;
    }

    public void addNamedQuery(String name, Query query)
    {
        Preconditions.checkState(!namedQueries.containsKey(name), "Named query already registered: %s", name);
        namedQueries.put(name, query);
    }

    public Query getNamedQuery(String name)
    {
        Query result = namedQueries.get(name);

        if (result == null && parent != null) {
            return parent.getNamedQuery(name);
        }

        return result;
    }

    public boolean isNamedQueryDeclared(String name)
    {
        return namedQueries.containsKey(name);
    }
}
