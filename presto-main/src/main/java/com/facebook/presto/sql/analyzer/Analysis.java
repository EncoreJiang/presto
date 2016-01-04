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

import com.facebook.presto.metadata.QualifiedObjectName;
import com.facebook.presto.metadata.Signature;
import com.facebook.presto.metadata.TableHandle;
import com.facebook.presto.spi.ColumnHandle;
import com.facebook.presto.spi.type.Type;
import com.facebook.presto.sql.tree.Delete;
import com.facebook.presto.sql.tree.Expression;
import com.facebook.presto.sql.tree.FunctionCall;
import com.facebook.presto.sql.tree.InPredicate;
import com.facebook.presto.sql.tree.Join;
import com.facebook.presto.sql.tree.Node;
import com.facebook.presto.sql.tree.Query;
import com.facebook.presto.sql.tree.QuerySpecification;
import com.facebook.presto.sql.tree.Relation;
import com.facebook.presto.sql.tree.SampledRelation;
import com.facebook.presto.sql.tree.SubqueryExpression;
import com.facebook.presto.sql.tree.Table;
import com.google.common.base.Preconditions;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.SetMultimap;

import javax.annotation.concurrent.Immutable;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

import static com.google.common.base.Preconditions.checkArgument;
import static java.util.Objects.requireNonNull;

public class Analysis
{
    private Query query;
    private String updateType;

    private final IdentityHashMap<Table, Query> namedQueries = new IdentityHashMap<>();

    private RelationType outputDescriptor;
    private final IdentityHashMap<Node, RelationType> outputDescriptors = new IdentityHashMap<>();
    private final IdentityHashMap<Expression, Integer> resolvedNames = new IdentityHashMap<>();

    private final IdentityHashMap<QuerySpecification, List<FunctionCall>> aggregates = new IdentityHashMap<>();
    private final IdentityHashMap<QuerySpecification, List<List<FieldOrExpression>>> groupByExpressions = new IdentityHashMap<>();
    private final IdentityHashMap<Node, Expression> where = new IdentityHashMap<>();
    private final IdentityHashMap<QuerySpecification, Expression> having = new IdentityHashMap<>();
    private final IdentityHashMap<Node, List<FieldOrExpression>> orderByExpressions = new IdentityHashMap<>();
    private final IdentityHashMap<Node, List<FieldOrExpression>> outputExpressions = new IdentityHashMap<>();
    private final IdentityHashMap<QuerySpecification, List<FunctionCall>> windowFunctions = new IdentityHashMap<>();

    private final IdentityHashMap<Join, Expression> joins = new IdentityHashMap<>();
    private final SetMultimap<Node, InPredicate> inPredicates = HashMultimap.create();
    private final SetMultimap<Node, SubqueryExpression> scalarSubqueries = HashMultimap.create();
    private final IdentityHashMap<Join, JoinInPredicates> joinInPredicates = new IdentityHashMap<>();

    private final IdentityHashMap<Table, TableHandle> tables = new IdentityHashMap<>();

    private final IdentityHashMap<Expression, Type> types = new IdentityHashMap<>();
    private final IdentityHashMap<Expression, Type> coercions = new IdentityHashMap<>();
    private final IdentityHashMap<Relation, Type[]> relationCoercions = new IdentityHashMap<>();
    private final IdentityHashMap<FunctionCall, Signature> functionSignature = new IdentityHashMap<>();

    private final IdentityHashMap<Field, ColumnHandle> columns = new IdentityHashMap<>();

    private final IdentityHashMap<SampledRelation, Double> sampleRatios = new IdentityHashMap<>();

    // for create table
    private Optional<QualifiedObjectName> createTableDestination = Optional.empty();
    private Map<String, Expression> createTableProperties = ImmutableMap.of();
    private boolean createTableAsSelectWithData = true;

    private Optional<Insert> insert = Optional.empty();

    // for delete
    private Optional<Delete> delete = Optional.empty();

    public Query getQuery()
    {
        return query;
    }

    public void setQuery(Query query)
    {
        this.query = query;
    }

    public String getUpdateType()
    {
        return updateType;
    }

    public void setUpdateType(String updateType)
    {
        this.updateType = updateType;
    }

    public boolean isCreateTableAsSelectWithData()
    {
        return createTableAsSelectWithData;
    }

    public void setCreateTableAsSelectWithData(boolean createTableAsSelectWithData)
    {
        this.createTableAsSelectWithData = createTableAsSelectWithData;
    }

    public void addResolvedNames(Map<Expression, Integer> mappings)
    {
        resolvedNames.putAll(mappings);
    }

    public Optional<Integer> getFieldIndex(Expression expression)
    {
        return Optional.ofNullable(resolvedNames.get(expression));
    }

    public void setAggregates(QuerySpecification node, List<FunctionCall> aggregates)
    {
        this.aggregates.put(node, aggregates);
    }

    public List<FunctionCall> getAggregates(QuerySpecification query)
    {
        return aggregates.get(query);
    }

    public IdentityHashMap<Expression, Type> getTypes()
    {
        return new IdentityHashMap<>(types);
    }

    public Type getType(Expression expression)
    {
        checkArgument(types.containsKey(expression), "Expression not analyzed: %s", expression);
        return types.get(expression);
    }

    public Type[] getRelationCoercion(Relation relation)
    {
        return relationCoercions.get(relation);
    }

    public void addRelationCoercion(Relation relation, Type[] types)
    {
        relationCoercions.put(relation, types);
    }

    public IdentityHashMap<Expression, Type> getCoercions()
    {
        return coercions;
    }

    public Type getCoercion(Expression expression)
    {
        return coercions.get(expression);
    }

    public void setGroupingSets(QuerySpecification node, List<List<FieldOrExpression>> expressions)
    {
        groupByExpressions.put(node, expressions);
    }

    public List<List<FieldOrExpression>> getGroupingSets(QuerySpecification node)
    {
        return groupByExpressions.get(node);
    }

    public void setWhere(Node node, Expression expression)
    {
        where.put(node, expression);
    }

    public Expression getWhere(QuerySpecification node)
    {
        return where.get(node);
    }

    public void setOrderByExpressions(Node node, List<FieldOrExpression> items)
    {
        orderByExpressions.put(node, items);
    }

    public List<FieldOrExpression> getOrderByExpressions(Node node)
    {
        return orderByExpressions.get(node);
    }

    public void setOutputExpressions(Node node, List<FieldOrExpression> expressions)
    {
        outputExpressions.put(node, expressions);
    }

    public List<FieldOrExpression> getOutputExpressions(Node node)
    {
        return outputExpressions.get(node);
    }

    public void setHaving(QuerySpecification node, Expression expression)
    {
        having.put(node, expression);
    }

    public void setJoinCriteria(Join node, Expression criteria)
    {
        joins.put(node, criteria);
    }

    public Expression getJoinCriteria(Join join)
    {
        return joins.get(join);
    }

    public void recordSubqueries(Node node, ExpressionAnalysis expressionAnalysis)
    {
        this.inPredicates.putAll(node, expressionAnalysis.getSubqueryInPredicates());
        this.scalarSubqueries.putAll(node, expressionAnalysis.getScalarSubqueries());
    }

    public Set<InPredicate> getInPredicates(Node node)
    {
        return inPredicates.get(node);
    }

    public Set<SubqueryExpression> getScalarSubqueries(Node node)
    {
        return scalarSubqueries.get(node);
    }

    public void addJoinInPredicates(Join node, JoinInPredicates joinInPredicates)
    {
        this.joinInPredicates.put(node, joinInPredicates);
    }

    public JoinInPredicates getJoinInPredicates(Join node)
    {
        return joinInPredicates.get(node);
    }

    public void setWindowFunctions(QuerySpecification node, List<FunctionCall> functions)
    {
        windowFunctions.put(node, functions);
    }

    public Map<QuerySpecification, List<FunctionCall>> getWindowFunctions()
    {
        return windowFunctions;
    }

    public List<FunctionCall> getWindowFunctions(QuerySpecification query)
    {
        return windowFunctions.get(query);
    }

    public void setOutputDescriptor(RelationType descriptor)
    {
        outputDescriptor = descriptor;
    }

    public RelationType getOutputDescriptor()
    {
        return outputDescriptor;
    }

    public void setOutputDescriptor(Node node, RelationType descriptor)
    {
        outputDescriptors.put(node, descriptor);
    }

    public RelationType getOutputDescriptor(Node node)
    {
        Preconditions.checkState(outputDescriptors.containsKey(node), "Output descriptor missing for %s. Broken analysis?", node);
        return outputDescriptors.get(node);
    }

    public TableHandle getTableHandle(Table table)
    {
        return tables.get(table);
    }

    public void registerTable(Table table, TableHandle handle)
    {
        tables.put(table, handle);
    }

    public Signature getFunctionSignature(FunctionCall function)
    {
        return functionSignature.get(function);
    }

    public void addFunctionSignatures(IdentityHashMap<FunctionCall, Signature> infos)
    {
        functionSignature.putAll(infos);
    }

    public Set<Expression> getColumnReferences()
    {
        return resolvedNames.keySet();
    }

    public void addTypes(IdentityHashMap<Expression, Type> types)
    {
        this.types.putAll(types);
    }

    public void addCoercion(Expression expression, Type type)
    {
        this.coercions.put(expression, type);
    }

    public void addCoercions(IdentityHashMap<Expression, Type> coercions)
    {
        this.coercions.putAll(coercions);
    }

    public Expression getHaving(QuerySpecification query)
    {
        return having.get(query);
    }

    public void setColumn(Field field, ColumnHandle handle)
    {
        columns.put(field, handle);
    }

    public ColumnHandle getColumn(Field field)
    {
        return columns.get(field);
    }

    public void setCreateTableDestination(QualifiedObjectName destination)
    {
        this.createTableDestination = Optional.of(destination);
    }

    public Optional<QualifiedObjectName> getCreateTableDestination()
    {
        return createTableDestination;
    }

    public void setCreateTableProperties(Map<String, Expression> createTableProperties)
    {
        this.createTableProperties = createTableProperties;
    }

    public Map<String, Expression> getCreateTableProperties()
    {
        return createTableProperties;
    }

    public void setInsert(Insert insert)
    {
        this.insert = Optional.of(insert);
    }

    public Optional<Insert> getInsert()
    {
        return insert;
    }

    public void setDelete(Delete delete)
    {
        this.delete = Optional.of(delete);
    }

    public Optional<Delete> getDelete()
    {
        return delete;
    }

    public Query getNamedQuery(Table table)
    {
        return namedQueries.get(table);
    }

    public void registerNamedQuery(Table tableReference, Query query)
    {
        requireNonNull(tableReference, "tableReference is null");
        requireNonNull(query, "query is null");

        namedQueries.put(tableReference, query);
    }

    public void setSampleRatio(SampledRelation relation, double ratio)
    {
        sampleRatios.put(relation, ratio);
    }

    public double getSampleRatio(SampledRelation relation)
    {
        Preconditions.checkState(sampleRatios.containsKey(relation), "Sample ratio missing for %s. Broken analysis?", relation);
        return sampleRatios.get(relation);
    }

    public static class JoinInPredicates
    {
        private final Set<InPredicate> leftInPredicates;
        private final Set<InPredicate> rightInPredicates;

        public JoinInPredicates(Set<InPredicate> leftInPredicates, Set<InPredicate> rightInPredicates)
        {
            this.leftInPredicates = ImmutableSet.copyOf(requireNonNull(leftInPredicates, "leftInPredicates is null"));
            this.rightInPredicates = ImmutableSet.copyOf(requireNonNull(rightInPredicates, "rightInPredicates is null"));
        }

        public Set<InPredicate> getLeftInPredicates()
        {
            return leftInPredicates;
        }

        public Set<InPredicate> getRightInPredicates()
        {
            return rightInPredicates;
        }

        @Override
        public int hashCode()
        {
            return Objects.hash(leftInPredicates, rightInPredicates);
        }

        @Override
        public boolean equals(Object obj)
        {
            if (this == obj) {
                return true;
            }
            if (obj == null || getClass() != obj.getClass()) {
                return false;
            }
            final JoinInPredicates other = (JoinInPredicates) obj;
            return Objects.equals(this.leftInPredicates, other.leftInPredicates) &&
                    Objects.equals(this.rightInPredicates, other.rightInPredicates);
        }
    }

    @Immutable
    public static final class Insert
    {
        private final TableHandle target;
        private final List<ColumnHandle> columns;

        public Insert(TableHandle target, List<ColumnHandle> columns)
        {
            this.target = requireNonNull(target, "target is null");
            this.columns = requireNonNull(columns, "columns is null");
            checkArgument(columns.size() > 0, "No columns given to insert");
        }

        public List<ColumnHandle> getColumns()
        {
            return columns;
        }

        public TableHandle getTarget()
        {
            return target;
        }
    }
}
