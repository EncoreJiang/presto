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

import com.facebook.presto.Session;
import com.facebook.presto.metadata.FunctionKind;
import com.facebook.presto.metadata.Metadata;
import com.facebook.presto.metadata.MetadataUtil;
import com.facebook.presto.metadata.QualifiedObjectName;
import com.facebook.presto.metadata.SessionPropertyManager.SessionPropertyValue;
import com.facebook.presto.metadata.SqlFunction;
import com.facebook.presto.metadata.TableHandle;
import com.facebook.presto.metadata.TableMetadata;
import com.facebook.presto.metadata.ViewDefinition;
import com.facebook.presto.security.AccessControl;
import com.facebook.presto.security.AllowAllAccessControl;
import com.facebook.presto.security.ViewAccessControl;
import com.facebook.presto.spi.ColumnHandle;
import com.facebook.presto.spi.ColumnMetadata;
import com.facebook.presto.spi.PrestoException;
import com.facebook.presto.spi.SchemaTableName;
import com.facebook.presto.spi.security.Identity;
import com.facebook.presto.spi.type.BigintType;
import com.facebook.presto.spi.type.Type;
import com.facebook.presto.spi.type.TypeSignature;
import com.facebook.presto.sql.ExpressionUtils;
import com.facebook.presto.sql.parser.ParsingException;
import com.facebook.presto.sql.parser.SqlParser;
import com.facebook.presto.sql.planner.DependencyExtractor;
import com.facebook.presto.sql.planner.ExpressionInterpreter;
import com.facebook.presto.sql.planner.NoOpSymbolResolver;
import com.facebook.presto.sql.planner.Symbol;
import com.facebook.presto.sql.planner.optimizations.CanonicalizeExpressions;
import com.facebook.presto.sql.tree.AliasedRelation;
import com.facebook.presto.sql.tree.AllColumns;
import com.facebook.presto.sql.tree.Cast;
import com.facebook.presto.sql.tree.ComparisonExpression;
import com.facebook.presto.sql.tree.CreateTableAsSelect;
import com.facebook.presto.sql.tree.CreateView;
import com.facebook.presto.sql.tree.DefaultTraversalVisitor;
import com.facebook.presto.sql.tree.Delete;
import com.facebook.presto.sql.tree.DereferenceExpression;
import com.facebook.presto.sql.tree.Except;
import com.facebook.presto.sql.tree.Explain;
import com.facebook.presto.sql.tree.ExplainFormat;
import com.facebook.presto.sql.tree.ExplainOption;
import com.facebook.presto.sql.tree.ExplainType;
import com.facebook.presto.sql.tree.Expression;
import com.facebook.presto.sql.tree.FrameBound;
import com.facebook.presto.sql.tree.FunctionCall;
import com.facebook.presto.sql.tree.GroupingElement;
import com.facebook.presto.sql.tree.Insert;
import com.facebook.presto.sql.tree.Intersect;
import com.facebook.presto.sql.tree.Join;
import com.facebook.presto.sql.tree.JoinCriteria;
import com.facebook.presto.sql.tree.JoinOn;
import com.facebook.presto.sql.tree.JoinUsing;
import com.facebook.presto.sql.tree.LikePredicate;
import com.facebook.presto.sql.tree.LongLiteral;
import com.facebook.presto.sql.tree.NaturalJoin;
import com.facebook.presto.sql.tree.Node;
import com.facebook.presto.sql.tree.QualifiedName;
import com.facebook.presto.sql.tree.QualifiedNameReference;
import com.facebook.presto.sql.tree.Query;
import com.facebook.presto.sql.tree.QuerySpecification;
import com.facebook.presto.sql.tree.Relation;
import com.facebook.presto.sql.tree.Row;
import com.facebook.presto.sql.tree.SampledRelation;
import com.facebook.presto.sql.tree.SelectItem;
import com.facebook.presto.sql.tree.ShowCatalogs;
import com.facebook.presto.sql.tree.ShowColumns;
import com.facebook.presto.sql.tree.ShowFunctions;
import com.facebook.presto.sql.tree.ShowPartitions;
import com.facebook.presto.sql.tree.ShowSchemas;
import com.facebook.presto.sql.tree.ShowSession;
import com.facebook.presto.sql.tree.ShowTables;
import com.facebook.presto.sql.tree.SimpleGroupBy;
import com.facebook.presto.sql.tree.SingleColumn;
import com.facebook.presto.sql.tree.SortItem;
import com.facebook.presto.sql.tree.Statement;
import com.facebook.presto.sql.tree.StringLiteral;
import com.facebook.presto.sql.tree.Table;
import com.facebook.presto.sql.tree.TableSubquery;
import com.facebook.presto.sql.tree.Union;
import com.facebook.presto.sql.tree.Unnest;
import com.facebook.presto.sql.tree.Use;
import com.facebook.presto.sql.tree.Values;
import com.facebook.presto.sql.tree.Window;
import com.facebook.presto.sql.tree.WindowFrame;
import com.facebook.presto.sql.tree.With;
import com.facebook.presto.sql.tree.WithQuery;
import com.facebook.presto.type.ArrayType;
import com.facebook.presto.type.MapType;
import com.facebook.presto.type.RowType;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Multimap;
import com.google.common.collect.Sets;
import com.google.common.primitives.Ints;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

import static com.facebook.presto.connector.informationSchema.InformationSchemaMetadata.TABLE_COLUMNS;
import static com.facebook.presto.connector.informationSchema.InformationSchemaMetadata.TABLE_INTERNAL_PARTITIONS;
import static com.facebook.presto.connector.informationSchema.InformationSchemaMetadata.TABLE_SCHEMATA;
import static com.facebook.presto.connector.informationSchema.InformationSchemaMetadata.TABLE_TABLES;
import static com.facebook.presto.metadata.FunctionKind.AGGREGATE;
import static com.facebook.presto.metadata.FunctionKind.APPROXIMATE_AGGREGATE;
import static com.facebook.presto.metadata.FunctionKind.WINDOW;
import static com.facebook.presto.metadata.MetadataUtil.createQualifiedObjectName;
import static com.facebook.presto.spi.StandardErrorCode.INVALID_FUNCTION_ARGUMENT;
import static com.facebook.presto.spi.type.BigintType.BIGINT;
import static com.facebook.presto.spi.type.BooleanType.BOOLEAN;
import static com.facebook.presto.sql.QueryUtil.aliased;
import static com.facebook.presto.sql.QueryUtil.aliasedName;
import static com.facebook.presto.sql.QueryUtil.aliasedNullToEmpty;
import static com.facebook.presto.sql.QueryUtil.aliasedYesNoToBoolean;
import static com.facebook.presto.sql.QueryUtil.ascending;
import static com.facebook.presto.sql.QueryUtil.caseWhen;
import static com.facebook.presto.sql.QueryUtil.equal;
import static com.facebook.presto.sql.QueryUtil.functionCall;
import static com.facebook.presto.sql.QueryUtil.logicalAnd;
import static com.facebook.presto.sql.QueryUtil.nameReference;
import static com.facebook.presto.sql.QueryUtil.ordering;
import static com.facebook.presto.sql.QueryUtil.row;
import static com.facebook.presto.sql.QueryUtil.selectAll;
import static com.facebook.presto.sql.QueryUtil.selectList;
import static com.facebook.presto.sql.QueryUtil.simpleQuery;
import static com.facebook.presto.sql.QueryUtil.subquery;
import static com.facebook.presto.sql.QueryUtil.table;
import static com.facebook.presto.sql.QueryUtil.unaliasedName;
import static com.facebook.presto.sql.QueryUtil.values;
import static com.facebook.presto.sql.analyzer.ExpressionAnalyzer.createConstantAnalyzer;
import static com.facebook.presto.sql.analyzer.ExpressionAnalyzer.getExpressionTypes;
import static com.facebook.presto.sql.analyzer.SemanticErrorCode.AMBIGUOUS_ATTRIBUTE;
import static com.facebook.presto.sql.analyzer.SemanticErrorCode.CATALOG_NOT_SPECIFIED;
import static com.facebook.presto.sql.analyzer.SemanticErrorCode.COLUMN_NAME_NOT_SPECIFIED;
import static com.facebook.presto.sql.analyzer.SemanticErrorCode.DUPLICATE_COLUMN_NAME;
import static com.facebook.presto.sql.analyzer.SemanticErrorCode.DUPLICATE_RELATION;
import static com.facebook.presto.sql.analyzer.SemanticErrorCode.INVALID_ORDINAL;
import static com.facebook.presto.sql.analyzer.SemanticErrorCode.INVALID_SCHEMA_NAME;
import static com.facebook.presto.sql.analyzer.SemanticErrorCode.INVALID_WINDOW_FRAME;
import static com.facebook.presto.sql.analyzer.SemanticErrorCode.MISMATCHED_COLUMN_ALIASES;
import static com.facebook.presto.sql.analyzer.SemanticErrorCode.MISMATCHED_SET_COLUMN_TYPES;
import static com.facebook.presto.sql.analyzer.SemanticErrorCode.MISSING_CATALOG;
import static com.facebook.presto.sql.analyzer.SemanticErrorCode.MISSING_COLUMN;
import static com.facebook.presto.sql.analyzer.SemanticErrorCode.MISSING_SCHEMA;
import static com.facebook.presto.sql.analyzer.SemanticErrorCode.MISSING_TABLE;
import static com.facebook.presto.sql.analyzer.SemanticErrorCode.MUST_BE_AGGREGATE_OR_GROUP_BY;
import static com.facebook.presto.sql.analyzer.SemanticErrorCode.MUST_BE_WINDOW_FUNCTION;
import static com.facebook.presto.sql.analyzer.SemanticErrorCode.NESTED_WINDOW;
import static com.facebook.presto.sql.analyzer.SemanticErrorCode.NON_NUMERIC_SAMPLE_PERCENTAGE;
import static com.facebook.presto.sql.analyzer.SemanticErrorCode.NOT_SUPPORTED;
import static com.facebook.presto.sql.analyzer.SemanticErrorCode.ORDER_BY_MUST_BE_IN_SELECT;
import static com.facebook.presto.sql.analyzer.SemanticErrorCode.SCHEMA_NOT_SPECIFIED;
import static com.facebook.presto.sql.analyzer.SemanticErrorCode.TABLE_ALREADY_EXISTS;
import static com.facebook.presto.sql.analyzer.SemanticErrorCode.TYPE_MISMATCH;
import static com.facebook.presto.sql.analyzer.SemanticErrorCode.VIEW_ANALYSIS_ERROR;
import static com.facebook.presto.sql.analyzer.SemanticErrorCode.VIEW_IS_STALE;
import static com.facebook.presto.sql.analyzer.SemanticErrorCode.VIEW_PARSE_ERROR;
import static com.facebook.presto.sql.analyzer.SemanticErrorCode.WILDCARD_WITHOUT_FROM;
import static com.facebook.presto.sql.planner.ExpressionInterpreter.expressionOptimizer;
import static com.facebook.presto.sql.tree.BooleanLiteral.FALSE_LITERAL;
import static com.facebook.presto.sql.tree.BooleanLiteral.TRUE_LITERAL;
import static com.facebook.presto.sql.tree.ComparisonExpression.Type.EQUAL;
import static com.facebook.presto.sql.tree.ExplainFormat.Type.TEXT;
import static com.facebook.presto.sql.tree.ExplainType.Type.LOGICAL;
import static com.facebook.presto.sql.tree.FrameBound.Type.CURRENT_ROW;
import static com.facebook.presto.sql.tree.FrameBound.Type.FOLLOWING;
import static com.facebook.presto.sql.tree.FrameBound.Type.PRECEDING;
import static com.facebook.presto.sql.tree.FrameBound.Type.UNBOUNDED_FOLLOWING;
import static com.facebook.presto.sql.tree.FrameBound.Type.UNBOUNDED_PRECEDING;
import static com.facebook.presto.sql.tree.WindowFrame.Type.RANGE;
import static com.facebook.presto.type.UnknownType.UNKNOWN;
import static com.facebook.presto.util.ImmutableCollectors.toImmutableList;
import static com.facebook.presto.util.ImmutableCollectors.toImmutableSet;
import static com.facebook.presto.util.Types.checkType;
import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.base.Strings.nullToEmpty;
import static com.google.common.collect.Iterables.elementsEqual;
import static com.google.common.collect.Iterables.getLast;
import static com.google.common.collect.Iterables.transform;
import static java.util.Objects.requireNonNull;
import static java.util.stream.Collectors.toList;

class StatementAnalyzer
        extends DefaultTraversalVisitor<RelationType, AnalysisContext>
{
    private final Analysis analysis;
    private final Metadata metadata;
    private final Session session;
    private final Optional<QueryExplainer> queryExplainer;
    private final boolean experimentalSyntaxEnabled;
    private final SqlParser sqlParser;
    private final AccessControl accessControl;

    public StatementAnalyzer(
            Analysis analysis,
            Metadata metadata,
            SqlParser sqlParser,
            AccessControl accessControl, Session session,
            boolean experimentalSyntaxEnabled,
            Optional<QueryExplainer> queryExplainer)
    {
        this.analysis = requireNonNull(analysis, "analysis is null");
        this.metadata = requireNonNull(metadata, "metadata is null");
        this.sqlParser = requireNonNull(sqlParser, "sqlParser is null");
        this.accessControl = requireNonNull(accessControl, "accessControl is null");
        this.session = requireNonNull(session, "session is null");
        this.experimentalSyntaxEnabled = experimentalSyntaxEnabled;
        this.queryExplainer = requireNonNull(queryExplainer, "queryExplainer is null");
    }

    @Override
    protected RelationType visitShowTables(ShowTables showTables, AnalysisContext context)
    {
        String catalogName = session.getCatalog().orElse(null);
        String schemaName = session.getSchema().orElse(null);

        Optional<QualifiedName> schema = showTables.getSchema();
        if (schema.isPresent()) {
            List<String> parts = schema.get().getParts();
            if (parts.size() > 2) {
                throw new SemanticException(INVALID_SCHEMA_NAME, showTables, "Too many parts in schema name: %s", schema);
            }
            if (parts.size() == 2) {
                catalogName = parts.get(0);
            }
            schemaName = schema.get().getSuffix();
        }

        if (catalogName == null) {
            throw new SemanticException(CATALOG_NOT_SPECIFIED, showTables, "Catalog must be specified when session catalog is not set");
        }
        if (schemaName == null) {
            throw new SemanticException(SCHEMA_NOT_SPECIFIED, showTables, "Schema must be specified when session schema is not set");
        }

        if (!metadata.listSchemaNames(session, catalogName).contains(schemaName)) {
            throw new SemanticException(MISSING_SCHEMA, showTables, "Schema '%s' does not exist", schemaName);
        }

        Expression predicate = equal(nameReference("table_schema"), new StringLiteral(schemaName));

        Optional<String> likePattern = showTables.getLikePattern();
        if (likePattern.isPresent()) {
            Expression likePredicate = new LikePredicate(nameReference("table_name"), new StringLiteral(likePattern.get()), null);
            predicate = logicalAnd(predicate, likePredicate);
        }

        Query query = simpleQuery(
                selectList(aliasedName("table_name", "Table")),
                from(catalogName, TABLE_TABLES),
                predicate,
                ordering(ascending("table_name")));

        return process(query, context);
    }

    @Override
    protected RelationType visitShowSchemas(ShowSchemas node, AnalysisContext context)
    {
        if (!node.getCatalog().isPresent() && !session.getCatalog().isPresent()) {
            throw new SemanticException(CATALOG_NOT_SPECIFIED, node, "Catalog must be specified when session catalog is not set");
        }

        Query query = simpleQuery(
                selectList(aliasedName("schema_name", "Schema")),
                from(node.getCatalog().orElseGet(() -> session.getCatalog().get()), TABLE_SCHEMATA),
                ordering(ascending("schema_name")));

        return process(query, context);
    }

    @Override
    protected RelationType visitShowCatalogs(ShowCatalogs node, AnalysisContext context)
    {
        List<Expression> rows = metadata.getCatalogNames().keySet().stream()
                .map(name -> row(new StringLiteral(name)))
                .collect(toList());

        Query query = simpleQuery(
                selectList(new AllColumns()),
                aliased(new Values(rows), "catalogs", ImmutableList.of("Catalog")));

        return process(query, context);
    }

    @Override
    protected RelationType visitShowColumns(ShowColumns showColumns, AnalysisContext context)
    {
        QualifiedObjectName tableName = createQualifiedObjectName(session, showColumns, showColumns.getTable());

        if (!metadata.getView(session, tableName).isPresent() &&
                !metadata.getTableHandle(session, tableName).isPresent()) {
            throw new SemanticException(MISSING_TABLE, showColumns, "Table '%s' does not exist", tableName);
        }

        Query query = simpleQuery(
                selectList(
                        aliasedName("column_name", "Column"),
                        aliasedName("data_type", "Type"),
                        aliasedYesNoToBoolean("is_nullable", "Null"),
                        aliasedYesNoToBoolean("is_partition_key", "Partition Key"),
                        aliasedNullToEmpty("comment", "Comment")),
                from(tableName.getCatalogName(), TABLE_COLUMNS),
                logicalAnd(
                        equal(nameReference("table_schema"), new StringLiteral(tableName.getSchemaName())),
                        equal(nameReference("table_name"), new StringLiteral(tableName.getObjectName()))),
                ordering(ascending("ordinal_position")));

        return process(query, context);
    }

    @Override
    protected RelationType visitUse(Use node, AnalysisContext context)
    {
        analysis.setUpdateType("USE");
        throw new SemanticException(NOT_SUPPORTED, node, "USE statement is not supported");
    }

    @Override
    protected RelationType visitShowPartitions(ShowPartitions showPartitions, AnalysisContext context)
    {
        QualifiedObjectName table = createQualifiedObjectName(session, showPartitions, showPartitions.getTable());
        Optional<TableHandle> tableHandle = metadata.getTableHandle(session, table);
        if (!tableHandle.isPresent()) {
            throw new SemanticException(MISSING_TABLE, showPartitions, "Table '%s' does not exist", table);
        }

            /*
                Generate a dynamic pivot to output one column per partition key.
                For example, a table with two partition keys (ds, cluster_name)
                would generate the following query:

                SELECT
                  partition_number
                , max(CASE WHEN partition_key = 'ds' THEN partition_value END) ds
                , max(CASE WHEN partition_key = 'cluster_name' THEN partition_value END) cluster_name
                FROM ...
                GROUP BY partition_number

                The values are also cast to the type of the partition column.
                The query is then wrapped to allow custom filtering and ordering.
            */

        ImmutableList.Builder<SelectItem> selectList = ImmutableList.builder();
        ImmutableList.Builder<SelectItem> wrappedList = ImmutableList.builder();
        selectList.add(unaliasedName("partition_number"));
        for (ColumnMetadata column : metadata.getTableMetadata(session, tableHandle.get()).getColumns()) {
            if (!column.isPartitionKey()) {
                continue;
            }
            Expression key = equal(nameReference("partition_key"), new StringLiteral(column.getName()));
            Expression value = caseWhen(key, nameReference("partition_value"));
            value = new Cast(value, column.getType().getTypeSignature().toString());
            Expression function = functionCall("max", value);
            selectList.add(new SingleColumn(function, column.getName()));
            wrappedList.add(unaliasedName(column.getName()));
        }

        Query query = simpleQuery(
                selectAll(selectList.build()),
                from(table.getCatalogName(), TABLE_INTERNAL_PARTITIONS),
                Optional.of(logicalAnd(
                        equal(nameReference("table_schema"), new StringLiteral(table.getSchemaName())),
                        equal(nameReference("table_name"), new StringLiteral(table.getObjectName())))),
                ImmutableList.of(new SimpleGroupBy(ImmutableList.of(nameReference("partition_number")))),
                Optional.empty(),
                ImmutableList.of(),
                Optional.empty());

        query = simpleQuery(
                selectAll(wrappedList.build()),
                subquery(query),
                showPartitions.getWhere(),
                ImmutableList.of(),
                Optional.empty(),
                ImmutableList.<SortItem>builder()
                        .addAll(showPartitions.getOrderBy())
                        .add(ascending("partition_number"))
                        .build(),
                showPartitions.getLimit());

        return process(query, context);
    }

    @Override
    protected RelationType visitShowFunctions(ShowFunctions node, AnalysisContext context)
    {
        ImmutableList.Builder<Expression> rows = ImmutableList.builder();
        for (SqlFunction function : metadata.listFunctions()) {
            if (function.getSignature().getKind() == APPROXIMATE_AGGREGATE) {
                continue;
            }
            rows.add(row(
                    new StringLiteral(function.getSignature().getName()),
                    new StringLiteral(function.getSignature().getReturnType().toString()),
                    new StringLiteral(Joiner.on(", ").join(function.getSignature().getArgumentTypes())),
                    new StringLiteral(getFunctionType(function)),
                    function.isDeterministic() ? TRUE_LITERAL : FALSE_LITERAL,
                    new StringLiteral(nullToEmpty(function.getDescription()))));
        }

        Map<String, String> columns = ImmutableMap.<String, String>builder()
                .put("function_name", "Function")
                .put("return_type", "Return Type")
                .put("argument_types", "Argument Types")
                .put("function_type", "Function Type")
                .put("deterministic", "Deterministic")
                .put("description", "Description")
                .build();

        Query query = simpleQuery(
                selectAll(columns.entrySet().stream()
                        .map(entry -> aliasedName(entry.getKey(), entry.getValue()))
                        .collect(toImmutableList())),
                aliased(new Values(rows.build()), "functions", ImmutableList.copyOf(columns.keySet())),
                ordering(
                        ascending("function_name"),
                        ascending("return_type"),
                        ascending("argument_types"),
                        ascending("function_type")));

        return process(query, context);
    }

    private static String getFunctionType(SqlFunction function)
    {
        FunctionKind kind = function.getSignature().getKind();
        switch (kind) {
            case AGGREGATE:
            case APPROXIMATE_AGGREGATE:
                return "aggregate";
            case WINDOW:
                return "window";
            case SCALAR:
                return "scalar";
        }
        throw new IllegalArgumentException("Unsupported function kind: " + kind);
    }

    @Override
    protected RelationType visitShowSession(ShowSession node, AnalysisContext context)
    {
        ImmutableList.Builder<Expression> rows = ImmutableList.builder();
        List<SessionPropertyValue> sessionProperties = metadata.getSessionPropertyManager().getAllSessionProperties(session);
        for (SessionPropertyValue sessionProperty : sessionProperties) {
            if (sessionProperty.isHidden()) {
                continue;
            }

            String value = sessionProperty.getValue();
            String defaultValue = sessionProperty.getDefaultValue();
            rows.add(row(
                    new StringLiteral(sessionProperty.getFullyQualifiedName()),
                    new StringLiteral(nullToEmpty(value)),
                    new StringLiteral(nullToEmpty(defaultValue)),
                    new StringLiteral(sessionProperty.getType()),
                    new StringLiteral(sessionProperty.getDescription()),
                    TRUE_LITERAL));
        }

        // add bogus row so we can support empty sessions
        StringLiteral empty = new StringLiteral("");
        rows.add(row(empty, empty, empty, empty, empty, FALSE_LITERAL));

        Query query = simpleQuery(
                selectList(
                        aliasedName("name", "Name"),
                        aliasedName("value", "Value"),
                        aliasedName("default", "Default"),
                        aliasedName("type", "Type"),
                        aliasedName("description", "Description")),
                aliased(
                        new Values(rows.build()),
                        "session",
                        ImmutableList.of("name", "value", "default", "type", "description", "include")),
                nameReference("include"));

        return process(query, context);
    }

    @Override
    protected RelationType visitInsert(Insert insert, AnalysisContext context)
    {
        QualifiedObjectName targetTable = createQualifiedObjectName(session, insert, insert.getTarget());
        if (metadata.getView(session, targetTable).isPresent()) {
            throw new SemanticException(NOT_SUPPORTED, insert, "Inserting into views is not supported");
        }

        analysis.setUpdateType("INSERT");

        // analyze the query that creates the data
        RelationType queryDescriptor = process(insert.getQuery(), context);

        // verify the insert destination columns match the query
        Optional<TableHandle> targetTableHandle = metadata.getTableHandle(session, targetTable);
        if (!targetTableHandle.isPresent()) {
            throw new SemanticException(MISSING_TABLE, insert, "Table '%s' does not exist", targetTable);
        }
        accessControl.checkCanInsertIntoTable(session.getRequiredTransactionId(), session.getIdentity(), targetTable);

        TableMetadata tableMetadata = metadata.getTableMetadata(session, targetTableHandle.get());
        List<String> tableColumns = tableMetadata.getVisibleColumnNames();

        final List<String> insertColumns;
        if (insert.getColumns().isPresent()) {
            insertColumns = insert.getColumns().get().stream()
                    .map(String::toLowerCase)
                    .collect(toImmutableList());

            Set<String> columnNames = new HashSet<>();
            for (String insertColumn : insertColumns) {
                if (!tableColumns.contains(insertColumn)) {
                    throw new SemanticException(MISSING_COLUMN, insert, "Insert column name does not exist in target table: %s", insertColumn);
                }
                if (!columnNames.add(insertColumn)) {
                    throw new SemanticException(DUPLICATE_COLUMN_NAME, insert, "Insert column name is specified more than once: %s", insertColumn);
                }
            }
        }
        else {
            insertColumns = tableColumns;
        }

        Map<String, ColumnHandle> columnHandles = metadata.getColumnHandles(session, targetTableHandle.get());
        analysis.setInsert(new Analysis.Insert(
                targetTableHandle.get(),
                insertColumns.stream().map(column -> columnHandles.get(column)).collect(toImmutableList())));

        Iterable<Type> tableTypes = insertColumns.stream()
                .map(insertColumn -> tableMetadata.getColumn(insertColumn).getType())
                .collect(toImmutableList());

        Iterable<Type> queryTypes = transform(queryDescriptor.getVisibleFields(), Field::getType);

        if (!elementsEqual(tableTypes, queryTypes)) {
            throw new SemanticException(MISMATCHED_SET_COLUMN_TYPES, insert, "Insert query has mismatched column types: " +
                    "Table: (" + Joiner.on(", ").join(tableTypes) + "), " +
                    "Query: (" + Joiner.on(", ").join(queryTypes) + ")");
        }

        return new RelationType(Field.newUnqualified("rows", BIGINT));
    }

    @Override
    protected RelationType visitDelete(Delete node, AnalysisContext context)
    {
        Table table = node.getTable();
        QualifiedObjectName tableName = createQualifiedObjectName(session, table, table.getName());
        if (metadata.getView(session, tableName).isPresent()) {
            throw new SemanticException(NOT_SUPPORTED, node, "Deleting from views is not supported");
        }

        analysis.setUpdateType("DELETE");

        analysis.setDelete(node);

        // Analyzer checks for select permissions but DELETE has a separate permission, so disable access checks
        // TODO: we shouldn't need to create a new analyzer. The access control should be carried in the context object
        StatementAnalyzer analyzer = new StatementAnalyzer(
                analysis,
                metadata,
                sqlParser,
                new AllowAllAccessControl(),
                session,
                experimentalSyntaxEnabled,
                queryExplainer);

        RelationType descriptor = analyzer.process(table, context);
        node.getWhere().ifPresent(where -> analyzer.analyzeWhere(node, descriptor, context, where));

        accessControl.checkCanDeleteFromTable(session.getRequiredTransactionId(), session.getIdentity(), tableName);

        return new RelationType(Field.newUnqualified("rows", BIGINT));
    }

    @Override
    protected RelationType visitCreateTableAsSelect(CreateTableAsSelect node, AnalysisContext context)
    {
        analysis.setUpdateType("CREATE TABLE");

        // turn this into a query that has a new table writer node on top.
        QualifiedObjectName targetTable = createQualifiedObjectName(session, node, node.getName());
        analysis.setCreateTableDestination(targetTable);

        for (Expression expression : node.getProperties().values()) {
            // analyze table property value expressions which must be constant
            createConstantAnalyzer(metadata, session)
                    .analyze(expression, new RelationType(), context);
        }
        analysis.setCreateTableProperties(node.getProperties());

        Optional<TableHandle> targetTableHandle = metadata.getTableHandle(session, targetTable);
        if (targetTableHandle.isPresent()) {
            throw new SemanticException(TABLE_ALREADY_EXISTS, node, "Destination table '%s' already exists", targetTable);
        }
        accessControl.checkCanCreateTable(session.getRequiredTransactionId(), session.getIdentity(), targetTable);

        analysis.setCreateTableAsSelectWithData(node.isWithData());

        // analyze the query that creates the table
        RelationType descriptor = process(node.getQuery(), context);

        validateColumnNames(node, descriptor);

        return new RelationType(Field.newUnqualified("rows", BIGINT));
    }

    @Override
    protected RelationType visitCreateView(CreateView node, AnalysisContext context)
    {
        analysis.setUpdateType("CREATE VIEW");

        // analyze the query that creates the view
        StatementAnalyzer analyzer = new StatementAnalyzer(
                analysis,
                metadata,
                sqlParser,
                new ViewAccessControl(accessControl),
                session,
                experimentalSyntaxEnabled,
                queryExplainer);
        RelationType descriptor = analyzer.process(node.getQuery(), new AnalysisContext());

        QualifiedObjectName viewName = createQualifiedObjectName(session, node, node.getName());
        accessControl.checkCanCreateView(session.getRequiredTransactionId(), session.getIdentity(), viewName);

        validateColumnNames(node, descriptor);

        return descriptor;
    }

    private static void validateColumnNames(Statement node, RelationType descriptor)
    {
        // verify that all column names are specified and unique
        // TODO: collect errors and return them all at once
        Set<String> names = new HashSet<>();
        for (Field field : descriptor.getVisibleFields()) {
            Optional<String> fieldName = field.getName();
            if (!fieldName.isPresent()) {
                throw new SemanticException(COLUMN_NAME_NOT_SPECIFIED, node, "Column name not specified at position %s", descriptor.indexOf(field) + 1);
            }
            if (!names.add(fieldName.get())) {
                throw new SemanticException(DUPLICATE_COLUMN_NAME, node, "Column name '%s' specified more than once", fieldName.get());
            }
        }
    }

    @Override
    protected RelationType visitExplain(Explain node, AnalysisContext context)
            throws SemanticException
    {
        checkState(queryExplainer.isPresent(), "query explainer not available");
        ExplainType.Type planType = LOGICAL;
        ExplainFormat.Type planFormat = TEXT;
        List<ExplainOption> options = node.getOptions();

        for (ExplainOption option : options) {
            if (option instanceof ExplainType) {
                planType = ((ExplainType) option).getType();
                break;
            }
        }

        for (ExplainOption option : options) {
            if (option instanceof ExplainFormat) {
                planFormat = ((ExplainFormat) option).getType();
                break;
            }
        }

        String queryPlan = getQueryPlan(node, planType, planFormat);

        Query query = simpleQuery(
                selectList(new AllColumns()),
                aliased(
                        values(row(new StringLiteral((queryPlan)))),
                        "plan",
                        ImmutableList.of("Query Plan")));

        return process(query, context);
    }

    private String getQueryPlan(Explain node, ExplainType.Type planType, ExplainFormat.Type planFormat)
            throws IllegalArgumentException
    {
        switch (planFormat) {
            case GRAPHVIZ:
                return queryExplainer.get().getGraphvizPlan(session, node.getStatement(), planType);
            case TEXT:
                return queryExplainer.get().getPlan(session, node.getStatement(), planType);
        }
        throw new IllegalArgumentException("Invalid Explain Format: " + planFormat.toString());
    }

    @Override
    protected RelationType visitQuery(Query node, AnalysisContext parentContext)
    {
        AnalysisContext context = new AnalysisContext(parentContext);

        if (node.getApproximate().isPresent()) {
            if (!experimentalSyntaxEnabled) {
                throw new SemanticException(NOT_SUPPORTED, node, "approximate queries are not enabled");
            }
            context.setApproximate(true);
        }

        analyzeWith(node, context);

        RelationType descriptor = process(node.getQueryBody(), context);
        analyzeOrderBy(node, descriptor, context);

        // Input fields == Output fields
        analysis.setOutputDescriptor(node, descriptor);
        analysis.setOutputExpressions(node, descriptorToFields(descriptor));
        analysis.setQuery(node);

        return descriptor;
    }

    @Override
    protected RelationType visitUnnest(Unnest node, AnalysisContext context)
    {
        ImmutableList.Builder<Field> outputFields = ImmutableList.builder();
        for (Expression expression : node.getExpressions()) {
            ExpressionAnalysis expressionAnalysis = analyzeExpression(expression, context.getLateralTupleDescriptor(), context);
            Type expressionType = expressionAnalysis.getType(expression);
            if (expressionType instanceof ArrayType) {
                outputFields.add(Field.newUnqualified(Optional.empty(), ((ArrayType) expressionType).getElementType()));
            }
            else if (expressionType instanceof MapType) {
                outputFields.add(Field.newUnqualified(Optional.empty(), ((MapType) expressionType).getKeyType()));
                outputFields.add(Field.newUnqualified(Optional.empty(), ((MapType) expressionType).getValueType()));
            }
            else {
                throw new PrestoException(INVALID_FUNCTION_ARGUMENT, "Cannot unnest type: " + expressionType);
            }
        }
        if (node.isWithOrdinality()) {
            outputFields.add(Field.newUnqualified(Optional.empty(), BigintType.BIGINT));
        }
        RelationType descriptor = new RelationType(outputFields.build());
        analysis.setOutputDescriptor(node, descriptor);
        return descriptor;
    }

    @Override
    protected RelationType visitTable(Table table, AnalysisContext context)
    {
        if (!table.getName().getPrefix().isPresent()) {
            // is this a reference to a WITH query?
            String name = table.getName().getSuffix();

            Query query = context.getNamedQuery(name);
            if (query != null) {
                analysis.registerNamedQuery(table, query);

                // re-alias the fields with the name assigned to the query in the WITH declaration
                RelationType queryDescriptor = analysis.getOutputDescriptor(query);
                ImmutableList.Builder<Field> fields = ImmutableList.builder();
                for (Field field : queryDescriptor.getAllFields()) {
                    fields.add(Field.newQualified(QualifiedName.of(name), field.getName(), field.getType(), false));
                }

                RelationType descriptor = new RelationType(fields.build());
                analysis.setOutputDescriptor(table, descriptor);
                return descriptor;
            }
        }

        QualifiedObjectName name = MetadataUtil.createQualifiedObjectName(session, table, table.getName());

        Optional<ViewDefinition> optionalView = metadata.getView(session, name);
        if (optionalView.isPresent()) {
            ViewDefinition view = optionalView.get();

            Query query = parseView(view.getOriginalSql(), name, table);

            analysis.registerNamedQuery(table, query);

            accessControl.checkCanSelectFromView(session.getRequiredTransactionId(), session.getIdentity(), name);
            RelationType descriptor = analyzeView(query, name, view.getCatalog(), view.getSchema(), view.getOwner(), table);

            if (isViewStale(view.getColumns(), descriptor.getVisibleFields())) {
                throw new SemanticException(VIEW_IS_STALE, table, "View '%s' is stale; it must be re-created", name);
            }

            analysis.setOutputDescriptor(table, descriptor);
            return descriptor;
        }

        Optional<TableHandle> tableHandle = metadata.getTableHandle(session, name);
        if (!tableHandle.isPresent()) {
            if (!metadata.getCatalogNames().containsKey(name.getCatalogName())) {
                throw new SemanticException(MISSING_CATALOG, table, "Catalog %s does not exist", name.getCatalogName());
            }
            if (!metadata.listSchemaNames(session, name.getCatalogName()).contains(name.getSchemaName())) {
                throw new SemanticException(MISSING_SCHEMA, table, "Schema %s does not exist", name.getSchemaName());
            }

            if (table.getName().getSuffix().equalsIgnoreCase("DUAL")) {
                // TODO: remove this in a few releases
                throw new SemanticException(MISSING_TABLE, table, "DUAL table is no longer supported. Please use VALUES or FROM-less queries instead");
            }

            throw new SemanticException(MISSING_TABLE, table, "Table %s does not exist", name);
        }
        accessControl.checkCanSelectFromTable(session.getRequiredTransactionId(), session.getIdentity(), name);
        TableMetadata tableMetadata = metadata.getTableMetadata(session, tableHandle.get());
        Map<String, ColumnHandle> columnHandles = metadata.getColumnHandles(session, tableHandle.get());

        // TODO: discover columns lazily based on where they are needed (to support datasources that can't enumerate all tables)
        ImmutableList.Builder<Field> fields = ImmutableList.builder();
        for (ColumnMetadata column : tableMetadata.getColumns()) {
            Field field = Field.newQualified(table.getName(), Optional.of(column.getName()), column.getType(), column.isHidden());
            fields.add(field);
            ColumnHandle columnHandle = columnHandles.get(column.getName());
            checkArgument(columnHandle != null, "Unknown field %s", field);
            analysis.setColumn(field, columnHandle);
        }

        analysis.registerTable(table, tableHandle.get());

        RelationType descriptor = new RelationType(fields.build());
        analysis.setOutputDescriptor(table, descriptor);
        return descriptor;
    }

    @Override
    protected RelationType visitAliasedRelation(AliasedRelation relation, AnalysisContext context)
    {
        RelationType child = process(relation.getRelation(), context);

        // todo this check should be inside of TupleDescriptor.withAlias, but the exception needs the node object
        if (relation.getColumnNames() != null) {
            int totalColumns = child.getVisibleFieldCount();
            if (totalColumns != relation.getColumnNames().size()) {
                throw new SemanticException(MISMATCHED_COLUMN_ALIASES, relation, "Column alias list has %s entries but '%s' has %s columns available", relation.getColumnNames().size(), relation.getAlias(), totalColumns);
            }
        }

        RelationType descriptor = child.withAlias(relation.getAlias(), relation.getColumnNames());

        analysis.setOutputDescriptor(relation, descriptor);
        return descriptor;
    }

    @Override
    protected RelationType visitSampledRelation(final SampledRelation relation, AnalysisContext context)
    {
        if (relation.getColumnsToStratifyOn().isPresent()) {
            throw new SemanticException(NOT_SUPPORTED, relation, "STRATIFY ON is not yet implemented");
        }

        if (!DependencyExtractor.extractNames(relation.getSamplePercentage(), analysis.getColumnReferences()).isEmpty()) {
            throw new SemanticException(NON_NUMERIC_SAMPLE_PERCENTAGE, relation.getSamplePercentage(), "Sample percentage cannot contain column references");
        }

        IdentityHashMap<Expression, Type> expressionTypes = getExpressionTypes(session, metadata, sqlParser, ImmutableMap.<Symbol, Type>of(), relation.getSamplePercentage());
        ExpressionInterpreter samplePercentageEval = expressionOptimizer(relation.getSamplePercentage(), metadata, session, expressionTypes);

        Object samplePercentageObject = samplePercentageEval.optimize(symbol -> {
            throw new SemanticException(NON_NUMERIC_SAMPLE_PERCENTAGE, relation.getSamplePercentage(), "Sample percentage cannot contain column references");
        });

        if (!(samplePercentageObject instanceof Number)) {
            throw new SemanticException(NON_NUMERIC_SAMPLE_PERCENTAGE, relation.getSamplePercentage(), "Sample percentage should evaluate to a numeric expression");
        }

        double samplePercentageValue = ((Number) samplePercentageObject).doubleValue();

        if (samplePercentageValue < 0.0) {
            throw new SemanticException(SemanticErrorCode.SAMPLE_PERCENTAGE_OUT_OF_RANGE, relation.getSamplePercentage(), "Sample percentage must be greater than or equal to 0");
        }
        else if ((samplePercentageValue > 100.0) && (relation.getType() != SampledRelation.Type.POISSONIZED || relation.isRescaled())) {
            throw new SemanticException(SemanticErrorCode.SAMPLE_PERCENTAGE_OUT_OF_RANGE, relation.getSamplePercentage(), "Sample percentage must be less than or equal to 100");
        }

        if (relation.isRescaled() && !experimentalSyntaxEnabled) {
            throw new SemanticException(NOT_SUPPORTED, relation, "Rescaling is not enabled");
        }

        RelationType descriptor = process(relation.getRelation(), context);

        analysis.setOutputDescriptor(relation, descriptor);
        analysis.setSampleRatio(relation, samplePercentageValue / 100);

        return descriptor;
    }

    @Override
    protected RelationType visitTableSubquery(TableSubquery node, AnalysisContext context)
    {
        StatementAnalyzer analyzer = new StatementAnalyzer(analysis, metadata, sqlParser, accessControl, session, experimentalSyntaxEnabled, Optional.empty());
        RelationType descriptor = analyzer.process(node.getQuery(), context);

        analysis.setOutputDescriptor(node, descriptor);

        return descriptor;
    }

    @Override
    protected RelationType visitQuerySpecification(QuerySpecification node, AnalysisContext parentContext)
    {
        // TODO: extract candidate names from SELECT, WHERE, HAVING, GROUP BY and ORDER BY expressions
        // to pass down to analyzeFrom

        AnalysisContext context = new AnalysisContext(parentContext);

        RelationType tupleDescriptor = analyzeFrom(node, context);

        node.getWhere().ifPresent(where -> analyzeWhere(node, tupleDescriptor, context, where));

        List<FieldOrExpression> outputExpressions = analyzeSelect(node, tupleDescriptor, context);
        List<List<FieldOrExpression>> groupByExpressions = analyzeGroupBy(node, tupleDescriptor, context, outputExpressions);
        List<FieldOrExpression> orderByExpressions = analyzeOrderBy(node, tupleDescriptor, context, outputExpressions);
        analyzeHaving(node, tupleDescriptor, context);

        analyzeAggregations(node, tupleDescriptor, groupByExpressions, outputExpressions, orderByExpressions, context, analysis.getColumnReferences());
        analyzeWindowFunctions(node, outputExpressions, orderByExpressions);

        RelationType descriptor = computeOutputDescriptor(node, tupleDescriptor);
        analysis.setOutputDescriptor(node, descriptor);

        return descriptor;
    }

    @Override
    protected RelationType visitUnion(Union node, AnalysisContext context)
    {
        checkState(node.getRelations().size() >= 2);

        RelationType[] descriptors = node.getRelations().stream()
                .map(relation -> process(relation, context).withOnlyVisibleFields())
                .toArray(RelationType[]::new);
        Type[] outputFieldTypes = descriptors[0].getVisibleFields().stream()
                .map(Field::getType)
                .toArray(Type[]::new);
        for (RelationType descriptor : descriptors) {
            int outputFieldSize = outputFieldTypes.length;
            int descFieldSize = descriptor.getVisibleFields().size();
            if (outputFieldSize != descFieldSize) {
                throw new SemanticException(MISMATCHED_SET_COLUMN_TYPES,
                        node,
                        "union query has different number of fields: %d, %d",
                        outputFieldSize, descFieldSize);
            }
            for (int i = 0; i < descriptor.getVisibleFields().size(); i++) {
                Type descFieldType = descriptor.getFieldByIndex(i).getType();
                Optional<Type> commonSuperType = metadata.getTypeManager().getCommonSuperType(outputFieldTypes[i], descFieldType);
                if (!commonSuperType.isPresent()) {
                    throw new SemanticException(TYPE_MISMATCH,
                            node,
                            "column %d in union query has incompatible types: %s, %s",
                            i, outputFieldTypes[i].getDisplayName(), descFieldType.getDisplayName());
                }
                outputFieldTypes[i] = commonSuperType.get();
            }
        }

        Field[] outputDescriptorFields = new Field[outputFieldTypes.length];
        RelationType firstDescriptor = descriptors[0].withOnlyVisibleFields();
        for (int i = 0; i < outputFieldTypes.length; i++) {
            Field oldField = firstDescriptor.getFieldByIndex(i);
            outputDescriptorFields[i] = new Field(oldField.getRelationAlias(), oldField.getName(), outputFieldTypes[i], oldField.isHidden());
        }
        RelationType outputDescriptor = new RelationType(outputDescriptorFields);
        analysis.setOutputDescriptor(node, outputDescriptor);

        for (int i = 0; i < node.getRelations().size(); i++) {
            Relation relation = node.getRelations().get(i);
            RelationType descriptor = descriptors[i];
            for (int j = 0; j < descriptor.getVisibleFields().size(); j++) {
                Type outputFieldType = outputFieldTypes[j];
                Type descFieldType = descriptor.getFieldByIndex(j).getType();
                if (!outputFieldType.equals(descFieldType)) {
                    analysis.addRelationCoercion(relation, outputFieldTypes);
                    break;
                }
            }
        }

        return outputDescriptor;
    }

    @Override
    protected RelationType visitIntersect(Intersect node, AnalysisContext context)
    {
        throw new SemanticException(NOT_SUPPORTED, node, "INTERSECT not yet implemented");
    }

    @Override
    protected RelationType visitExcept(Except node, AnalysisContext context)
    {
        throw new SemanticException(NOT_SUPPORTED, node, "EXCEPT not yet implemented");
    }

    @Override
    protected RelationType visitJoin(Join node, AnalysisContext context)
    {
        JoinCriteria criteria = node.getCriteria().orElse(null);
        if (criteria instanceof NaturalJoin) {
            throw new SemanticException(NOT_SUPPORTED, node, "Natural join not supported");
        }

        AnalysisContext leftContext = new AnalysisContext(context);
        RelationType left = process(node.getLeft(), context);
        leftContext.setLateralTupleDescriptor(left);
        RelationType right = process(node.getRight(), leftContext);

        // todo this check should be inside of TupleDescriptor.join and then remove the public getRelationAlias method, but the exception needs the node object
        Sets.SetView<QualifiedName> duplicateAliases = Sets.intersection(left.getRelationAliases(), right.getRelationAliases());
        if (!duplicateAliases.isEmpty()) {
            throw new SemanticException(DUPLICATE_RELATION, node, "Relations appear more than once: %s", duplicateAliases);
        }

        RelationType output = left.joinWith(right);

        if (node.getType() == Join.Type.CROSS || node.getType() == Join.Type.IMPLICIT) {
            analysis.setOutputDescriptor(node, output);
            return output;
        }

        if (criteria instanceof JoinUsing) {
            // TODO: implement proper "using" semantics with respect to output columns
            List<String> columns = ((JoinUsing) criteria).getColumns();

            List<Expression> expressions = new ArrayList<>();
            for (String column : columns) {
                Expression leftExpression = new QualifiedNameReference(QualifiedName.of(column));
                Expression rightExpression = new QualifiedNameReference(QualifiedName.of(column));

                ExpressionAnalysis leftExpressionAnalysis = analyzeExpression(leftExpression, left, context);
                ExpressionAnalysis rightExpressionAnalysis = analyzeExpression(rightExpression, right, context);
                checkState(leftExpressionAnalysis.getSubqueryInPredicates().isEmpty(), "INVARIANT");
                checkState(rightExpressionAnalysis.getSubqueryInPredicates().isEmpty(), "INVARIANT");

                addCoercionForJoinCriteria(node, leftExpression, rightExpression);
                expressions.add(new ComparisonExpression(EQUAL, leftExpression, rightExpression));
            }

            analysis.setJoinCriteria(node, ExpressionUtils.and(expressions));
        }
        else if (criteria instanceof JoinOn) {
            Expression expression = ((JoinOn) criteria).getExpression();

            // ensure all names can be resolved, types match, etc (we don't need to record resolved names, subexpression types, etc. because
            // we do it further down when after we determine which subexpressions apply to left vs right tuple)
            ExpressionAnalyzer analyzer = ExpressionAnalyzer.create(analysis, session, metadata, sqlParser, accessControl, experimentalSyntaxEnabled);
            analyzer.analyze(expression, output, context);

            Analyzer.verifyNoAggregatesOrWindowFunctions(metadata, expression, "JOIN");

            // expressionInterpreter/optimizer only understands a subset of expression types
            // TODO: remove this when the new expression tree is implemented
            Expression canonicalized = CanonicalizeExpressions.canonicalizeExpression(expression);
            analyzer.analyze(canonicalized, output, context);

            Object optimizedExpression = expressionOptimizer(canonicalized, metadata, session, analyzer.getExpressionTypes()).optimize(NoOpSymbolResolver.INSTANCE);

            if (!(optimizedExpression instanceof Expression) && optimizedExpression instanceof Boolean) {
                // If the JoinOn clause evaluates to a boolean expression, simulate a cross join by adding the relevant redundant expression
                if (optimizedExpression.equals(Boolean.TRUE)) {
                    optimizedExpression = new ComparisonExpression(EQUAL, new LongLiteral("0"), new LongLiteral("0"));
                }
                else {
                    optimizedExpression = new ComparisonExpression(EQUAL, new LongLiteral("0"), new LongLiteral("1"));
                }
            }

            if (!(optimizedExpression instanceof Expression)) {
                throw new SemanticException(TYPE_MISMATCH, node, "Join clause must be a boolean expression");
            }
            // The optimization above may have rewritten the expression tree which breaks all the identity maps, so redo the analysis
            // to re-analyze coercions that might be necessary
            analyzer = ExpressionAnalyzer.create(analysis, session, metadata, sqlParser, accessControl, experimentalSyntaxEnabled);
            analyzer.analyze((Expression) optimizedExpression, output, context);
            analysis.addCoercions(analyzer.getExpressionCoercions());

            for (Expression conjunct : ExpressionUtils.extractConjuncts((Expression) optimizedExpression)) {
                conjunct = ExpressionUtils.normalize(conjunct);
                if (!(conjunct instanceof ComparisonExpression)) {
                    throw new SemanticException(NOT_SUPPORTED, node, "Non-equi joins not supported: %s", conjunct);
                }

                ComparisonExpression comparison = (ComparisonExpression) conjunct;
                Set<QualifiedName> firstDependencies = DependencyExtractor.extractNames(comparison.getLeft(), analyzer.getColumnReferences());
                Set<QualifiedName> secondDependencies = DependencyExtractor.extractNames(comparison.getRight(), analyzer.getColumnReferences());

                Expression leftExpression;
                Expression rightExpression;
                if (firstDependencies.stream().allMatch(left.canResolvePredicate()) && secondDependencies.stream().allMatch(right.canResolvePredicate())) {
                    leftExpression = comparison.getLeft();
                    rightExpression = comparison.getRight();
                }
                else if (firstDependencies.stream().allMatch(right.canResolvePredicate()) && secondDependencies.stream().allMatch(left.canResolvePredicate())) {
                    leftExpression = comparison.getRight();
                    rightExpression = comparison.getLeft();
                }
                else {
                    // must have a complex expression that involves both tuples on one side of the comparison expression (e.g., coalesce(left.x, right.x) = 1)
                    throw new SemanticException(NOT_SUPPORTED, node, "Non-equi joins not supported: %s", conjunct);
                }

                // analyze the clauses to record the types of all subexpressions and resolve names against the left/right underlying tuples
                ExpressionAnalysis leftExpressionAnalysis = analyzeExpression(leftExpression, left, context);
                ExpressionAnalysis rightExpressionAnalysis = analyzeExpression(rightExpression, right, context);
                addCoercionForJoinCriteria(node, leftExpression, rightExpression);
                analysis.addJoinInPredicates(node, new Analysis.JoinInPredicates(leftExpressionAnalysis.getSubqueryInPredicates(), rightExpressionAnalysis.getSubqueryInPredicates()));
            }

            analysis.setJoinCriteria(node, (Expression) optimizedExpression);
        }
        else {
            throw new UnsupportedOperationException("unsupported join criteria: " + criteria.getClass().getName());
        }

        analysis.setOutputDescriptor(node, output);
        return output;
    }

    private void addCoercionForJoinCriteria(Join node, Expression leftExpression, Expression rightExpression)
    {
        Type leftType = analysis.getType(leftExpression);
        Type rightType = analysis.getType(rightExpression);
        Optional<Type> superType = metadata.getTypeManager().getCommonSuperType(leftType, rightType);
        if (!superType.isPresent()) {
            throw new SemanticException(TYPE_MISMATCH, node, "Join criteria has incompatible types: %s, %s", leftType.getDisplayName(), rightType.getDisplayName());
        }
        if (!leftType.equals(superType.get())) {
            analysis.addCoercion(leftExpression, superType.get());
        }
        if (!rightType.equals(superType.get())) {
            analysis.addCoercion(rightExpression, superType.get());
        }
    }

    @Override
    protected RelationType visitValues(Values node, AnalysisContext context)
    {
        checkState(node.getRows().size() >= 1);

        // get unique row types
        Set<List<Type>> rowTypes = node.getRows().stream()
                .map(row -> analyzeExpression(row, new RelationType(), context).getType(row))
                .map(type -> {
                    if (type instanceof RowType) {
                        return type.getTypeParameters();
                    }
                    return ImmutableList.of(type);
                })
                .collect(toImmutableSet());

        // determine common super type of the rows
        List<Type> fieldTypes = new ArrayList<>(rowTypes.iterator().next());
        for (List<Type> rowType : rowTypes) {
            for (int i = 0; i < rowType.size(); i++) {
                Type fieldType = rowType.get(i);
                Type superType = fieldTypes.get(i);

                Optional<Type> commonSuperType = metadata.getTypeManager().getCommonSuperType(fieldType, superType);
                if (!commonSuperType.isPresent()) {
                    throw new SemanticException(MISMATCHED_SET_COLUMN_TYPES,
                            node,
                            "Values rows have mismatched types: %s vs %s",
                            Iterables.get(rowTypes, 0),
                            Iterables.get(rowTypes, 1));
                }
                fieldTypes.set(i, commonSuperType.get());
            }
        }

        // add coercions for the rows
        for (Expression row : node.getRows()) {
            if (row instanceof Row) {
                List<Expression> items = ((Row) row).getItems();
                for (int i = 0; i < items.size(); i++) {
                    Type expectedType = fieldTypes.get(i);
                    Expression item = items.get(i);
                    if (!analysis.getType(item).equals(expectedType)) {
                        analysis.addCoercion(item, expectedType);
                    }
                }
            }
            else {
                Type expectedType = fieldTypes.get(0);
                if (!analysis.getType(row).equals(expectedType)) {
                    analysis.addCoercion(row, expectedType);
                }
            }
        }

        RelationType descriptor = new RelationType(fieldTypes.stream()
                .map(valueType -> Field.newUnqualified(Optional.empty(), valueType))
                .collect(toImmutableList()));

        analysis.setOutputDescriptor(node, descriptor);
        return descriptor;
    }

    private void analyzeWindowFunctions(QuerySpecification node, List<FieldOrExpression> outputExpressions, List<FieldOrExpression> orderByExpressions)
    {
        WindowFunctionExtractor extractor = new WindowFunctionExtractor();

        for (FieldOrExpression fieldOrExpression : Iterables.concat(outputExpressions, orderByExpressions)) {
            if (fieldOrExpression.isExpression()) {
                extractor.process(fieldOrExpression.getExpression(), null);
                new WindowFunctionValidator().process(fieldOrExpression.getExpression(), analysis);
            }
        }

        List<FunctionCall> windowFunctions = extractor.getWindowFunctions();

        for (FunctionCall windowFunction : windowFunctions) {
            Window window = windowFunction.getWindow().get();

            WindowFunctionExtractor nestedExtractor = new WindowFunctionExtractor();
            for (Expression argument : windowFunction.getArguments()) {
                nestedExtractor.process(argument, null);
            }

            for (Expression expression : window.getPartitionBy()) {
                nestedExtractor.process(expression, null);
            }

            for (SortItem sortItem : window.getOrderBy()) {
                nestedExtractor.process(sortItem.getSortKey(), null);
            }

            if (window.getFrame().isPresent()) {
                nestedExtractor.process(window.getFrame().get(), null);
            }

            if (!nestedExtractor.getWindowFunctions().isEmpty()) {
                throw new SemanticException(NESTED_WINDOW, node, "Cannot nest window functions inside window function '%s': %s",
                        windowFunction,
                        extractor.getWindowFunctions());
            }

            if (windowFunction.isDistinct()) {
                throw new SemanticException(NOT_SUPPORTED, node, "DISTINCT in window function parameters not yet supported: %s", windowFunction);
            }

            if (window.getFrame().isPresent()) {
                analyzeWindowFrame(window.getFrame().get());
            }

            List<TypeSignature> argumentTypes = Lists.transform(windowFunction.getArguments(), expression -> analysis.getType(expression).getTypeSignature());

            FunctionKind kind = metadata.getFunctionRegistry().resolveFunction(windowFunction.getName(), argumentTypes, false).getKind();
            if (kind != AGGREGATE && kind != APPROXIMATE_AGGREGATE && kind != WINDOW) {
                throw new SemanticException(MUST_BE_WINDOW_FUNCTION, node, "Not a window function: %s", windowFunction.getName());
            }
        }

        analysis.setWindowFunctions(node, windowFunctions);
    }

    private static void analyzeWindowFrame(WindowFrame frame)
    {
        FrameBound.Type startType = frame.getStart().getType();
        FrameBound.Type endType = frame.getEnd().orElse(new FrameBound(CURRENT_ROW)).getType();

        if (startType == UNBOUNDED_FOLLOWING) {
            throw new SemanticException(INVALID_WINDOW_FRAME, frame, "Window frame start cannot be UNBOUNDED FOLLOWING");
        }
        if (endType == UNBOUNDED_PRECEDING) {
            throw new SemanticException(INVALID_WINDOW_FRAME, frame, "Window frame end cannot be UNBOUNDED PRECEDING");
        }
        if ((startType == CURRENT_ROW) && (endType == PRECEDING)) {
            throw new SemanticException(INVALID_WINDOW_FRAME, frame, "Window frame starting from CURRENT ROW cannot end with PRECEDING");
        }
        if ((startType == FOLLOWING) && (endType == PRECEDING)) {
            throw new SemanticException(INVALID_WINDOW_FRAME, frame, "Window frame starting from FOLLOWING cannot end with PRECEDING");
        }
        if ((startType == FOLLOWING) && (endType == CURRENT_ROW)) {
            throw new SemanticException(INVALID_WINDOW_FRAME, frame, "Window frame starting from FOLLOWING cannot end with CURRENT ROW");
        }
        if ((frame.getType() == RANGE) && ((startType == PRECEDING) || (endType == PRECEDING))) {
            throw new SemanticException(INVALID_WINDOW_FRAME, frame, "Window frame RANGE PRECEDING is only supported with UNBOUNDED");
        }
        if ((frame.getType() == RANGE) && ((startType == FOLLOWING) || (endType == FOLLOWING))) {
            throw new SemanticException(INVALID_WINDOW_FRAME, frame, "Window frame RANGE FOLLOWING is only supported with UNBOUNDED");
        }
    }

    private void analyzeHaving(QuerySpecification node, RelationType tupleDescriptor, AnalysisContext context)
    {
        if (node.getHaving().isPresent()) {
            Expression predicate = node.getHaving().get();

            ExpressionAnalysis expressionAnalysis = analyzeExpression(predicate, tupleDescriptor, context);
            analysis.recordSubqueries(node, expressionAnalysis);

            Type predicateType = expressionAnalysis.getType(predicate);
            if (!predicateType.equals(BOOLEAN) && !predicateType.equals(UNKNOWN)) {
                throw new SemanticException(TYPE_MISMATCH, predicate, "HAVING clause must evaluate to a boolean: actual type %s", predicateType);
            }

            analysis.setHaving(node, predicate);
        }
    }

    private List<FieldOrExpression> analyzeOrderBy(QuerySpecification node, RelationType tupleDescriptor, AnalysisContext context, List<FieldOrExpression> outputExpressions)
    {
        List<SortItem> items = node.getOrderBy();

        ImmutableList.Builder<FieldOrExpression> orderByExpressionsBuilder = ImmutableList.builder();

        if (!items.isEmpty()) {
            // Compute aliased output terms so we can resolve order by expressions against them first
            ImmutableMultimap.Builder<QualifiedName, Expression> byAliasBuilder = ImmutableMultimap.builder();
            for (SelectItem item : node.getSelect().getSelectItems()) {
                if (item instanceof SingleColumn) {
                    Optional<String> alias = ((SingleColumn) item).getAlias();
                    if (alias.isPresent()) {
                        byAliasBuilder.put(QualifiedName.of(alias.get()), ((SingleColumn) item).getExpression()); // TODO: need to know if alias was quoted
                    }
                }
            }
            Multimap<QualifiedName, Expression> byAlias = byAliasBuilder.build();

            for (SortItem item : items) {
                Expression expression = item.getSortKey();

                FieldOrExpression orderByExpression = null;
                if (expression instanceof QualifiedNameReference && !((QualifiedNameReference) expression).getName().getPrefix().isPresent()) {
                    // if this is a simple name reference, try to resolve against output columns

                    QualifiedName name = ((QualifiedNameReference) expression).getName();
                    Collection<Expression> expressions = byAlias.get(name);
                    if (expressions.size() > 1) {
                        throw new SemanticException(AMBIGUOUS_ATTRIBUTE, expression, "'%s' in ORDER BY is ambiguous", name.getSuffix());
                    }
                    else if (expressions.size() == 1) {
                        orderByExpression = new FieldOrExpression(Iterables.getOnlyElement(expressions));
                    }

                    // otherwise, couldn't resolve name against output aliases, so fall through...
                }
                else if (expression instanceof LongLiteral) {
                    // this is an ordinal in the output tuple

                    long ordinal = ((LongLiteral) expression).getValue();
                    if (ordinal < 1 || ordinal > outputExpressions.size()) {
                        throw new SemanticException(INVALID_ORDINAL, expression, "ORDER BY position %s is not in select list", ordinal);
                    }

                    orderByExpression = outputExpressions.get((int) (ordinal - 1));

                    if (orderByExpression.isExpression()) {
                        Type type = analysis.getType(orderByExpression.getExpression());
                        if (!type.isOrderable()) {
                            throw new SemanticException(TYPE_MISMATCH, node, "The type of expression in position %s is not orderable (actual: %s), and therefore cannot be used in ORDER BY: %s", ordinal, type, orderByExpression);
                        }
                    }
                    else {
                        Type type = tupleDescriptor.getFieldByIndex(orderByExpression.getFieldIndex()).getType();
                        if (!type.isOrderable()) {
                            throw new SemanticException(TYPE_MISMATCH, node, "The type of expression in position %s is not orderable (actual: %s), and therefore cannot be used in ORDER BY", ordinal, type);
                        }
                    }
                }

                // otherwise, just use the expression as is
                if (orderByExpression == null) {
                    orderByExpression = new FieldOrExpression(expression);
                }

                if (orderByExpression.isExpression()) {
                    ExpressionAnalysis expressionAnalysis = analyzeExpression(orderByExpression.getExpression(), tupleDescriptor, context);
                    analysis.recordSubqueries(node, expressionAnalysis);

                    Type type = expressionAnalysis.getType(orderByExpression.getExpression());
                    if (!type.isOrderable()) {
                        throw new SemanticException(TYPE_MISMATCH, node, "Type %s is not orderable, and therefore cannot be used in ORDER BY: %s", type, expression);
                    }
                }

                orderByExpressionsBuilder.add(orderByExpression);
            }
        }

        List<FieldOrExpression> orderByExpressions = orderByExpressionsBuilder.build();
        analysis.setOrderByExpressions(node, orderByExpressions);

        if (node.getSelect().isDistinct() && !outputExpressions.containsAll(orderByExpressions)) {
            throw new SemanticException(ORDER_BY_MUST_BE_IN_SELECT, node.getSelect(), "For SELECT DISTINCT, ORDER BY expressions must appear in select list");
        }
        return orderByExpressions;
    }

    private List<List<FieldOrExpression>> analyzeGroupBy(QuerySpecification node, RelationType tupleDescriptor, AnalysisContext context, List<FieldOrExpression> outputExpressions)
    {
        List<Set<Set<Expression>>> enumeratedGroupingSets = node.getGroupBy().stream()
                .map(GroupingElement::enumerateGroupingSets)
                .distinct()
                .collect(toImmutableList());

        // compute cross product of enumerated grouping sets, if there are any
        List<List<Expression>> computedGroupingSets = ImmutableList.of();
        if (!enumeratedGroupingSets.isEmpty()) {
            computedGroupingSets = Sets.cartesianProduct(enumeratedGroupingSets).stream()
                    .map(groupingSetList -> groupingSetList.stream()
                            .flatMap(Collection::stream)
                            .distinct()
                            .collect(toImmutableList()))
                    .distinct()
                    .collect(toImmutableList());
        }

        // if there are aggregates, but no grouping columns, create a grand total grouping set
        if (computedGroupingSets.isEmpty() && !extractAggregates(node).isEmpty()) {
            computedGroupingSets = ImmutableList.of(ImmutableList.of());
        }

        if (computedGroupingSets.size() > 1) {
            throw new SemanticException(NOT_SUPPORTED, node, "Grouping by multiple sets of columns is not yet supported");
        }

        List<List<FieldOrExpression>> analyzedGroupingSets = computedGroupingSets.stream()
                .map(groupingSet -> analyzeGroupingColumns(groupingSet, node, tupleDescriptor, context, outputExpressions))
                .collect(toImmutableList());

        analysis.setGroupingSets(node, analyzedGroupingSets);
        return analyzedGroupingSets;
    }

    private List<FieldOrExpression> analyzeGroupingColumns(List<Expression> groupingColumns, QuerySpecification node, RelationType tupleDescriptor, AnalysisContext context, List<FieldOrExpression> outputExpressions)
    {
        ImmutableList.Builder<FieldOrExpression> groupingColumnsBuilder = ImmutableList.builder();
        for (Expression groupingColumn : groupingColumns) {
            // first, see if this is an ordinal
            FieldOrExpression groupByExpression;

            if (groupingColumn instanceof LongLiteral) {
                long ordinal = ((LongLiteral) groupingColumn).getValue();
                if (ordinal < 1 || ordinal > outputExpressions.size()) {
                    throw new SemanticException(INVALID_ORDINAL, groupingColumn, "GROUP BY position %s is not in select list", ordinal);
                }

                groupByExpression = outputExpressions.get((int) (ordinal - 1));
            }
            else {
                ExpressionAnalysis expressionAnalysis = analyzeExpression(groupingColumn, tupleDescriptor, context);
                analysis.recordSubqueries(node, expressionAnalysis);
                groupByExpression = new FieldOrExpression(groupingColumn);
            }

            Type type;
            if (groupByExpression.isExpression()) {
                Analyzer.verifyNoAggregatesOrWindowFunctions(metadata, groupByExpression.getExpression(), "GROUP BY");
                type = analysis.getType(groupByExpression.getExpression());
            }
            else {
                type = tupleDescriptor.getFieldByIndex(groupByExpression.getFieldIndex()).getType();
            }
            if (!type.isComparable()) {
                throw new SemanticException(TYPE_MISMATCH, node, "%s is not comparable, and therefore cannot be used in GROUP BY", type);
            }

            groupingColumnsBuilder.add(groupByExpression);
        }
        return groupingColumnsBuilder.build();
    }

    private RelationType computeOutputDescriptor(QuerySpecification node, RelationType inputTupleDescriptor)
    {
        ImmutableList.Builder<Field> outputFields = ImmutableList.builder();

        for (SelectItem item : node.getSelect().getSelectItems()) {
            if (item instanceof AllColumns) {
                // expand * and T.*
                Optional<QualifiedName> starPrefix = ((AllColumns) item).getPrefix();

                for (Field field : inputTupleDescriptor.resolveFieldsWithPrefix(starPrefix)) {
                    outputFields.add(Field.newUnqualified(field.getName(), field.getType()));
                }
            }
            else if (item instanceof SingleColumn) {
                SingleColumn column = (SingleColumn) item;
                Expression expression = column.getExpression();

                Optional<String> alias = column.getAlias();
                if (!alias.isPresent()) {
                    QualifiedName name = null;
                    if (expression instanceof QualifiedNameReference) {
                        name = ((QualifiedNameReference) expression).getName();
                    }
                    else if (expression instanceof DereferenceExpression) {
                        name = DereferenceExpression.getQualifiedName((DereferenceExpression) expression);
                    }
                    if (name != null) {
                        alias = Optional.of(getLast(name.getOriginalParts()));
                    }
                }

                outputFields.add(Field.newUnqualified(alias, analysis.getType(expression))); // TODO don't use analysis as a side-channel. Use outputExpressions to look up the type
            }
            else {
                throw new IllegalArgumentException("Unsupported SelectItem type: " + item.getClass().getName());
            }
        }

        return new RelationType(outputFields.build());
    }

    private List<FieldOrExpression> analyzeSelect(QuerySpecification node, RelationType tupleDescriptor, AnalysisContext context)
    {
        ImmutableList.Builder<FieldOrExpression> outputExpressionBuilder = ImmutableList.builder();

        for (SelectItem item : node.getSelect().getSelectItems()) {
            if (item instanceof AllColumns) {
                // expand * and T.*
                Optional<QualifiedName> starPrefix = ((AllColumns) item).getPrefix();

                List<Field> fields = tupleDescriptor.resolveFieldsWithPrefix(starPrefix);
                if (fields.isEmpty()) {
                    if (starPrefix.isPresent()) {
                        throw new SemanticException(MISSING_TABLE, item, "Table '%s' not found", starPrefix.get());
                    }
                    else {
                        throw new SemanticException(WILDCARD_WITHOUT_FROM, item, "SELECT * not allowed in queries without FROM clause");
                    }
                }

                for (Field field : fields) {
                    int fieldIndex = tupleDescriptor.indexOf(field);
                    outputExpressionBuilder.add(new FieldOrExpression(fieldIndex));

                    if (node.getSelect().isDistinct() && !field.getType().isComparable()) {
                        throw new SemanticException(TYPE_MISMATCH, node.getSelect(), "DISTINCT can only be applied to comparable types (actual: %s)", field.getType());
                    }
                }
            }
            else if (item instanceof SingleColumn) {
                SingleColumn column = (SingleColumn) item;
                ExpressionAnalysis expressionAnalysis = analyzeExpression(column.getExpression(), tupleDescriptor, context);
                analysis.recordSubqueries(node, expressionAnalysis);
                outputExpressionBuilder.add(new FieldOrExpression(column.getExpression()));

                Type type = expressionAnalysis.getType(column.getExpression());
                if (node.getSelect().isDistinct() && !type.isComparable()) {
                    throw new SemanticException(TYPE_MISMATCH, node.getSelect(), "DISTINCT can only be applied to comparable types (actual: %s): %s", type, column.getExpression());
                }
            }
            else {
                throw new IllegalArgumentException("Unsupported SelectItem type: " + item.getClass().getName());
            }
        }

        ImmutableList<FieldOrExpression> result = outputExpressionBuilder.build();
        analysis.setOutputExpressions(node, result);

        return result;
    }

    public void analyzeWhere(Node node, RelationType tupleDescriptor, AnalysisContext context, Expression predicate)
    {
        Analyzer.verifyNoAggregatesOrWindowFunctions(metadata, predicate, "WHERE");

        ExpressionAnalysis expressionAnalysis = analyzeExpression(predicate, tupleDescriptor, context);
        analysis.recordSubqueries(node, expressionAnalysis);

        Type predicateType = expressionAnalysis.getType(predicate);
        if (!predicateType.equals(BOOLEAN)) {
            if (!predicateType.equals(UNKNOWN)) {
                throw new SemanticException(TYPE_MISMATCH, predicate, "WHERE clause must evaluate to a boolean: actual type %s", predicateType);
            }
            // coerce null to boolean
            analysis.addCoercion(predicate, BOOLEAN);
        }

        analysis.setWhere(node, predicate);
    }

    private RelationType analyzeFrom(QuerySpecification node, AnalysisContext context)
    {
        RelationType fromDescriptor = new RelationType();

        if (node.getFrom().isPresent()) {
            fromDescriptor = process(node.getFrom().get(), context);
        }

        return fromDescriptor;
    }

    private void analyzeAggregations(QuerySpecification node,
            RelationType tupleDescriptor,
            List<List<FieldOrExpression>> groupingSets,
            List<FieldOrExpression> outputExpressions,
            List<FieldOrExpression> orderByExpressions,
            AnalysisContext context,
            Set<Expression> columnReferences)
    {
        List<FunctionCall> aggregates = extractAggregates(node);

        if (context.isApproximate()) {
            if (aggregates.stream().anyMatch(FunctionCall::isDistinct)) {
                throw new SemanticException(NOT_SUPPORTED, node, "DISTINCT aggregations not supported for approximate queries");
            }
        }

        ImmutableList<FieldOrExpression> allGroupingColumns = groupingSets.stream()
                .flatMap(Collection::stream)
                .distinct()
                .collect(toImmutableList());

        // is this an aggregation query?
        if (!groupingSets.isEmpty()) {
            // ensure SELECT, ORDER BY and HAVING are constant with respect to group
            // e.g, these are all valid expressions:
            //     SELECT f(a) GROUP BY a
            //     SELECT f(a + 1) GROUP BY a + 1
            //     SELECT a + sum(b) GROUP BY a

            for (FieldOrExpression fieldOrExpression : Iterables.concat(outputExpressions, orderByExpressions)) {
                verifyAggregations(node, allGroupingColumns, tupleDescriptor, fieldOrExpression, columnReferences);
            }

            if (node.getHaving().isPresent()) {
                verifyAggregations(node, allGroupingColumns, tupleDescriptor, new FieldOrExpression(node.getHaving().get()), columnReferences);
            }
        }
    }

    private List<FunctionCall> extractAggregates(QuerySpecification node)
    {
        AggregateExtractor extractor = new AggregateExtractor(metadata);
        for (SelectItem item : node.getSelect().getSelectItems()) {
            if (item instanceof SingleColumn) {
                extractor.process(((SingleColumn) item).getExpression(), null);
            }
        }

        for (SortItem item : node.getOrderBy()) {
            extractor.process(item.getSortKey(), null);
        }

        if (node.getHaving().isPresent()) {
            extractor.process(node.getHaving().get(), null);
        }

        List<FunctionCall> aggregates = extractor.getAggregates();
        analysis.setAggregates(node, aggregates);

        return aggregates;
    }

    private void verifyAggregations(
            QuerySpecification node,
            List<FieldOrExpression> groupByExpressions,
            RelationType tupleDescriptor,
            FieldOrExpression fieldOrExpression,
            Set<Expression> columnReferences)
    {
        AggregationAnalyzer analyzer = new AggregationAnalyzer(groupByExpressions, metadata, tupleDescriptor, columnReferences);

        if (fieldOrExpression.isExpression()) {
            analyzer.analyze(fieldOrExpression.getExpression());
        }
        else {
            int fieldIndex = fieldOrExpression.getFieldIndex();
            if (!analyzer.analyze(fieldIndex)) {
                Field field = tupleDescriptor.getFieldByIndex(fieldIndex);

                if (field.getRelationAlias().isPresent()) {
                    if (field.getName().isPresent()) {
                        throw new SemanticException(MUST_BE_AGGREGATE_OR_GROUP_BY, node, "Column '%s.%s' not in GROUP BY clause", field.getRelationAlias().get(), field.getName().get());
                    }
                    else {
                        throw new SemanticException(MUST_BE_AGGREGATE_OR_GROUP_BY, node, "Columns from '%s' not in GROUP BY clause", field.getRelationAlias().get());
                    }
                }
                else {
                    if (field.getName().isPresent()) {
                        throw new SemanticException(MUST_BE_AGGREGATE_OR_GROUP_BY, node, "Column '%s' not in GROUP BY clause", field.getName().get());
                    }
                    else {
                        throw new SemanticException(MUST_BE_AGGREGATE_OR_GROUP_BY, node, "Some columns from FROM clause not in GROUP BY clause");
                    }
                }
            }
        }
    }

    private RelationType analyzeView(Query query, QualifiedObjectName name, Optional<String> catalog, Optional<String> schema, Optional<String> owner, Table node)
    {
        try {
            // run view as view owner if set; otherwise, run as session user
            Identity identity;
            AccessControl viewAccessControl;
            if (owner.isPresent()) {
                identity = new Identity(owner.get(), Optional.empty());
                viewAccessControl = new ViewAccessControl(accessControl);
            }
            else {
                identity = session.getIdentity();
                viewAccessControl = accessControl;
            }

            Session viewSession = Session.builder(metadata.getSessionPropertyManager())
                    .setQueryId(session.getQueryId())
                    .setTransactionId(session.getTransactionId().orElse(null))
                    .setIdentity(identity)
                    .setSource(session.getSource().orElse(null))
                    .setCatalog(catalog.orElse(null))
                    .setSchema(schema.orElse(null))
                    .setTimeZoneKey(session.getTimeZoneKey())
                    .setLocale(session.getLocale())
                    .setRemoteUserAddress(session.getRemoteUserAddress().orElse(null))
                    .setUserAgent(session.getUserAgent().orElse(null))
                    .setStartTime(session.getStartTime())
                    .build();

            StatementAnalyzer analyzer = new StatementAnalyzer(analysis, metadata, sqlParser, viewAccessControl, viewSession, experimentalSyntaxEnabled, Optional.empty());
            return analyzer.process(query, new AnalysisContext());
        }
        catch (RuntimeException e) {
            throw new SemanticException(VIEW_ANALYSIS_ERROR, node, "Failed analyzing stored view '%s': %s", name, e.getMessage());
        }
    }

    private Query parseView(String view, QualifiedObjectName name, Table node)
    {
        try {
            Statement statement = sqlParser.createStatement(view);
            return checkType(statement, Query.class, "parsed view");
        }
        catch (ParsingException e) {
            throw new SemanticException(VIEW_PARSE_ERROR, node, "Failed parsing stored view '%s': %s", name, e.getMessage());
        }
    }

    private static boolean isViewStale(List<ViewDefinition.ViewColumn> columns, Collection<Field> fields)
    {
        if (columns.size() != fields.size()) {
            return true;
        }

        List<Field> fieldList = ImmutableList.copyOf(fields);
        for (int i = 0; i < columns.size(); i++) {
            ViewDefinition.ViewColumn column = columns.get(i);
            Field field = fieldList.get(i);
            if (!column.getName().equals(field.getName().orElse(null)) ||
                    !column.getType().equals(field.getType())) {
                return true;
            }
        }

        return false;
    }

    private ExpressionAnalysis analyzeExpression(Expression expression, RelationType tupleDescriptor, AnalysisContext context)
    {
        return ExpressionAnalyzer.analyzeExpression(
                session,
                metadata,
                accessControl, sqlParser,
                tupleDescriptor,
                analysis,
                experimentalSyntaxEnabled,
                context,
                expression);
    }

    private static List<FieldOrExpression> descriptorToFields(RelationType tupleDescriptor)
    {
        ImmutableList.Builder<FieldOrExpression> builder = ImmutableList.builder();
        for (int fieldIndex = 0; fieldIndex < tupleDescriptor.getAllFieldCount(); fieldIndex++) {
            builder.add(new FieldOrExpression(fieldIndex));
        }
        return builder.build();
    }

    private void analyzeWith(Query node, AnalysisContext context)
    {
        // analyze WITH clause
        if (!node.getWith().isPresent()) {
            return;
        }

        With with = node.getWith().get();
        if (with.isRecursive()) {
            throw new SemanticException(NOT_SUPPORTED, with, "Recursive WITH queries are not supported");
        }

        for (WithQuery withQuery : with.getQueries()) {
            if (withQuery.getColumnNames() != null && !withQuery.getColumnNames().isEmpty()) {
                throw new SemanticException(NOT_SUPPORTED, withQuery, "Column alias not supported in WITH queries");
            }

            Query query = withQuery.getQuery();
            process(query, context);

            String name = withQuery.getName();
            if (context.isNamedQueryDeclared(name)) {
                throw new SemanticException(DUPLICATE_RELATION, withQuery, "WITH query name '%s' specified more than once", name);
            }

            context.addNamedQuery(name, query);
        }
    }

    private void analyzeOrderBy(Query node, RelationType tupleDescriptor, AnalysisContext context)
    {
        List<SortItem> items = node.getOrderBy();

        ImmutableList.Builder<FieldOrExpression> orderByFieldsBuilder = ImmutableList.builder();

        if (!items.isEmpty()) {
            for (SortItem item : items) {
                Expression expression = item.getSortKey();

                FieldOrExpression orderByField;
                if (expression instanceof LongLiteral) {
                    // this is an ordinal in the output tuple

                    long ordinal = ((LongLiteral) expression).getValue();
                    if (ordinal < 1 || ordinal > tupleDescriptor.getVisibleFieldCount()) {
                        throw new SemanticException(INVALID_ORDINAL, expression, "ORDER BY position %s is not in select list", ordinal);
                    }

                    orderByField = new FieldOrExpression(Ints.checkedCast(ordinal - 1));
                }
                else {
                    // otherwise, just use the expression as is
                    orderByField = new FieldOrExpression(expression);
                    ExpressionAnalysis expressionAnalysis = ExpressionAnalyzer.analyzeExpression(session,
                            metadata,
                            accessControl, sqlParser,
                            tupleDescriptor,
                            analysis,
                            experimentalSyntaxEnabled,
                            context,
                            orderByField.getExpression());
                    analysis.recordSubqueries(node, expressionAnalysis);
                }

                orderByFieldsBuilder.add(orderByField);
            }
        }

        analysis.setOrderByExpressions(node, orderByFieldsBuilder.build());
    }

    private static Relation from(String catalog, SchemaTableName table)
    {
        return table(QualifiedName.of(catalog, table.getSchemaName(), table.getTableName()));
    }
}
