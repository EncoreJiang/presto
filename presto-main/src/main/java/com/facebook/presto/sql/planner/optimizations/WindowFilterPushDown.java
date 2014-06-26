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
package com.facebook.presto.sql.planner.optimizations;

import com.facebook.presto.metadata.Metadata;
import com.facebook.presto.metadata.Signature;
import com.facebook.presto.operator.window.RowNumberFunction;
import com.facebook.presto.operator.window.WindowFunction;
import com.facebook.presto.spi.ConnectorSession;
import com.facebook.presto.spi.type.Type;
import com.facebook.presto.sql.planner.DependencyExtractor;
import com.facebook.presto.sql.planner.PlanNodeIdAllocator;
import com.facebook.presto.sql.planner.Symbol;
import com.facebook.presto.sql.planner.SymbolAllocator;
import com.facebook.presto.sql.planner.plan.FilterNode;
import com.facebook.presto.sql.planner.plan.LimitNode;
import com.facebook.presto.sql.planner.plan.PlanNode;
import com.facebook.presto.sql.planner.plan.PlanNodeRewriter;
import com.facebook.presto.sql.planner.plan.PlanRewriter;
import com.facebook.presto.sql.planner.plan.RowNumberLimitNode;
import com.facebook.presto.sql.planner.plan.TableScanNode;
import com.facebook.presto.sql.planner.plan.TopNNode;
import com.facebook.presto.sql.planner.plan.TopNRowNumberNode;
import com.facebook.presto.sql.planner.plan.WindowNode;
import com.facebook.presto.sql.tree.ComparisonExpression;
import com.facebook.presto.sql.tree.DefaultExpressionTraversalVisitor;
import com.facebook.presto.sql.tree.DoubleLiteral;
import com.facebook.presto.sql.tree.Expression;
import com.facebook.presto.sql.tree.FunctionCall;
import com.facebook.presto.sql.tree.Literal;
import com.facebook.presto.sql.tree.LongLiteral;
import com.facebook.presto.sql.tree.QualifiedNameReference;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

public class WindowFilterPushDown
        extends PlanOptimizer
{
    private final Metadata metadata;

    public WindowFilterPushDown(Metadata metadata)
    {
        this.metadata = checkNotNull(metadata, "metadata is null");
    }

    @Override
    public PlanNode optimize(PlanNode plan, ConnectorSession session, Map<Symbol, Type> types, SymbolAllocator symbolAllocator, PlanNodeIdAllocator idAllocator)
    {
        checkNotNull(plan, "plan is null");
        checkNotNull(session, "session is null");
        checkNotNull(types, "types is null");
        checkNotNull(symbolAllocator, "symbolAllocator is null");
        checkNotNull(idAllocator, "idAllocator is null");

        return PlanRewriter.rewriteWith(new Rewriter(idAllocator, metadata), plan, null);
    }

    private static class WindowContext
    {
        private final Optional<WindowNode> windowNode;
        private final Expression expression;

        private WindowContext(Optional<WindowNode> windowNode, Expression expression)
        {
            this.windowNode = checkNotNull(windowNode, "windowNode is null");
            this.expression = checkNotNull(expression, "expression is null");
        }

        public Optional<WindowNode> getWindowNode()
        {
            return windowNode;
        }

        public Expression getExpression()
        {
            return expression;
        }
    }

    private static class Rewriter
            extends PlanNodeRewriter<WindowContext>
    {
        private final PlanNodeIdAllocator idAllocator;
        private final Metadata metadata;

        private Rewriter(PlanNodeIdAllocator idAllocator, Metadata metadata)
        {
            this.metadata = checkNotNull(metadata, "metadata is null");
            this.idAllocator = checkNotNull(idAllocator, "idAllocator is null");
        }

        @Override
        public PlanNode rewriteWindow(WindowNode node, WindowContext context, PlanRewriter<WindowContext> planRewriter)
        {
            if (context != null &&
                    node.getWindowFunctions().size() == 1 &&
                    isRowNumberFunction(node) &&
                    filterContainsWindowFunctions(node, context.getExpression()) &&
                    extractLimitOptional(node, context.getExpression()).isPresent()) {
                Optional<Integer> limit = extractLimitOptional(node, context.getExpression());
                if (limit.isPresent()) {
                    checkArgument(limit.get() <= Integer.MAX_VALUE, "filter on row_number greater than allowed value");
                }
                if (node.getPartitionBy().isEmpty()) {
                    WindowContext windowContext = new WindowContext(Optional.of(node), new LongLiteral(String.valueOf(limit.get())));
                    PlanNode rewrittenSource = planRewriter.rewrite(node.getSource(), windowContext);
                    return new WindowNode(idAllocator.getNextId(),
                            rewrittenSource,
                            node.getPartitionBy(),
                            node.getOrderBy(),
                            node.getOrderings(),
                            node.getWindowFunctions(),
                            node.getSignatures());
                }
                else if (node.getOrderBy().isEmpty()) {
                    PlanNode rewrittenSource = planRewriter.rewrite(node.getSource(), null);
                    return new RowNumberLimitNode(idAllocator.getNextId(),
                            rewrittenSource,
                            node.getPartitionBy(),
                            node.getWindowFunctions(),
                            node.getSignatures(),
                            limit.get());
                }
                else {
                    PlanNode rewrittenSource = planRewriter.rewrite(node.getSource(), null);
                    return new TopNRowNumberNode(idAllocator.getNextId(),
                            rewrittenSource,
                            node.getPartitionBy(),
                            node.getOrderBy(),
                            node.getOrderings(),
                            node.getWindowFunctions(),
                            node.getSignatures(),
                            limit.get(),
                            false);

                }
            }
            return planRewriter.defaultRewrite(node, null);
        }

        private boolean isRowNumberFunction(WindowNode node)
        {
            checkArgument(node.getWindowFunctions().size() == 1);
            Map.Entry<Symbol, FunctionCall> entry = Iterables.getOnlyElement(node.getWindowFunctions().entrySet());
            ImmutableList.Builder<Integer> arguments = ImmutableList.builder();
            List<Symbol> outputSymbols = node.getOutputSymbols();
            for (Expression argument : entry.getValue().getArguments()) {
                Symbol argumentSymbol = Symbol.fromQualifiedName(((QualifiedNameReference) argument).getName());
                arguments.add(outputSymbols.indexOf(argumentSymbol));
            }
            Symbol symbol = entry.getKey();
            Signature signature = node.getSignatures().get(symbol);
            WindowFunction function = metadata.getExactFunction(signature).bindWindowFunction(arguments.build()).createWindowFunction();
            return function instanceof RowNumberFunction;
        }

        private boolean filterContainsWindowFunctions(WindowNode node, Expression filterPredicate)
        {
            Set<Symbol> windowFunctionSymbols = node.getWindowFunctions().keySet();
            Sets.SetView<Symbol> commonSymbols = Sets.intersection(DependencyExtractor.extractUnique(filterPredicate), windowFunctionSymbols);
            return !commonSymbols.isEmpty();
        }

        @Override
        public PlanNode rewriteLimit(LimitNode node, WindowContext context, PlanRewriter<WindowContext> planRewriter)
        {
            if (context != null && context.getExpression() instanceof LongLiteral &&
                    ((LongLiteral) context.getExpression()).getValue() < node.getCount()) {
                PlanNode rewrittenSource = planRewriter.rewrite(node.getSource(), context);
                if (rewrittenSource != node.getSource()) {
                    return rewrittenSource;
                }
            }
            return planRewriter.defaultRewrite(node, null);
        }

        @Override
        public PlanNode rewriteTopN(TopNNode node, WindowContext context, PlanRewriter<WindowContext> planRewriter)
        {
            return planRewriter.defaultRewrite(node, null);
        }

        @Override
        public PlanNode rewriteTableScan(TableScanNode node, WindowContext context, PlanRewriter<WindowContext> planRewriter)
        {
            if (context != null && context.getExpression() instanceof LongLiteral && context.getWindowNode().isPresent()) {
                PlanNode rewrittenNode = planRewriter.rewrite(node, null);
                WindowNode windowNode = context.getWindowNode().get();
                if (windowNode.getOrderBy().isEmpty()) {
                    return new LimitNode(idAllocator.getNextId(), rewrittenNode, ((LongLiteral) context.getExpression()).getValue(), Optional.<Symbol>absent());
                }
                else {
                    return new TopNNode(
                            idAllocator.getNextId(),
                            rewrittenNode,
                            ((LongLiteral) context.getExpression()).getValue(),
                            windowNode.getOrderBy(),
                            windowNode.getOrderings(),
                            false,
                            Optional.<Symbol>absent());
                }
            }
            return planRewriter.defaultRewrite(node, null);
        }

        @Override
        public PlanNode rewriteFilter(FilterNode node, WindowContext context, PlanRewriter<WindowContext> planRewriter)
        {
            WindowContext sourceContext = new WindowContext(Optional.<WindowNode>absent(), node.getPredicate());
            PlanNode rewrittenSource = planRewriter.rewrite(node.getSource(), sourceContext);
            if (rewrittenSource != node.getSource()) {
                return rewrittenSource;
            }
            return planRewriter.defaultRewrite(node, null);
        }
    }

    private static Optional<Integer> extractLimitOptional(WindowNode node, Expression filterPredicate)
    {
        Symbol rowNumberSymbol = Iterables.getOnlyElement(node.getWindowFunctions().entrySet()).getKey();
        return WindowLimitExtractor.extract(filterPredicate, rowNumberSymbol);
    }

    public static final class WindowLimitExtractor
    {
        private WindowLimitExtractor() {}

        public static Optional<Integer> extract(Expression expression, Symbol rowNumberSymbol)
        {
            Visitor visitor = new Visitor();
            Long limit = visitor.process(expression, rowNumberSymbol);
            if (limit == null) {
                return Optional.absent();
            }
            else {
                checkArgument(limit < Integer.MAX_VALUE, "filter on row_number greater than allowed value");

                return Optional.of(limit.intValue());
            }
        }

        private static class Visitor
                extends DefaultExpressionTraversalVisitor<Long, Symbol>
        {
            @Override
            protected Long visitComparisonExpression(ComparisonExpression node, Symbol rowNumberSymbol)
            {
                QualifiedNameReference reference = extractReference(node);
                Literal literal = extractLiteral(node);
                if (!Symbol.fromQualifiedName(reference.getName()).equals(rowNumberSymbol)) {
                    return null;
                }

                if (node.getLeft() instanceof QualifiedNameReference && node.getRight() instanceof Literal) {
                    if (node.getType() == ComparisonExpression.Type.LESS_THAN_OR_EQUAL) {
                        return extractValue(literal);
                    }
                    else if (node.getType() == ComparisonExpression.Type.LESS_THAN) {
                        return extractValue(literal) - 1;
                    }
                }
                else if (node.getLeft() instanceof Literal && node.getRight() instanceof QualifiedNameReference) {
                    if (node.getType() == ComparisonExpression.Type.GREATER_THAN_OR_EQUAL) {
                        return extractValue(literal);
                    }
                    else if (node.getType() == ComparisonExpression.Type.GREATER_THAN) {
                        return extractValue(literal) - 1;
                    }
                }
                return null;
            }
        }

        private static QualifiedNameReference extractReference(ComparisonExpression expression)
        {
            if (expression.getLeft() instanceof QualifiedNameReference) {
                return (QualifiedNameReference) expression.getLeft();
            }
            else if (expression.getRight() instanceof QualifiedNameReference) {
                return (QualifiedNameReference) expression.getRight();
            }
            throw new IllegalArgumentException("Comparison does not have a child of type QualifiedNameReference");
        }

        private static Literal extractLiteral(ComparisonExpression expression)
        {
            if (expression.getLeft() instanceof Literal) {
                return (Literal) expression.getLeft();
            }
            else if (expression.getRight() instanceof Literal) {
                return (Literal) expression.getRight();
            }
            throw new IllegalArgumentException("Comparison does not have a child of type Literal");
        }

        private static long extractValue(Literal literal)
        {
            if (literal instanceof DoubleLiteral) {
                return (long) ((DoubleLiteral) literal).getValue();
            }
            if (literal instanceof LongLiteral) {
                return ((LongLiteral) literal).getValue();
            }
            throw new IllegalArgumentException("Row number compared to non numeric literal");
        }
    }
}
