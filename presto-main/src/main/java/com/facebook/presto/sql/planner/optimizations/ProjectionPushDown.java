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

import com.facebook.presto.Session;
import com.facebook.presto.spi.type.Type;
import com.facebook.presto.sql.planner.ExpressionSymbolInliner;
import com.facebook.presto.sql.planner.PlanNodeIdAllocator;
import com.facebook.presto.sql.planner.Symbol;
import com.facebook.presto.sql.planner.SymbolAllocator;
import com.facebook.presto.sql.planner.plan.ExchangeNode;
import com.facebook.presto.sql.planner.plan.PlanNode;
import com.facebook.presto.sql.planner.plan.ProjectNode;
import com.facebook.presto.sql.planner.plan.SimplePlanRewriter;
import com.facebook.presto.sql.planner.plan.UnionNode;
import com.facebook.presto.sql.tree.Expression;
import com.facebook.presto.sql.tree.ExpressionTreeRewriter;
import com.facebook.presto.sql.tree.QualifiedNameReference;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static com.facebook.presto.sql.planner.plan.ChildReplacer.replaceChildren;
import static java.util.Objects.requireNonNull;

public class ProjectionPushDown
        extends PlanOptimizer
{
    @Override
    public PlanNode optimize(PlanNode plan, Session session, Map<Symbol, Type> types, SymbolAllocator symbolAllocator, PlanNodeIdAllocator idAllocator)
    {
        requireNonNull(plan, "plan is null");
        requireNonNull(session, "session is null");
        requireNonNull(types, "types is null");
        requireNonNull(symbolAllocator, "symbolAllocator is null");
        requireNonNull(idAllocator, "idAllocator is null");

        return SimplePlanRewriter.rewriteWith(new Rewriter(idAllocator, symbolAllocator), plan);
    }

    private static class Rewriter
            extends SimplePlanRewriter<Void>
    {
        private final PlanNodeIdAllocator idAllocator;
        private final SymbolAllocator symbolAllocator;

        public Rewriter(PlanNodeIdAllocator idAllocator, SymbolAllocator symbolAllocator)
        {
            this.idAllocator = requireNonNull(idAllocator, "idAllocator is null");
            this.symbolAllocator = requireNonNull(symbolAllocator, "symbolAllocator is null");
        }

        @Override
        public PlanNode visitProject(ProjectNode node, RewriteContext<Void> context)
        {
            PlanNode source = context.rewrite(node.getSource());

            if (source instanceof UnionNode) {
                return pushProjectionThrough(node, (UnionNode) source);
            }
            else if (source instanceof ExchangeNode) {
                return pushProjectionThrough(node, (ExchangeNode) source);
            }
            return replaceChildren(node, ImmutableList.of(source));
        }

        private PlanNode pushProjectionThrough(ProjectNode node, UnionNode source)
        {
            // OutputLayout of the resultant Union, will be same as the layout of the Project
            List<Symbol> outputLayout = node.getOutputSymbols();

            // Mapping from the output symbol to ordered list of symbols from each of the sources
            ImmutableListMultimap.Builder<Symbol, Symbol> mappings = ImmutableListMultimap.builder();

            // sources for the resultant UnionNode
            ImmutableList.Builder<PlanNode> outputSources = ImmutableList.builder();

            for (int i = 0; i < source.getSources().size(); i++) {
                Map<Symbol, QualifiedNameReference> outputToInput = source.sourceSymbolMap(i);   // Map: output of union -> input of this source to the union
                ImmutableMap.Builder<Symbol, Expression> assignments = ImmutableMap.builder();      // assignments for the new ProjectNode

                // mapping from current ProjectNode to new ProjectNode, used to identify the output layout
                Map<Symbol, Symbol> projectSymbolMapping = new HashMap<>();

                // Translate the assignments in the ProjectNode using symbols of the source of the UnionNode
                for (Map.Entry<Symbol, Expression> entry : node.getAssignments().entrySet()) {
                    Expression translatedExpression = translateExpression(entry.getValue(), outputToInput);
                    Type type = symbolAllocator.getTypes().get(entry.getKey());
                    Symbol symbol = symbolAllocator.newSymbol(translatedExpression, type);
                    assignments.put(symbol, translatedExpression);
                    projectSymbolMapping.put(entry.getKey(), symbol);
                }
                outputSources.add(new ProjectNode(idAllocator.getNextId(), source.getSources().get(i), assignments.build()));
                outputLayout.forEach(symbol -> mappings.put(symbol, projectSymbolMapping.get(symbol)));
            }

            return new UnionNode(node.getId(), outputSources.build(), mappings.build());
        }

        private PlanNode pushProjectionThrough(ProjectNode node, ExchangeNode exchange)
        {
            ImmutableList.Builder<PlanNode> newSourceBuilder = ImmutableList.builder();
            ImmutableList.Builder<List<Symbol>> inputsBuilder = ImmutableList.builder();
            for (int i = 0; i < exchange.getSources().size(); i++) {
                Map<Symbol, QualifiedNameReference> outputToInputMap = extractExchangeOutputToInput(exchange, i);

                Map<Symbol, Expression> projections = new LinkedHashMap<>(); // Use LinkedHashMap to make output symbol order deterministic
                ImmutableList.Builder<Symbol> inputs = ImmutableList.builder();
                if (exchange.getPartitionKeys().isPresent()) {
                    // Need to retain the partition keys for the exchange
                    exchange.getPartitionKeys().get().stream()
                            .map(outputToInputMap::get)
                            .forEach(nameReference -> {
                                Symbol symbol = Symbol.fromQualifiedName(nameReference.getName());
                                projections.put(symbol, nameReference);
                                inputs.add(symbol);
                            });
                }
                if (exchange.getHashSymbol().isPresent()) {
                    // Need to retain the hash symbol for the exchange
                    projections.put(exchange.getHashSymbol().get(), exchange.getHashSymbol().get().toQualifiedNameReference());
                    inputs.add(exchange.getHashSymbol().get());
                }
                for (Map.Entry<Symbol, Expression> projection : node.getAssignments().entrySet()) {
                    Expression translatedExpression = translateExpression(projection.getValue(), outputToInputMap);
                    Type type = symbolAllocator.getTypes().get(projection.getKey());
                    Symbol symbol = symbolAllocator.newSymbol(translatedExpression, type);
                    projections.put(symbol, translatedExpression);
                    inputs.add(symbol);
                }
                newSourceBuilder.add(new ProjectNode(idAllocator.getNextId(), exchange.getSources().get(i), projections));
                inputsBuilder.add(inputs.build());
            }

            // Construct the output symbols in the same order as the sources
            ImmutableList.Builder<Symbol> outputBuilder = ImmutableList.builder();
            if (exchange.getPartitionKeys().isPresent()) {
                exchange.getPartitionKeys().get().stream()
                        .forEach(outputBuilder::add);
            }
            if (exchange.getHashSymbol().isPresent()) {
                outputBuilder.add(exchange.getHashSymbol().get());
            }
            for (Map.Entry<Symbol, Expression> projection : node.getAssignments().entrySet()) {
                outputBuilder.add(projection.getKey());
            }

            return new ExchangeNode(
                    exchange.getId(),
                    exchange.getType(),
                    exchange.getPartitionFunction(),
                    newSourceBuilder.build(),
                    outputBuilder.build(),
                    inputsBuilder.build());
        }
    }

    private static Map<Symbol, QualifiedNameReference> extractExchangeOutputToInput(ExchangeNode exchange, int sourceIndex)
    {
        Map<Symbol, QualifiedNameReference> outputToInputMap = new HashMap<>();
        for (int i = 0; i < exchange.getOutputSymbols().size(); i++) {
            outputToInputMap.put(exchange.getOutputSymbols().get(i), exchange.getInputs().get(sourceIndex).get(i).toQualifiedNameReference());
        }
        return outputToInputMap;
    }

    private static Expression translateExpression(Expression inputExpression, Map<Symbol, QualifiedNameReference> symbolMapping)
    {
        return ExpressionTreeRewriter.rewriteWith(new ExpressionSymbolInliner(symbolMapping), inputExpression);
    }
}
