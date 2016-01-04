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
package com.facebook.presto.metadata;

import com.facebook.presto.operator.scalar.ScalarFunctionImplementation;
import com.facebook.presto.spi.type.Type;
import com.facebook.presto.spi.type.TypeManager;
import com.facebook.presto.spi.type.TypeSignature;
import com.facebook.presto.util.ImmutableCollectors;
import com.google.common.collect.ImmutableList;

import java.lang.invoke.MethodHandle;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static com.facebook.presto.metadata.FunctionRegistry.mangleOperatorName;
import static java.util.Objects.requireNonNull;

public abstract class SqlOperator
        extends SqlScalarFunction
{
    public static SqlOperator create(
            OperatorType operatorType,
            List<TypeSignature> argumentTypes,
            TypeSignature returnType,
            MethodHandle methodHandle,
            Optional<MethodHandle> instanceFactory,
            boolean nullable,
            List<Boolean> nullableArguments)
    {
        return new SimpleSqlOperator(operatorType, argumentTypes, returnType, methodHandle, instanceFactory, nullable, nullableArguments);
    }

    protected SqlOperator(OperatorType operatorType, List<TypeParameter> typeParameters, String returnType, List<String> argumentTypes)
    {
        super(mangleOperatorName(operatorType), typeParameters, returnType, argumentTypes);
    }

    @Override
    public final boolean isHidden()
    {
        return true;
    }

    @Override
    public final boolean isDeterministic()
    {
        return true;
    }

    @Override
    public final String getDescription()
    {
        // Operators are internal, and don't need a description
        return null;
    }

    public static class SimpleSqlOperator
            extends SqlOperator
    {
        private final MethodHandle methodHandle;
        private final Optional<MethodHandle> instanceFactory;
        private final boolean nullable;
        private final List<Boolean> nullableArguments;

        public SimpleSqlOperator(
                OperatorType operatorType,
                List<TypeSignature> argumentTypes,
                TypeSignature returnType,
                MethodHandle methodHandle,
                Optional<MethodHandle> instanceFactory,
                boolean nullable,
                List<Boolean> nullableArguments)
        {
            super(operatorType,
                    ImmutableList.of(),
                    returnType.toString(),
                    argumentTypes.stream()
                            .map(TypeSignature::toString)
                            .collect(ImmutableCollectors.toImmutableList()));
            this.methodHandle = requireNonNull(methodHandle, "methodHandle is null");
            this.instanceFactory = requireNonNull(instanceFactory, "instanceFactory is null");
            this.nullable = nullable;
            this.nullableArguments = ImmutableList.copyOf(requireNonNull(nullableArguments, "nullableArguments is null"));
        }

        @Override
        public ScalarFunctionImplementation specialize(Map<String, Type> types, int arity, TypeManager typeManager, FunctionRegistry functionRegistry)
        {
            return new ScalarFunctionImplementation(nullable, nullableArguments, methodHandle, instanceFactory, isDeterministic());
        }
    }
}
