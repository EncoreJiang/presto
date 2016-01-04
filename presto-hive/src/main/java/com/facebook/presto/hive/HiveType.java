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
package com.facebook.presto.hive;

import com.facebook.presto.spi.PrestoException;
import com.facebook.presto.spi.type.StandardTypes;
import com.facebook.presto.spi.type.Type;
import com.facebook.presto.spi.type.TypeManager;
import com.facebook.presto.spi.type.TypeSignature;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;
import com.google.common.collect.ImmutableList;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.typeinfo.ListTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.MapTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.PrimitiveTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.StructTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfo;

import javax.annotation.Nonnull;

import java.util.List;

import static com.facebook.presto.hive.HiveUtil.isArrayType;
import static com.facebook.presto.hive.HiveUtil.isMapType;
import static com.facebook.presto.hive.HiveUtil.isRowType;
import static com.facebook.presto.hive.util.Types.checkType;
import static com.facebook.presto.spi.StandardErrorCode.NOT_SUPPORTED;
import static com.facebook.presto.spi.type.BigintType.BIGINT;
import static com.facebook.presto.spi.type.BooleanType.BOOLEAN;
import static com.facebook.presto.spi.type.DateType.DATE;
import static com.facebook.presto.spi.type.DoubleType.DOUBLE;
import static com.facebook.presto.spi.type.TimestampType.TIMESTAMP;
import static com.facebook.presto.spi.type.VarbinaryType.VARBINARY;
import static com.facebook.presto.spi.type.VarcharType.VARCHAR;
import static java.lang.String.format;
import static java.util.Objects.requireNonNull;
import static java.util.stream.Collectors.toList;
import static org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector.Category;
import static org.apache.hadoop.hive.serde2.typeinfo.TypeInfoFactory.binaryTypeInfo;
import static org.apache.hadoop.hive.serde2.typeinfo.TypeInfoFactory.booleanTypeInfo;
import static org.apache.hadoop.hive.serde2.typeinfo.TypeInfoFactory.byteTypeInfo;
import static org.apache.hadoop.hive.serde2.typeinfo.TypeInfoFactory.dateTypeInfo;
import static org.apache.hadoop.hive.serde2.typeinfo.TypeInfoFactory.doubleTypeInfo;
import static org.apache.hadoop.hive.serde2.typeinfo.TypeInfoFactory.floatTypeInfo;
import static org.apache.hadoop.hive.serde2.typeinfo.TypeInfoFactory.getListTypeInfo;
import static org.apache.hadoop.hive.serde2.typeinfo.TypeInfoFactory.getMapTypeInfo;
import static org.apache.hadoop.hive.serde2.typeinfo.TypeInfoFactory.getStructTypeInfo;
import static org.apache.hadoop.hive.serde2.typeinfo.TypeInfoFactory.intTypeInfo;
import static org.apache.hadoop.hive.serde2.typeinfo.TypeInfoFactory.longTypeInfo;
import static org.apache.hadoop.hive.serde2.typeinfo.TypeInfoFactory.shortTypeInfo;
import static org.apache.hadoop.hive.serde2.typeinfo.TypeInfoFactory.stringTypeInfo;
import static org.apache.hadoop.hive.serde2.typeinfo.TypeInfoFactory.timestampTypeInfo;
import static org.apache.hadoop.hive.serde2.typeinfo.TypeInfoUtils.getTypeInfoFromTypeString;
import static org.apache.hadoop.hive.serde2.typeinfo.TypeInfoUtils.getTypeInfosFromTypeString;

public final class HiveType
{
    public static final HiveType HIVE_BOOLEAN = new HiveType(booleanTypeInfo);
    public static final HiveType HIVE_BYTE = new HiveType(byteTypeInfo);
    public static final HiveType HIVE_SHORT = new HiveType(shortTypeInfo);
    public static final HiveType HIVE_INT = new HiveType(intTypeInfo);
    public static final HiveType HIVE_LONG = new HiveType(longTypeInfo);
    public static final HiveType HIVE_FLOAT = new HiveType(floatTypeInfo);
    public static final HiveType HIVE_DOUBLE = new HiveType(doubleTypeInfo);
    public static final HiveType HIVE_STRING = new HiveType(stringTypeInfo);
    public static final HiveType HIVE_TIMESTAMP = new HiveType(timestampTypeInfo);
    public static final HiveType HIVE_DATE = new HiveType(dateTypeInfo);
    public static final HiveType HIVE_BINARY = new HiveType(binaryTypeInfo);

    private final String hiveTypeName;
    private final TypeInfo typeInfo;

    private HiveType(TypeInfo typeInfo)
    {
        requireNonNull(typeInfo, "typeInfo is null");
        this.hiveTypeName = typeInfo.getTypeName();
        this.typeInfo = typeInfo;
    }

    @JsonValue
    public String getHiveTypeName()
    {
        return hiveTypeName;
    }

    public Category getCategory()
    {
        return typeInfo.getCategory();
    }

    public TypeInfo getTypeInfo()
    {
        return typeInfo;
    }

    public TypeSignature getTypeSignature()
    {
        return getTypeSignature(typeInfo);
    }

    public Type getType(TypeManager typeManager)
    {
        return typeManager.getType(getTypeSignature());
    }

    @Override
    public boolean equals(Object o)
    {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }

        HiveType hiveType = (HiveType) o;

        if (!hiveTypeName.equals(hiveType.hiveTypeName)) {
            return false;
        }

        return true;
    }

    @Override
    public int hashCode()
    {
        return hiveTypeName.hashCode();
    }

    @Override
    public String toString()
    {
        return hiveTypeName;
    }

    public static boolean isSupportedType(TypeInfo typeInfo)
    {
        switch (typeInfo.getCategory()) {
            case PRIMITIVE:
                PrimitiveObjectInspector.PrimitiveCategory primitiveCategory = ((PrimitiveTypeInfo) typeInfo).getPrimitiveCategory();
                return getPrimitiveType(primitiveCategory) != null;
            case MAP:
                MapTypeInfo mapTypeInfo = checkType(typeInfo, MapTypeInfo.class, "typeInfo");
                return isSupportedType(mapTypeInfo.getMapKeyTypeInfo()) && isSupportedType(mapTypeInfo.getMapValueTypeInfo());
            case LIST:
                ListTypeInfo listTypeInfo = checkType(typeInfo, ListTypeInfo.class, "typeInfo");
                return isSupportedType(listTypeInfo.getListElementTypeInfo());
            case STRUCT:
                StructTypeInfo structTypeInfo = checkType(typeInfo, StructTypeInfo.class, "typeInfo");
                return structTypeInfo.getAllStructFieldTypeInfos().stream()
                        .allMatch(HiveType::isSupportedType);
        }
        return false;
    }

    @JsonCreator
    @Nonnull
    public static HiveType valueOf(String hiveTypeName)
    {
        requireNonNull(hiveTypeName, "hiveTypeName is null");
        return toHiveType(getTypeInfoFromTypeString(hiveTypeName));
    }

    @Nonnull
    public static List<HiveType> toHiveTypes(String hiveTypes)
    {
        requireNonNull(hiveTypes, "hiveTypes is null");
        return ImmutableList.copyOf(getTypeInfosFromTypeString(hiveTypes).stream()
                .map(HiveType::toHiveType)
                .collect(toList()));
    }

    @Nonnull
    public static HiveType toHiveType(TypeInfo typeInfo)
    {
        requireNonNull(typeInfo, "typeInfo is null");
        if (!isSupportedType(typeInfo)) {
            throw new PrestoException(NOT_SUPPORTED, format("Unsupported Hive type: %s", typeInfo));
        }
        return new HiveType(typeInfo);
    }

    @Nonnull
    public static HiveType toHiveType(Type type)
    {
        requireNonNull(type, "type is null");
        return new HiveType(toTypeInfo(type));
    }

    @Nonnull
    private static TypeInfo toTypeInfo(Type type)
    {
        if (BOOLEAN.equals(type)) {
            return HIVE_BOOLEAN.typeInfo;
        }
        if (BIGINT.equals(type)) {
            return HIVE_LONG.typeInfo;
        }
        if (DOUBLE.equals(type)) {
            return HIVE_DOUBLE.typeInfo;
        }
        if (VARCHAR.equals(type)) {
            return HIVE_STRING.typeInfo;
        }
        if (VARBINARY.equals(type)) {
            return HIVE_BINARY.typeInfo;
        }
        if (DATE.equals(type)) {
            return HIVE_DATE.typeInfo;
        }
        if (TIMESTAMP.equals(type)) {
            return HIVE_TIMESTAMP.typeInfo;
        }
        if (isArrayType(type)) {
            TypeInfo elementType = toTypeInfo(type.getTypeParameters().get(0));
            return getListTypeInfo(elementType);
        }
        if (isMapType(type)) {
            TypeInfo keyType = toTypeInfo(type.getTypeParameters().get(0));
            TypeInfo valueType = toTypeInfo(type.getTypeParameters().get(1));
            return getMapTypeInfo(keyType, valueType);
        }
        if (isRowType(type)) {
            return getStructTypeInfo(
                    type.getTypeSignature().getLiteralParameters().stream()
                            .map(String.class::cast)
                            .collect(toList()),
                    type.getTypeParameters().stream()
                            .map(HiveType::toTypeInfo)
                            .collect(toList()));
        }
        throw new PrestoException(NOT_SUPPORTED, format("Unsupported Hive type: %s", type));
    }

    @Nonnull
    private static TypeSignature getTypeSignature(TypeInfo typeInfo)
    {
        switch (typeInfo.getCategory()) {
            case PRIMITIVE:
                PrimitiveObjectInspector.PrimitiveCategory primitiveCategory = ((PrimitiveTypeInfo) typeInfo).getPrimitiveCategory();
                Type primitiveType = getPrimitiveType(primitiveCategory);
                if (primitiveType == null) {
                    break;
                }
                return primitiveType.getTypeSignature();
            case MAP:
                MapTypeInfo mapTypeInfo = checkType(typeInfo, MapTypeInfo.class, "fieldInspector");
                TypeSignature keyType = getTypeSignature(mapTypeInfo.getMapKeyTypeInfo());
                TypeSignature valueType = getTypeSignature(mapTypeInfo.getMapValueTypeInfo());
                return new TypeSignature(StandardTypes.MAP, ImmutableList.of(keyType, valueType), ImmutableList.of());
            case LIST:
                ListTypeInfo listTypeInfo = checkType(typeInfo, ListTypeInfo.class, "fieldInspector");
                TypeSignature elementType = getTypeSignature(listTypeInfo.getListElementTypeInfo());
                return new TypeSignature(StandardTypes.ARRAY, ImmutableList.of(elementType), ImmutableList.of());
            case STRUCT:
                StructTypeInfo structTypeInfo = checkType(typeInfo, StructTypeInfo.class, "fieldInspector");
                List<Object> fieldNames = ImmutableList.copyOf(structTypeInfo.getAllStructFieldNames());
                List<TypeSignature> fieldTypes = structTypeInfo.getAllStructFieldTypeInfos()
                        .stream()
                        .map(HiveType::getTypeSignature)
                        .collect(toList());
                return new TypeSignature(StandardTypes.ROW, fieldTypes, fieldNames);
        }
        throw new PrestoException(NOT_SUPPORTED, format("Unsupported Hive type: %s", typeInfo));
    }

    private static Type getPrimitiveType(PrimitiveObjectInspector.PrimitiveCategory primitiveCategory)
    {
        switch (primitiveCategory) {
            case BOOLEAN:
                return BOOLEAN;
            case BYTE:
                return BIGINT;
            case SHORT:
                return BIGINT;
            case INT:
                return BIGINT;
            case LONG:
                return BIGINT;
            case FLOAT:
                return DOUBLE;
            case DOUBLE:
                return DOUBLE;
            case STRING:
                return VARCHAR;
            case DATE:
                return DATE;
            case TIMESTAMP:
                return TIMESTAMP;
            case BINARY:
                return VARBINARY;
            default:
                return null;
        }
    }
}
