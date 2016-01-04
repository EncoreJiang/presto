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
package com.facebook.presto.type;

import com.facebook.presto.spi.type.Type;
import com.facebook.presto.spi.type.TypeSignature;
import org.testng.annotations.Test;

import java.util.Optional;

import static com.facebook.presto.spi.type.BigintType.BIGINT;
import static com.facebook.presto.spi.type.DateType.DATE;
import static com.facebook.presto.spi.type.DoubleType.DOUBLE;
import static com.facebook.presto.spi.type.TimeType.TIME;
import static com.facebook.presto.spi.type.TimeWithTimeZoneType.TIME_WITH_TIME_ZONE;
import static com.facebook.presto.spi.type.TimestampType.TIMESTAMP;
import static com.facebook.presto.spi.type.TimestampWithTimeZoneType.TIMESTAMP_WITH_TIME_ZONE;
import static com.facebook.presto.spi.type.TypeSignature.parseTypeSignature;
import static com.facebook.presto.spi.type.VarbinaryType.VARBINARY;
import static com.facebook.presto.spi.type.VarcharType.VARCHAR;
import static com.facebook.presto.type.JsonPathType.JSON_PATH;
import static com.facebook.presto.type.LikePatternType.LIKE_PATTERN;
import static com.facebook.presto.type.RegexpType.REGEXP;
import static com.facebook.presto.type.TypeRegistry.getCommonSuperTypeSignature;
import static com.facebook.presto.type.UnknownType.UNKNOWN;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertTrue;

public class TestTypeRegistry
{
    @Test
    public void testCanCoerce()
    {
        assertTrue(TypeRegistry.canCoerce(BIGINT, BIGINT));
        assertTrue(TypeRegistry.canCoerce(UNKNOWN, BIGINT));
        assertFalse(TypeRegistry.canCoerce(BIGINT, UNKNOWN));

        assertTrue(TypeRegistry.canCoerce(BIGINT, DOUBLE));
        assertTrue(TypeRegistry.canCoerce(DATE, TIMESTAMP));
        assertTrue(TypeRegistry.canCoerce(DATE, TIMESTAMP_WITH_TIME_ZONE));
        assertTrue(TypeRegistry.canCoerce(TIME, TIME_WITH_TIME_ZONE));
        assertTrue(TypeRegistry.canCoerce(TIMESTAMP, TIMESTAMP_WITH_TIME_ZONE));
        assertTrue(TypeRegistry.canCoerce(VARCHAR, REGEXP));
        assertTrue(TypeRegistry.canCoerce(VARCHAR, LIKE_PATTERN));
        assertTrue(TypeRegistry.canCoerce(VARCHAR, JSON_PATH));

        assertFalse(TypeRegistry.canCoerce(DOUBLE, BIGINT));
        assertFalse(TypeRegistry.canCoerce(TIMESTAMP, TIME_WITH_TIME_ZONE));
        assertFalse(TypeRegistry.canCoerce(TIMESTAMP_WITH_TIME_ZONE, TIMESTAMP));
        assertFalse(TypeRegistry.canCoerce(VARBINARY, VARCHAR));

        assertTrue(TypeRegistry.canCoerce(UNKNOWN.getTypeSignature(), parseTypeSignature("array<bigint>")));
        assertFalse(TypeRegistry.canCoerce(parseTypeSignature("array<bigint>"), UNKNOWN.getTypeSignature()));
        assertTrue(TypeRegistry.canCoerce(parseTypeSignature("array<bigint>"), parseTypeSignature("array<double>")));
        assertFalse(TypeRegistry.canCoerce(parseTypeSignature("array<double>"), parseTypeSignature("array<bigint>")));
        assertTrue(TypeRegistry.canCoerce(parseTypeSignature("map<bigint,double>"), parseTypeSignature("map<bigint,double>")));
        assertFalse(TypeRegistry.canCoerce(parseTypeSignature("map<bigint,double>"), parseTypeSignature("map<double,double>"))); // map covariant cast is not supported yet
        assertTrue(TypeRegistry.canCoerce(parseTypeSignature("row<bigint,double,varchar>"), parseTypeSignature("row<bigint,double,varchar>")));
        assertFalse(TypeRegistry.canCoerce(parseTypeSignature("row<bigint,double,varchar>('a','b','c')"), parseTypeSignature("row<bigint,double,varchar>")));
        assertFalse(TypeRegistry.canCoerce(parseTypeSignature("row<bigint,double,varchar>"), parseTypeSignature("row<bigint,double,varchar>('a','b','c')")));
        assertTrue(TypeRegistry.canCoerce(parseTypeSignature("row<bigint,double,varchar>('a','b','c')"), parseTypeSignature("row<bigint,double,varchar>('a','b','c')")));
    }

    @Test
    public void testGetCommonSuperType()
    {
        assertCommonSuperType(UNKNOWN, UNKNOWN, UNKNOWN);
        assertCommonSuperType(BIGINT, BIGINT, BIGINT);
        assertCommonSuperType(UNKNOWN, BIGINT, BIGINT);

        assertCommonSuperType(BIGINT, DOUBLE, DOUBLE);
        assertCommonSuperType(DATE, TIMESTAMP, TIMESTAMP);
        assertCommonSuperType(DATE, TIMESTAMP_WITH_TIME_ZONE, TIMESTAMP_WITH_TIME_ZONE);
        assertCommonSuperType(TIME, TIME_WITH_TIME_ZONE, TIME_WITH_TIME_ZONE);
        assertCommonSuperType(TIMESTAMP, TIMESTAMP_WITH_TIME_ZONE, TIMESTAMP_WITH_TIME_ZONE);
        assertCommonSuperType(VARCHAR, REGEXP, REGEXP);
        assertCommonSuperType(VARCHAR, LIKE_PATTERN, LIKE_PATTERN);
        assertCommonSuperType(VARCHAR, JSON_PATH, JSON_PATH);

        assertCommonSuperType(TIMESTAMP, TIME_WITH_TIME_ZONE, null);
        assertCommonSuperType(VARBINARY, VARCHAR, null);

        assertCommonSuperType("unknown", "array<bigint>", "array<bigint>");
        assertCommonSuperType("array<bigint>", "array<double>", "array<double>");
        assertCommonSuperType("array<bigint>", "array<unknown>", "array<bigint>");
        assertCommonSuperType("map<bigint,double>", "map<bigint,double>", "map<bigint,double>");
        assertCommonSuperType("map<bigint,double>", "map<double,double>", null); // map covariant cast is not supported yet
        assertCommonSuperType("row<bigint,double,varchar>", "row<bigint,double,varchar>", "row<bigint,double,varchar>");
        assertCommonSuperType("row<bigint,double,varchar>('a','b','c')", "row<bigint,double,varchar>", null);
        assertCommonSuperType("row<bigint,double,varchar>('a','b','c')", "row<bigint,double,varchar>('a','b','c')", "row<bigint,double,varchar>('a','b','c')");
    }

    private void assertCommonSuperType(Type firstType, Type secondType, Type expected)
    {
        TypeRegistry typeManager = new TypeRegistry();
        assertEquals(typeManager.getCommonSuperType(firstType, secondType), Optional.ofNullable(expected));
        assertEquals(typeManager.getCommonSuperType(secondType, firstType), Optional.ofNullable(expected));
    }

    private void assertCommonSuperType(String firstType, String secondType, String expected)
    {
        TypeSignature expectedType = expected == null ? null : parseTypeSignature(expected);
        assertEquals(getCommonSuperTypeSignature(parseTypeSignature(firstType), parseTypeSignature(secondType)), Optional.ofNullable(expectedType));
        assertEquals(getCommonSuperTypeSignature(parseTypeSignature(secondType), parseTypeSignature(firstType)), Optional.ofNullable(expectedType));
    }
}
