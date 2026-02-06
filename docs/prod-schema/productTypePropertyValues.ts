// /src/db/schema/productTypePropertyValues.ts

// Definition: Product Type Property Values stores the actual values for properties defined for each product (e.g., "16GB" for RAM).

import { integer, pgTable, serial, text } from "drizzle-orm/pg-core";
import { productTypeProperties } from "./productTypeProperties"; // Importing productTypeProperties schema
import { timestamps, userMetadata } from "./columns.helpers"; // Reuse timestamps and user metadata
import { productTypePropertyValueGroups } from "./productTypePropertyValueGroups";
// Define the "product_type_property_values" table schema
export const productTypePropertyValues = pgTable(
  "product_type_property_values",
  {
    id: serial("id").primaryKey(), // Auto-incremented primary key
    productTypePropertyId: serial("product_type_property_id")
      .references(() => productTypeProperties.id, {
        onDelete: "cascade",
      })
      .notNull(), // Foreign key to product_type_properties.id
    value: text("value").notNull(), // Actual value for the property (e.g., "16GB", "512GB")
    productTypePropertyValueGroupId: integer(
      "product_type_property_value_group_id"
    ).references(() => productTypePropertyValueGroups.id, {
      onDelete: "set null",
    }), // Foreign key to product_type_property_value_groups.id

    // Include metadata columns (timestamps, userMetadata)
    ...timestamps,
    ...userMetadata,
  }
);

// Reviewed
// Migrated : October 27, 2024
