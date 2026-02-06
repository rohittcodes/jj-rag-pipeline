// /src/db/schema/productTypePropertyGroups.ts

// Definition: Product Type Property Groups Table stores the property groups for each product type (E.g., "Performance", "Display")

import { pgTable, serial, text } from "drizzle-orm/pg-core";
import { productTypes } from "./productTypes"; // Importing productTypes schema
import { timestamps, userMetadata } from "./columns.helpers"; // Reuse timestamps and user metadata

// Define the "product_type_property_groups" table schema
export const productTypePropertyGroups = pgTable(
  "product_type_property_groups",
  {
    id: serial("id").primaryKey(), // Auto-incremented primary key
    name: text("name").notNull(), // Name of the property group (e.g., "Performance", "Display")

    productTypeId: serial("product_type_id")
      .references(() => productTypes.id, {
        onDelete: "cascade",
      })
      .notNull(), // Foreign key to product_types.id

    // Include metadata columns (timestamps, userMetadata)
    ...timestamps,
    ...userMetadata,
  }
);

// Reviewed
// Migrated : October 25, 2024
