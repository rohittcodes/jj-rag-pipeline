// /src/db/schema/productTypeProperties.ts

// Definition: Product Type Properties Table stores the properties for each product type (E.g., "RAM", "Processor")

import { pgTable, serial, text, boolean } from "drizzle-orm/pg-core";
import { productTypes } from "./productTypes"; // Importing productTypes schema
import { productTypePropertyGroups } from "./productTypePropertyGroups"; // Importing productTypePropertyGroups schema
import { timestamps, userMetadata } from "./columns.helpers"; // Reuse timestamps and user metadata

export const productTypeProperties = pgTable("product_type_properties", {
  id: serial("id").primaryKey(), // Auto-incremented primary key
  title: text("title").notNull(), // Name of the property (e.g., "RAM", "Processor")

  productTypeId: serial("product_type_id")
    .references(() => productTypes.id, {
      onDelete: "cascade",
    })
    .notNull(), // Foreign key to product_types.id

  groupId: serial("group_id").references(() => productTypePropertyGroups.id, {
    onDelete: "set null",
  }), // Foreign key to product_type_property_groups.id

  featured: boolean("featured").default(false), // If true, the property will be shown on the product card
  filterable: boolean("filterable").default(false), // If true, the property will be available for filtering
  required: boolean("required").default(false), // If true, the property is required for the product type
  // inputType: text("input_type").notNull(), // Defines the input type for the property (e.g., "text", "dropdown")
  prefix: text("prefix"), // Optional prefix for the property value (e.g., "i7-", "$")
  suffix: text("suffix"), // Optional suffix for the property value (e.g., "GB", " MHz")
  dataType: text("data_type").notNull().default("string"), // Type of the data this property holds, e.g., 'numeric', 'text', 'alphanumeric'

  // Include metadata columns (timestamps, userMetadata)
  ...timestamps,
  ...userMetadata,
});

// Reviewed
// Migrated : October 25, 2024
