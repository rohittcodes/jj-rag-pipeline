// /src/db/schema/configProperties.ts

// Definition: Config Properties table stores the properties assigned to a specific configuration.

import { pgTable, serial } from "drizzle-orm/pg-core";
import { configs } from "./configs"; // Importing configs schema
import { productTypePropertyValues } from "./productTypePropertyValues"; // Importing configPropertyValues schema
import { timestamps, userMetadata } from "./columns.helpers"; // Reuse timestamps and user metadata

// Define the "config_properties" table schema
export const configProperties = pgTable("config_properties", {
  id: serial("id").primaryKey(), // Auto-incremented primary key

  configId: serial("config_id")
    .references(() => configs.id, {
      onDelete: "cascade",
    })
    .notNull(), // Foreign key to configs.id

  valueId: serial("value_id")
    .references(() => productTypePropertyValues.id, {
      onDelete: "cascade",
    })
    .notNull(), // Foreign key to config_property_values.id

  // Include metadata columns (timestamps, userMetadata)
  ...timestamps,
  ...userMetadata,
});

// Reviewed
// Migrated : October 27, 2024
