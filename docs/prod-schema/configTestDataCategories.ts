// /src/db/schema/configTestDataCategories.ts

// Definition: Config Test Data Categories stores the categories for different test data (e.g., "Settings", "Battery", "Performance").

import { pgTable, serial, text } from "drizzle-orm/pg-core";
import { timestamps, userMetadata } from "./columns.helpers"; // Reuse timestamps and user metadata

// Define the "config_test_data_categories" table schema
export const configTestDataCategories = pgTable("config_test_data_categories", {
  id: serial("id").primaryKey(), // Auto-incremented primary key

  name: text("name").notNull().unique(), // Name of the category (e.g., "Settings", "Battery", "Performance")

  // Include metadata columns (timestamps, userMetadata)
  ...timestamps,
  ...userMetadata,
});

// Reviewed
// Migrated : October 25, 2024
