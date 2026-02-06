// /src/db/schema/configTestDataMetrics.ts

// Definition: Config Test Data Metrics stores the metrics associated with different test data categories (e.g., "FPS", "Battery Life").

import { pgTable, serial, text } from "drizzle-orm/pg-core";
import { configTestDataCategories } from "./configTestDataCategories"; // Importing configTestDataCategories schema
import { timestamps, userMetadata } from "./columns.helpers"; // Reuse timestamps and user metadata

// Define the "config_test_data_metrics" table schema
export const configTestDataMetrics = pgTable("config_test_data_metrics", {
  id: serial("id").primaryKey(), // Auto-incremented primary key

  // Foreign key to config_test_data_categories.id
  categoryId: serial("category_id")
    .references(() => configTestDataCategories.id, {
      onDelete: "cascade",
    })
    .notNull(),

  metricName: text("metric_name").notNull(), // Name of the metric (e.g., "FPS", "Battery Life")

  // Include metadata columns (timestamps, userMetadata)
  ...timestamps,
  ...userMetadata,
});

// Reviewed
// Migrated : October 25, 2024
