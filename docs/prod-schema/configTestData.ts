// /src/db/schema/configTestData.ts

// Definition: Config Test Data stores the actual test data for configurations, referencing metrics and categories.

import { pgTable, serial, text, numeric, integer } from "drizzle-orm/pg-core";
import { configs } from "./configs"; // Importing configs schema
import { configTestDataMetrics } from "./configTestDataMetrics"; // Importing configTestDataMetrics schema
import { timestamps, userMetadata } from "./columns.helpers"; // Reuse timestamps and user metadata

// Define the "config_test_data" table schema
export const configTestData = pgTable("config_test_data", {
    id: serial("id").primaryKey(), // Auto-incremented primary key

    configId: serial("config_id")
        .references(() => configs.id, {
            onDelete: "cascade",
        })
        .notNull(), // Foreign key to configs.id

    performanceMode: text("performance_mode").notNull(), // Specifies the performance mode (e.g., 'Out of the box', 'High')
    // TODO: Enums on Frontend for performanceMode

    testRun: integer("test_run").notNull(), // Represents the test run number (e.g., 1st test, 2nd test)

    metricId: serial("metric_id")
        .references(() => configTestDataMetrics.id, {
            onDelete: "cascade",
        })
        .notNull(), // Foreign key to config_test_data_metrics.id

    metricValue: text("metric_value").notNull(), // Value of the metric (e.g., 60 FPS, 10 hours battery)

    // Include metadata columns (timestamps, userMetadata)
    ...timestamps,
    ...userMetadata,
});

// Reviewed
// ** New Table, no migration needed
