// /src/db/schema/configRatings.ts

// Definition: Config Ratings table stores individual ratings for each config-parameter combination.
// Each config can be rated on multiple parameters, with each rating contributing to the final score.

import { pgTable, serial, numeric, unique } from "drizzle-orm/pg-core";
import { configs } from "./configs";
import { ratingParameters } from "./ratingParameters";
import { timestamps, userMetadata } from "./columns.helpers";

// Define the "config_ratings" table schema
export const configRatings = pgTable(
  "config_ratings",
  {
    id: serial("id").primaryKey(), // Auto-incremented primary key

    configId: serial("config_id")
      .references(() => configs.id, {
        onDelete: "cascade",
      })
      .notNull(), // Foreign key to configs.id - cascade delete when config is deleted

    parameterId: serial("parameter_id")
      .references(() => ratingParameters.id, {
        onDelete: "cascade",
      })
      .notNull(), // Foreign key to rating_parameters.id - cascade delete when parameter is deleted

    ratingValue: numeric("rating_value").notNull(), // Rating value between 1-10

    // Include metadata columns (timestamps, userMetadata)
    ...timestamps,
    ...userMetadata,
  },
  (table) => ({
    // Unique constraint: one rating per config per parameter
    uniqueConfigParameter: unique("unique_config_parameter").on(
      table.configId,
      table.parameterId
    ),
  })
);


