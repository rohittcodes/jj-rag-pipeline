// /src/db/schema/configUseCaseRelation.ts

// Definition: Config Use Case Relation links configurations with use cases.

import { pgTable, serial, bigint, text, jsonb } from "drizzle-orm/pg-core";
import { configs } from "./configs"; // Importing configs schema
import { useCases } from "./useCases"; // Importing useCases schema
import { timestamps, userMetadata } from "./columns.helpers"; // Reuse timestamps and user metadata

type MatchType = "EXACT" | "PARTIAL";

// Define the "config_use_case_relation" table schema
export const configUseCaseRelation = pgTable("config_use_case_relation", {
  id: serial("id").primaryKey(), // Auto-incremented primary key

  // Foreign key to configs.id
  configId: serial("config_id")
    .references(() => configs.id, { onDelete: "cascade" })
    .notNull(),

  // Foreign key to use_cases.id
  useCaseId: serial("use_case_id")
    .references(() => useCases.id, { onDelete: "cascade" })
    .notNull(),

  matchType: text("match_type").$type<MatchType>(), // EXACT, PARTIAL
  nonMatchedConditions: jsonb("non_matched_conditions").$type<string[]>(),

  // classificationId: text("classification"), // TODO: Enums for classification?

  // Include metadata columns (timestamps, userMetadata)
  ...timestamps,
  ...userMetadata,
});

// Reviewed
// ** Will be added manually in the Admin Panel. No migration required.
