// /src/db/schema/configTags.ts

// Definition: Config Tags links tags to specific configurations and determines if they should be shown on all lists or specific ones.

import { pgTable, serial, boolean, integer } from "drizzle-orm/pg-core";
import { configs } from "./configs"; // Importing configs schema
import { tags } from "./tags"; // Importing tags schema
import { lists } from "./lists"; // Importing lists schema
import { timestamps, userMetadata } from "./columns.helpers"; // Reuse timestamps and user metadata

// Define the "config_tags" table schema
export const configTags = pgTable("config_tags", {
  id: serial("id").primaryKey(), // Auto-incremented primary key

  configId: serial("config_id")
    .references(() => configs.id, {
      onDelete: "cascade",
    })
    .notNull(), // Foreign key to configs.id

  tagId: serial("tag_id")
    .references(() => tags.id, {
      onDelete: "cascade",
    })
    .notNull(), // Foreign key to tags.id

  showOnAllLists: boolean("show_on_all_lists").default(false), // Determines whether the tag should be shown on all lists

  listId: integer("list_id").references(() => lists.id, {
    onDelete: "cascade",
  }),

  // listId: serial("list_id").references(() => lists.id, {
  //   onDelete: "cascade",
  // }), // Foreign key to lists.id, only if tag applies to a specific list

  // Include metadata columns (timestamps, userMetadata)
  ...timestamps,
  ...userMetadata,
});

// Reviewed
