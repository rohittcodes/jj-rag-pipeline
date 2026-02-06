// /src/db/schema/productTypes.ts

// Definition: Product Types Table stores the product type information (e.g., Laptops, Smartphones)

import { pgTable, serial, text } from "drizzle-orm/pg-core";
import { timestamps, userMetadata } from "./columns.helpers"; // Reuse timestamps and user metadata

// Define the "product_types" table schema
export const productTypes = pgTable("product_types", {
  id: serial("id").primaryKey(), // Auto-incremented primary key
  title: text("title").notNull().unique(), // Name of the product type (e.g., "Laptops", "Smartphones")
  fsDocId: text("fs_doc_id"), // Firestore Document ID

  // Include metadata columns (timestamps, userMetadata)
  ...timestamps,
  ...userMetadata,
});

// Reviewed
// Migrated : October 22, 2024
