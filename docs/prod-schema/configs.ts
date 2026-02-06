// /src/db/schema/configs.ts

// Definition: Configs Table stores the configuration information for a product

import {
  pgTable,
  serial,
  text,
  smallint,
  boolean,
  pgEnum,
  timestamp,
  jsonb,
  integer,
  numeric,
} from "drizzle-orm/pg-core";
import { products } from "./products"; // Importing products schema
import { timestamps, userMetadata } from "./columns.helpers"; // Reuse timestamps and user metadata

// Define the enum for testing_status
// This is defined manually in the Database using SQL
// CREATE TYPE testing_status AS ENUM ('NOT_TESTED', 'IN_TESTING', 'MODEL_TESTED');
const testingStatusEnum = pgEnum("testing_status", [
  "NOT_TESTED",
  "IN_TESTING",
  "MODEL_TESTED",
]);

// Define the "configs" table schema
export const configs = pgTable("configs", {
  id: serial("id").primaryKey(), // Auto-incremented primary key
  publicConfigId: text("public_config_id").unique().notNull(), // Public configuration ID for Smart Links (auto-generated, read-only)
  productId: serial("product_id")
    .references(() => products.id, {
      onDelete: "cascade",
    })
    .notNull(), // Foreign key to products.id - cascade delete when product is deleted

  classification: text("classification"), // Defines how the config is classified (e.g., 'First Pick')
  customClassification: jsonb("custom_classification"), // Custom classification stored as JSON
  fallbackAffiliateLink: serial("fallback_affiliate_link"), // Stores the fallback affiliate link id
  testersNotes: text("testers_notes"), // Internal notes for testers
  embargoDate: timestamp("embargo_date"), // Embargo date
  image: text("image"), // Image URL
  isArchived: boolean("is_archived").default(false), // Archived flag
  isForceCrazyDeal: boolean("is_force_crazy_deal").default(false), // Crazy deal flag
  modelYear: smallint("model_year"), // Year of the product model // DEPRECATED
  testingStatus: testingStatusEnum("testing_status")
    .notNull()
    .default("NOT_TESTED"), // Testing status enum
  sku: text("sku"), // SKU of the config
  upc: text("upc"), // UPC code of the config
  fsDocId: text("fs_doc_id").unique(), // Firestore document ID
  activeAffLinkDocId: text("active_aff_link_doc_id"), // Firestore document ID for active affiliate link

  // Fallback MSRP when affiliate links don't have pricing
  fallbackMsrp: numeric("fallback_msrp"), // Fallback MSRP when affiliate links don't have pricing

  // Final rating calculated from config ratings
  finalRating: numeric("final_rating"), // Calculated final rating (0-10 scale)

  // Include metadata columns (timestamps, userMetadata)
  ...timestamps,
  ...userMetadata,
});

// Reviewed
// Migrated : October 27, 2024
