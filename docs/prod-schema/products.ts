// /src/db/schema/products.ts

// Definition: Products Table stores the product information

import { boolean, integer, pgTable, serial, text } from "drizzle-orm/pg-core";
import { productTypes } from "./productTypes"; // Importing productTypes schema
import { timestamps, userMetadata } from "./columns.helpers"; // Reuse timestamps and user metadata
import { modelYears } from "./modelYears";
import { brands } from "./brands";

// Define the "products" table schema
export const products = pgTable("products", {
  id: serial("id").primaryKey(), // Primary key
  brand: text("brand").notNull(), // Brand of the product (e.g., "Apple", "Dell") // Backward Compatibility
  brandId: integer("brand_id")
    .references(() => brands.id, {
      onDelete: "restrict",
    })
    .notNull(),
  title: text("title").notNull(), // Product title (e.g., "MacBook Air", "XPS 13")
  slug: text("slug").notNull().unique(), // URL-friendly slug (e.g., "macbook-air", "xps-13")
  description: text("description"), // Optional product description
  seoDescription: text("seo_description"), // SEO description
  image: text("image"), // URL to the product image
  ytReviewVideoId: text("yt_review_video_id"), // YouTube review video ID
  isArchived: boolean("is_archived").default(false), // Archived flag
  year: integer("year").notNull(), // Backward Compatibility
  modelYearId: integer("model_year_id")
    .references(() => modelYears.id, {
      onDelete: "restrict", // Don't allow deleting years with products
    })
    .notNull(),
  productTypeId: serial("product_type_id")
    .references(() => productTypes.id, {
      onDelete: "restrict",
    })
    .notNull(), //   Foreign key to product_types.id

  fsDocId: text("fs_doc_id"), // Firestore document ID

  // Test Data PDF fields
  testDataPdfUrl: text("test_data_pdf_url"), // S3 URL for the test data PDF
  testDataPdfKey: text("test_data_pdf_key"), // S3 key for the test data PDF

  // Include metadata columns (timestamps, userMetadata)
  ...timestamps,
  ...userMetadata,
});

// Reviewed
// Migrated : October 22, 2024
