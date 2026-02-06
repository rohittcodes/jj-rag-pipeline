// src/db/schema/productTypePropertyValueGroups.ts
// [AI] This table stores definitions for groups that can be associated with product type property values.
// [AI] For example, for a property like "RAM Capacity", groups could be "Entry-Level", "Mid-Range", "High-Performance".
// [AI] For a property like "Processor Family", groups could be "Intel Core i5", "AMD Ryzen 7", etc.

import { pgTable, serial, text, integer } from "drizzle-orm/pg-core";
import { productTypeProperties } from "./productTypeProperties"; // [AI] Assuming this is the schema for product type properties
import { timestamps, userMetadata } from "./columns.helpers"; // [AI] Reusing common timestamp and user metadata columns

export const productTypePropertyValueGroups = pgTable(
  "product_type_property_value_groups",
  {
    id: serial("id").primaryKey(), // [AI] Standard auto-incrementing primary key

    productTypePropertyId: integer("product_type_property_id")
      .references(() => productTypeProperties.id, { onDelete: "cascade" }) // [AI] Foreign key to productTypeProperties.id. Links this group to a specific property (e.g., RAM, CPU).
      .notNull(),

    name: text("name").notNull(), // [AI] The name of the group (e.g., "Beginner", "Advanced", "Intel i7 Series").
    description: text("description"), // [AI] Optional fuller description of the group.
    displayOrder: integer("display_order").default(0).notNull(), // [AI] Optional: for sorting/ordering groups in the UI. Default to 0.

    // [AI] Include standard metadata columns
    ...timestamps,
    ...userMetadata,
  }
);

/* AI generated example:
// Example data for productTypePropertyValueGroups:

// Assuming productTypeProperties has an entry for "RAM Capacity" with id = 1
// { productTypePropertyId: 1, name: "Entry-Level RAM", description: "Suitable for basic tasks, up to 8GB.", displayOrder: 1 }
// { productTypePropertyId: 1, name: "Mid-Range RAM", description: "Good for multitasking, 16GB to 32GB.", displayOrder: 2 }
// { productTypePropertyId: 1, name: "High-Performance RAM", description: "For demanding applications, 64GB and above.", displayOrder: 3 }

// Assuming productTypeProperties has an entry for "CPU Series" with id = 2
// { productTypePropertyId: 2, name: "Intel Core i5 Series", displayOrder: 1 }
// { productTypePropertyId: 2, name: "Intel Core i7 Series", displayOrder: 2 }
// { productTypePropertyId: 2, name: "AMD Ryzen 5 Series", displayOrder: 3 }
*/

// Reviewed
// Migrated: [Date of migration]
