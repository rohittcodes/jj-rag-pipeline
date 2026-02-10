"use client";

import React from "react";
import { ExternalLink } from "lucide-react";
import { Recommendation } from "@/lib/rag-types";
import Image from "next/image";

interface ProductCardProps {
  recommendation: Recommendation;
}

export const ProductCard: React.FC<ProductCardProps> = ({ recommendation }) => {
  // Format specs into a readable string (first 3-4 key specs)
  const formatSpecs = () => {
    if (!recommendation.specs) return null;

    const specEntries = Object.entries(recommendation.specs);
    if (specEntries.length === 0) return null;

    const importantSpecs = specEntries.slice(0, 4);
    return importantSpecs.map(([_, value]) => value).join(" | ");
  };

  const specsText = formatSpecs();

  return (
    <div className="flex flex-col items-center w-full min-w-[220px] max-w-[220px] flex-shrink-0 border border-[#E5E5E5] rounded-lg p-3 gap-3 bg-white hover:shadow-md transition-shadow">
      {/* Product Image */}
      {recommendation.image_url && (
        <div className="w-full max-w-[140px] h-[100px] flex items-center justify-center">
          <Image
            src={recommendation.image_url}
            alt={recommendation.product_name}
            width={140}
            height={100}
            className="w-full h-full object-contain"
            unoptimized
          />
        </div>
      )}

      {/* Product Info */}
      <div className="flex flex-col items-center gap-1.5 w-full">
        {/* Product Name */}
        <h3 className="text-center text-[#0A0A0A] text-sm font-bold leading-5">
          {recommendation.product_name}
        </h3>

        {/* Specs */}
        {specsText && (
          <p className="text-center text-[#262626] text-[11px] font-normal leading-4 line-clamp-2">
            {specsText}
          </p>
        )}

        {/* Explanation/Recommendation */}
        {recommendation.explanation && (
          <p className="text-center text-[#737373] text-[10px] italic leading-4 line-clamp-2">
            "{recommendation.explanation}"
          </p>
        )}

        {/* Price */}
        {recommendation.price && (
          <div className="text-[#1F1F1F] font-bold text-base leading-5">
            ${Math.floor(recommendation.price)}
          </div>
        )}
      </div>

      {/* CTA Button */}
      {/* {recommendation.product_link && (
        <a
          href={recommendation.product_link}
          target="_blank"
          rel="noopener noreferrer"
          className="w-full bg-[#231F20] hover:bg-[#3a3637] text-[#FAFAFA] flex items-center justify-center transition-colors font-semibold text-xs leading-4 h-[28px] py-1.5 px-3 gap-1.5 rounded"
        >
          <span>EXPLORE GEAR</span>
          <ExternalLink className="w-3 h-3" />
        </a>
      )} */}

      {recommendation.ranking && (
        <div className="text-[10px] text-[#737373] font-medium">
          Ranked #{recommendation.ranking} by Josh
        </div>
      )}
    </div>
  );
};
