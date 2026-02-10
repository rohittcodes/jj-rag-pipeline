"use client";

import React from "react";
import { RiShoppingCartLine } from "react-icons/ri";
import { Recommendation } from "@/lib/rag-types";
import Image from "next/image";
import { Montserrat } from "next/font/google";

const montserrat = Montserrat({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-montserrat",
});

interface ProductCardProps {
  recommendation: Recommendation;
}

export const ProductCard: React.FC<ProductCardProps> = ({ recommendation }) => {
  const formatPrice = (price: number): string => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(price);
  };

  const getSpecsText = () => {
    if (recommendation.property_groups) {
      const featured: string[] = [];
      const groups = ["Performance", "Display", "Memory", "Storage"];
      for (const groupName of groups) {
        const props = recommendation.property_groups[groupName];
        if (props) {
          for (const prop of props) {
            const name = prop.property.toLowerCase();
            const value = prop.value;
            if (name.includes("processor") || name.includes("ram") || name.includes("gpu") || name.includes("screen size")) {
              featured.push(value);
            }
          }
        }
      }
      if (featured.length > 0) return featured.slice(0, 4).join(" | ");
    }
    if (recommendation.specs_raw && Object.keys(recommendation.specs_raw).length > 0) {
      return Object.values(recommendation.specs_raw).slice(0, 4).join(" | ");
    }
    return null;
  };

  const specsText = getSpecsText();
  const goUrl = recommendation.public_config_id 
    ? `https://go.bestlaptop.deals/c/${recommendation.public_config_id}`
    : recommendation.product_link;

  return (
    <div className={`flex flex-col md:flex-row items-center w-full border border-[#E5E5E5] rounded-xl overflow-hidden bg-white hover:shadow-md transition-all p-4 gap-6 ${montserrat.variable} font-sans`}>
      {/* Left: Product Image */}
      <div className="w-full md:w-[180px] h-[130px] flex-shrink-0 flex items-center justify-center bg-white rounded-lg">
        <Image
          src={recommendation.image_url || "/placeholder.png"}
          alt={recommendation.product_name}
          width={160}
          height={110}
          className="w-full h-full object-contain"
          unoptimized
        />
      </div>

      {/* Center: Product Details */}
      <div className="flex flex-col flex-grow gap-2 min-w-0">
        <div className="flex items-center gap-3 flex-wrap">
          <h3 className="text-[#0A0A0A] text-lg font-bold leading-tight truncate">
            {recommendation.product_name}
          </h3>
          {recommendation.ranking && (
            <span className="text-[10px] text-amber-600 font-black bg-amber-50 px-2 py-0.5 rounded border border-amber-100 uppercase tracking-widest">
              Josh's #{recommendation.ranking} Pick
            </span>
          )}
        </div>

        {specsText && (
          <p className="text-[#262626] text-sm font-semibold leading-snug">
            {specsText}
          </p>
        )}

        {recommendation.explanation && (
          <p className="text-[#737373] text-sm italic leading-relaxed line-clamp-2 pl-3 border-l-2 border-gray-100">
            "{recommendation.explanation}"
          </p>
        )}
      </div>

      {/* Right: Price + Single CTA */}
      <div className="flex flex-col items-center md:items-end justify-center gap-3 flex-shrink-0 min-w-[140px]">
        {recommendation.price && (
          <div className="flex flex-col items-center md:items-end">
            <span className="text-gray-400 text-[10px] font-bold uppercase tracking-widest leading-none">Starting at</span>
            <span className="text-[#1F1F1F] font-bold text-2xl leading-tight">
              {formatPrice(Math.floor(recommendation.price))}
            </span>
          </div>
        )}

        <a
          href={goUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="w-full md:w-auto flex cursor-pointer items-center justify-center transition-colors font-bold text-sm h-[40px] px-6 gap-2 rounded-lg bg-[#231F20] hover:bg-[#3a3637] text-[#FAFAFA] shadow-sm whitespace-nowrap"
        >
          <RiShoppingCartLine className="w-4 h-4" />
          <span>View Configuration</span>
        </a>
      </div>
    </div>
  );
};
