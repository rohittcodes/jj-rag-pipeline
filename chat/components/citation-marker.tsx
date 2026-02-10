"use client";

import React, { useState } from "react";
import { Info, ExternalLink } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface CitationMarkerProps {
  id: string;
  type: string;
  snippet: string;
  fullText?: string;
  url?: string;
}

export const CitationMarker: React.FC<CitationMarkerProps> = ({ id, type, snippet, fullText, url }) => {
  return (
    <TooltipProvider>
      <Tooltip delayDuration={0}>
        <TooltipTrigger asChild>
          <span className="cursor-help inline-flex items-center gap-1 bg-gray-100 hover:bg-gray-200 text-gray-900 px-1.5 py-0 rounded text-[10px] font-bold border border-gray-200 transition-colors mx-0.5 align-middle">
            <Info size={10} className="text-gray-500" /> {id}
          </span>
        </TooltipTrigger>
        <TooltipContent className="w-80 p-4 bg-white border border-gray-200 shadow-xl text-gray-900">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] font-bold uppercase tracking-wider text-gray-600 bg-gray-100 px-1.5 py-0.5 rounded border border-gray-200">
              {type} SOURCE
            </span>
            {url && (
              <a href={url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-700 transition-colors">
                <ExternalLink size={14} />
              </a>
            )}
          </div>
          <p className="text-xs font-semibold italic mb-2 text-gray-900">"{snippet}"</p>
          {fullText && (
            <>
              <div className="h-px bg-gray-100 my-2" />
              <p className="text-[11px] text-gray-600 leading-relaxed max-h-32 overflow-y-auto pr-1">
                {fullText}
              </p>
            </>
          )}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};
