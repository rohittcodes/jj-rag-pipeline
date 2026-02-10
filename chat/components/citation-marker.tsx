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
          <span className="cursor-help inline-flex items-center gap-1 bg-amber-500/10 hover:bg-amber-500/20 text-amber-600 dark:text-amber-500 px-1.5 py-0 rounded text-[10px] font-bold border border-amber-500/20 transition-colors mx-0.5 align-middle">
            <Info size={10} /> {id}
          </span>
        </TooltipTrigger>
        <TooltipContent className="w-80 p-4 bg-popover border-border shadow-xl">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] font-bold uppercase tracking-wider text-amber-600 bg-amber-500/10 px-1.5 py-0.5 rounded">
              {type} SOURCE
            </span>
            {url && (
              <a href={url} target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:text-blue-600 transition-colors">
                <ExternalLink size={14} />
              </a>
            )}
          </div>
          <p className="text-xs font-medium italic mb-2">"{snippet}"</p>
          {fullText && (
            <>
              <div className="h-px bg-border my-2" />
              <p className="text-[11px] text-muted-foreground leading-relaxed max-h-32 overflow-y-auto pr-1">
                {fullText}
              </p>
            </>
          )}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};
