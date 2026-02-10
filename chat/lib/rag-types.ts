export interface Source {
  title: string;
  url: string;
  type: string;
  text: string;
}

export interface Recommendation {
  product_name: string;
  confidence_score: number;
  explanation: string;
  price?: number;
  image_url?: string;
  product_link?: string;
  config_id?: number;
  specs?: Record<string, string>;
  brand?: string;
  ranking?: number;
}

export interface StreamData {
  recommendations: Recommendation[];
  sources: Source[];
  processing_time: number;
}
