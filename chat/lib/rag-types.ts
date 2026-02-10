export interface Source {
  title: string;
  url: string;
  type: string;
  text: string;
}

export interface AffiliateLink {
  id: number;
  url: string;
  current_price: number;
  msrp?: number;
  out_of_stock: boolean;
  store_name: string;
  store_logo?: string;
}

export interface Recommendation {
  product_name: string;
  confidence_score: number;
  explanation: string;
  price?: number;
  image_url?: string;
  product_link?: string;
  config_id?: number;
  public_config_id?: string;
  specs?: Record<string, string>;
  specs_raw?: Record<string, string>;
  property_groups?: Record<string, Array<{ property: string; value: string }>>;
  affiliate_links?: AffiliateLink[];
  brand?: string;
  ranking?: number;
}

export interface StreamData {
  recommendations: Recommendation[];
  sources: Source[];
  processing_time: number;
}
