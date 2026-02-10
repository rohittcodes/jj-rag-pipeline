export const LOCATION_ID_TO_COUNTRY_NAME: Record<number, string> = {
  1: 'United States',
  2: 'Canada',
  3: 'Australia',
  4: 'India',
  5: 'United Kingdom',
};

export const getCountryCode = (locationId: number): string => {
  const mapping: Record<number, string> = {
    1: 'US',
    2: 'CA',
    3: 'AU',
    4: 'IN',
    5: 'GB',
  };
  return mapping[locationId] || 'US';
};

export const getCurrencyInfo = (locationId: number): { code: string; symbol: string } => {
  const mapping: Record<number, { code: string; symbol: string }> = {
    1: { code: 'USD', symbol: '$' },
    2: { code: 'CAD', symbol: 'C$' },
    3: { code: 'AUD', symbol: 'A$' },
    4: { code: 'INR', symbol: '₹' },
    5: { code: 'GBP', symbol: '£' },
  };
  return mapping[locationId] || { code: 'USD', symbol: '$' };
};
