"""
S3 Client for downloading test data PDFs.
"""
import boto3
import os
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class S3Client:
    """Client for interacting with AWS S3 to download test data files."""
    
    def __init__(self):
        """Initialize S3 client with credentials from environment."""
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        self.bucket = os.getenv('AWS_S3_BUCKET_NAME', 'jj-product-test-data')
    
    def list_test_data_files(self, prefix: str = 'test-data-pdfs/') -> List[str]:
        """
        List all test data PDFs in S3.
        
        Args:
            prefix: S3 prefix to filter files (default: 'test-data-pdfs/')
        
        Returns:
            List of S3 keys
        """
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                return []
            
            return [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.pdf')]
        
        except Exception as e:
            print(f"[!] Error listing S3 files: {e}")
            return []
    
    def download_file(self, s3_key: str, local_path: Path) -> bool:
        """
        Download a file from S3 to local path.
        
        Args:
            s3_key: S3 object key
            local_path: Local file path to save to
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure parent directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download file
            self.s3.download_file(self.bucket, s3_key, str(local_path))
            return True
        
        except Exception as e:
            print(f"[!] Error downloading {s3_key}: {e}")
            return False
    
    def file_exists(self, s3_key: str) -> bool:
        """
        Check if a file exists in S3.
        
        Args:
            s3_key: S3 object key
        
        Returns:
            True if file exists, False otherwise
        """
        try:
            self.s3.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except:
            return False
    
    def get_file_metadata(self, s3_key: str) -> Optional[dict]:
        """
        Get metadata for an S3 file.
        
        Args:
            s3_key: S3 object key
        
        Returns:
            Dict with metadata (size, last_modified, etc.) or None if error
        """
        try:
            response = self.s3.head_object(Bucket=self.bucket, Key=s3_key)
            return {
                'size': response['ContentLength'],
                'last_modified': response['LastModified'],
                'content_type': response.get('ContentType', 'unknown')
            }
        except Exception as e:
            print(f"[!] Error getting metadata for {s3_key}: {e}")
            return None


# Test the S3 client
if __name__ == '__main__':
    client = S3Client()
    
    print(f"[*] Testing S3 Client")
    print(f"    Bucket: {client.bucket}")
    
    # List files
    print(f"\n[*] Listing test data files...")
    files = client.list_test_data_files()
    print(f"[+] Found {len(files)} PDF files")
    
    if files:
        print(f"\n[*] First 5 files:")
        for f in files[:5]:
            print(f"    - {f}")
        
        # Test download
        test_file = files[0]
        print(f"\n[*] Testing download of: {test_file}")
        local_path = Path('tmp/s3_test.pdf')
        
        if client.download_file(test_file, local_path):
            print(f"[+] Downloaded successfully to: {local_path}")
            print(f"[+] File size: {local_path.stat().st_size} bytes")
            
            # Get metadata
            metadata = client.get_file_metadata(test_file)
            if metadata:
                print(f"[+] S3 Metadata:")
                print(f"    Size: {metadata['size']} bytes")
                print(f"    Last Modified: {metadata['last_modified']}")
                print(f"    Content Type: {metadata['content_type']}")
        else:
            print(f"[-] Download failed")
    else:
        print(f"[-] No files found")
