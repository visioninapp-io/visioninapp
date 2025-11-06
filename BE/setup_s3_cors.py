"""
Script to configure S3 bucket CORS settings for presigned URL uploads.
Run this once to enable browser-based direct uploads to S3.
"""
import boto3
from app.core.config import settings

def setup_s3_cors():
    """Configure CORS for S3 bucket to allow browser uploads."""

    print(f"[S3 CORS Setup] Configuring bucket: {settings.AWS_BUCKET_NAME}")
    print(f"[S3 CORS Setup] Region: {settings.AWS_REGION}")

    try:
        # Create S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )

        # Define CORS configuration
        cors_configuration = {
            'CORSRules': [
                {
                    'AllowedOrigins': ['*'],  # Allow all origins for development
                    'AllowedMethods': ['GET', 'PUT', 'POST', 'DELETE', 'HEAD'],
                    'AllowedHeaders': ['*'],  # Allow all headers
                    'ExposeHeaders': ['ETag', 'x-amz-request-id', 'x-amz-id-2'],
                    'MaxAgeSeconds': 3600  # Cache preflight for 1 hour
                }
            ]
        }

        # Apply CORS configuration
        s3_client.put_bucket_cors(
            Bucket=settings.AWS_BUCKET_NAME,
            CORSConfiguration=cors_configuration
        )

        print("[S3 CORS Setup] ✓ CORS configuration applied successfully!")
        print("\nCORS Rules:")
        print("  - AllowedOrigins: * (all)")
        print("  - AllowedMethods: GET, PUT, POST, DELETE, HEAD")
        print("  - AllowedHeaders: * (all)")
        print("  - MaxAgeSeconds: 3600 (1 hour)")

        # Verify configuration
        response = s3_client.get_bucket_cors(Bucket=settings.AWS_BUCKET_NAME)
        print("\n[S3 CORS Setup] ✓ Verification successful!")
        print(f"[S3 CORS Setup] Current CORS rules: {len(response['CORSRules'])} rule(s)")

        return True

    except Exception as e:
        print(f"\n[S3 CORS Setup] ✗ Error: {str(e)}")
        print("\nIf you get permission errors, ensure your AWS credentials have s3:PutBucketCORS permission.")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("S3 CORS Configuration Setup")
    print("=" * 60)
    print()

    success = setup_s3_cors()

    print()
    print("=" * 60)
    if success:
        print("Setup completed successfully! ✓")
        print("You can now upload files directly from the browser.")
    else:
        print("Setup failed. Please check the error messages above.")
    print("=" * 60)
