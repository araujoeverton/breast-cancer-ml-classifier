import boto3
import json
import os

# Configuration
ENDPOINT_NAME = os.environ.get('ENDPOINT_NAME', 'cbis-ddsm-serverless-endpoint')
s3_client = boto3.client('s3')
sm_runtime = boto3.client('sagemaker-runtime')


def lambda_handler(event, context):
    print("Receiving event from S3...")

    # Read event details
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']

        print(f"Processing file: s3://{bucket}/{key}")

        # Download image from S3 to Lambda memory
        file_obj = s3_client.get_object(Bucket=bucket, Key=key)
        file_content = file_obj['Body'].read()

        # Send to SageMaker Serverless Endpoint
        print(f"Invoking endpoint: {ENDPOINT_NAME}")
        response = sm_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/x-image',
            Body=file_content
        )

        # Read the response
        result = json.loads(response['Body'].read().decode())
        prob_benign = result[0]
        prob_malignant = result[1]

        diagnosis = "MALIGNANT" if prob_malignant > 0.5 else "BENIGN"
        confidence = prob_malignant if diagnosis == "MALIGNANT" else prob_benign

        print(f"✅ Result for {key}: {diagnosis} ({confidence * 100:.2f}%)")

        # (Optional) Here you could save the result to DynamoDB or move the file

    return {
        'statusCode': 200,
        'body': json.dumps(f"Processing complete. Diagnosis: {diagnosis}")
    }