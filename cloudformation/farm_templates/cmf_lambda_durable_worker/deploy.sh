#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Package the Lambda source and deploy the durable worker stack.
#
# Two passes, because CloudFormation cannot upload local source itself and the worker
# function needs its real code before a version is published.
#
# Usage:
#   ./deploy.sh --farm-id farm-xxxx [--stack-name NAME] [--region REGION]
#               [--max-workers N] [--model-id MODEL]

set -euo pipefail

STACK_NAME="deadline-durable-lambda-worker"
REGION="${AWS_REGION:-us-west-2}"
MAX_WORKERS="5"
FARM_ID=""
MODEL_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --farm-id) FARM_ID="$2"; shift 2 ;;
    --stack-name) STACK_NAME="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --max-workers) MAX_WORKERS="$2"; shift 2 ;;
    # Convenience only. The stack names no model: a job template does, per task.
    --model-id) MODEL_ID="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$FARM_ID" ]]; then
  echo "error: --farm-id is required" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ARTIFACT_BUCKET="deadline-durable-worker-artifacts-${ACCOUNT_ID}-${REGION}"

if [[ -n "$MODEL_ID" ]]; then
  echo "==> Checking whether ${MODEL_ID} is available in ${REGION}"
  if ! aws bedrock list-foundation-models --region "$REGION" \
        --query "modelSummaries[?modelId=='${MODEL_ID}'].modelId" --output text | grep -q .; then
    echo "warning: model ${MODEL_ID} is not available in ${REGION}. Jobs naming it in a" >&2
    echo "         bedrock-async Request will fail. List async-capable candidates with:" >&2
    echo "         aws bedrock list-foundation-models --region ${REGION} \\" >&2
    echo "           --query \"modelSummaries[?contains(outputModalities,'VIDEO')].modelId\"" >&2
  fi
fi

echo "==> Building the deployment package"
cp "$SCRIPT_DIR"/lambda/*.py "$BUILD_DIR/"
cp -R "$SCRIPT_DIR"/lambda/providers "$BUILD_DIR/"
find "$BUILD_DIR/providers" -name '__pycache__' -type d -prune -exec rm -rf {} +
# Bundle the durable execution SDK rather than relying on the copy in the runtime, so a
# runtime update cannot change the behavior of in-flight executions.
python3 -m pip install \
  --quiet --target "$BUILD_DIR" \
  --only-binary :all: --platform manylinux2014_x86_64 \
  --python-version 3.14 --implementation cp \
  'aws-durable-execution-sdk-python<2'
(cd "$BUILD_DIR" && zip -qr lambda.zip . -x 'lambda.zip')

echo "==> Staging the package in s3://${ARTIFACT_BUCKET}"
if ! aws s3api head-bucket --bucket "$ARTIFACT_BUCKET" --region "$REGION" 2>/dev/null; then
  aws s3 mb "s3://${ARTIFACT_BUCKET}" --region "$REGION" >/dev/null
fi
# A content-addressed key makes each deploy a distinct S3 object, which is what causes
# CloudFormation to publish a new function version.
PACKAGE_HASH="$(python3 -c "
import hashlib,sys
print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()[:16])
" "$BUILD_DIR/lambda.zip")"
PACKAGE_KEY="durable-worker/${PACKAGE_HASH}.zip"
aws s3 cp "$BUILD_DIR/lambda.zip" "s3://${ARTIFACT_BUCKET}/${PACKAGE_KEY}" --region "$REGION" >/dev/null

echo "==> Deploying stack ${STACK_NAME}"
aws cloudformation deploy \
  --template-file "$SCRIPT_DIR/deadline-durable-lambda-worker.yaml" \
  --stack-name "$STACK_NAME" \
  --region "$REGION" \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
      FarmId="$FARM_ID" \
      MaxWorkerCount="$MAX_WORKERS" \
      LambdaCodeBucket="$ARTIFACT_BUCKET" \
      LambdaCodeKey="$PACKAGE_KEY"

echo
echo "==> Stack outputs"
aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" --region "$REGION" \
  --query 'Stacks[0].Outputs[].[OutputKey,OutputValue]' --output table
