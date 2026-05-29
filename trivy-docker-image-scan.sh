#!/bin/bash
# Extract the FINAL FROM image (last FROM statement)
dockerImageName=$(grep "^FROM" Dockerfile | tail -1 | awk '{print $2}' | sed 's/ AS.*//')
echo "Scanning image: $dockerImageName"

docker run --rm \
    -v /var/lib/jenkins/.trivy-cache:/root/.cache/trivy \
    aquasec/trivy:latest image \
    --exit-code 0 --severity HIGH $dockerImageName

docker run --rm \
    -v /var/lib/jenkins/.trivy-cache:/root/.cache/trivy \
    aquasec/trivy:latest image \
    --exit-code 1 --severity CRITICAL $dockerImageName

exit_code=$?
echo "Exit Code : $exit_code"
if [[ "${exit_code}" == 1 ]]; then
    echo "Image scanning failed. Vulnerabilities found"
    exit 1
else
    echo "Image scanning passed. No CRITICAL vulnerabilities found"
fi