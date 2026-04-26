#!/bin/sh
# wait-for-it.sh

set -e

host="$1"
shift
cmd="$@"

until nc -z "$host" 9000; do
  >&2 echo "MinIO is unavailable - sleeping"
  sleep 1
done

>&2 echo "MinIO is up - executing command"
exec $cmd