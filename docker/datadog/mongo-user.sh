#!/bin/sh
# Creates the read-only `datadog` monitoring user in the running logpipe_mongo (mongo-init.js only runs on
# a fresh volume). Needs DD_MONGO_PASSWORD in the environment: set -a; . ./.env; set +a
set -eu
: "${DD_MONGO_PASSWORD:?set DD_MONGO_PASSWORD}"
docker exec -e DD_MONGO_PASSWORD logpipe_mongo sh -c 'mongosh --quiet -u "$MONGO_INITDB_ROOT_USERNAME" \
  -p "$MONGO_INITDB_ROOT_PASSWORD" --authenticationDatabase admin --eval "
db.getSiblingDB(\"admin\").createUser({user: \"datadog\", pwd: process.env.DD_MONGO_PASSWORD, roles: [
  {role: \"read\", db: \"admin\"}, {role: \"clusterMonitor\", db: \"admin\"}, {role: \"read\", db: \"local\"},
  {role: \"read\", db: \"$MONGO_DB\"}]})"'
