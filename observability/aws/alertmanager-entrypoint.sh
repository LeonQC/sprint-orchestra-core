#!/bin/sh
# ECS hands secrets over as environment variables; alertmanager.yml reads the webhook from a file.
set -e
if [ -n "$SLACK_WEBHOOK_URL" ]; then
  printf '%s' "$SLACK_WEBHOOK_URL" > /etc/alertmanager/slack_webhook_url
fi
exec /bin/alertmanager --config.file=/etc/alertmanager/alertmanager.yml --storage.path=/alertmanager "$@"
