curl -s \
  --form-string "token=$PUSHOVER_TOKEN" \
  --form-string "user=$PUSHOVER_USER" \
  --form-string "message=deep-ultrasound analysis complete" \
  https://api.pushover.net/1/messages.json
