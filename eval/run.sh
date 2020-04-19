mkdir -p eval_logs
mydir=$(pwd)

docker run \
    -v "$HOME/.config/gcloud/application_default_credentials.json":/tmp/keys/application_default_credentials.json:ro \
    -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/keys/application_default_credentials.json \
    -v "$(pwd)/eval_logs":/eval_logs -v \
    -ti gfootball_eval "$@"
