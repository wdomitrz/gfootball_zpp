mkdir -p eval_logs
mydir=$(pwd)

set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

"$DIR/../docker/build_eval.sh"

docker run -ti gfootball_zpp "$@"

