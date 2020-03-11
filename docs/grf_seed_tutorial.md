[TODO] wstęp
* AI platform
* [https://github.com/google-research/football](https://github.com/google-research/football)
* leaderboard

# Run grf with seed tutorial
1. Install docker - [https://docs.docker.com/install/](https://docs.docker.com/install/)
2. Add yourself to the docker group `usermod -aG docker $USER`
3. relogin/reboot
4. Install `gcloud` - https://cloud.google.com/sdk/install
5. `gcloud auth login`
6. `gcloud config set project <project-name>`
7. `gcloud auth configure-docker`
	* choose Y
8. [TODO] clone our repo
9. [TODO] clone our seed
10. [TODO] use gfootball/gcp/run.sh

# tensorboard
It is possible to view experiment results via tensorboard
* `gcloud auth application-default login`
* `tensorboard --logdir=gs://<bucket_name>`

# Troubleshooting
### Authorization problems
If you run with sudo (i.e. as a different user) it is possible that you receive the following error:
	`
	unauthorized: You don't have the needed permissions to perform this operation,
	and you may have invalid credentials. To authenticate your request, follow the
	steps in:
	https://cloud.google.com/container-registry/docs/advanced-authentication
	`
So running without sudo might help you.

### Changing the region
* `gcloud compute regions list` lists the accessible regions alongside the available resources
* the fifth parameter of the `gcp/run.sh` script is optional - region name with default value `us-central1`
